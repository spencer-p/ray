
from aiohttp.web import Request, Response

import ray._raylet
import ray.dashboard.utils as dashboard_utils
import ray.exceptions
from ray._raylet import NodeID
from ray.dashboard.modules.reporter.utils import HealthChecker

routes = dashboard_utils.DashboardAgentRouteTable


class UnifiedHealth(dashboard_utils.DashboardAgentModule):
    """Health endpoint that unifies all health statuses to a single check.

    This module adds health a singular health check endpoint to the agent to
    check the overall health of the node.
    """

    def __init__(self, dashboard_agent):
        super().__init__(dashboard_agent)
        node_id = (
            NodeID.from_hex(dashboard_agent.node_id)
            if dashboard_agent.node_id
            else None
        )
        self._health_checker = HealthChecker(
            dashboard_agent.gcs_client,
            node_id,
        )

    @routes.get("/api/healthz")
    async def health_check(self, req: Request) -> Response:
        # Check local raylet health.
        try:
            alive = await self._health_checker.check_local_raylet_liveness()
            if alive is False:
                return Response(status=503, text="Local Raylet failed")
        except ray.exceptions.RpcError as e:
            # We only consider the error other than GCS unreachable as raylet failure
            # to avoid false positive.
            # In case of GCS failed, Raylet will crash eventually if GCS is not back
            # within a given time and the check will fail since agent can't live
            # without a local raylet.
            if e.rpc_code not in (
                ray._raylet.GRPC_STATUS_CODE_UNAVAILABLE,
                ray._raylet.GRPC_STATUS_CODE_UNKNOWN,
                ray._raylet.GRPC_STATUS_CODE_DEADLINE_EXCEEDED,
            ):
                return Response(
                    status=503, text=f"Local raylet health check failed: {e}"
                )

        # Check GCS health.
        try:
            gcs_alive = await self._health_checker.check_gcs_liveness()
            if not gcs_alive:
                return Response(status=503, text="GCS health check failed.")
        except Exception as e:
            return Response(status=503, text=f"GCS health check failed: {e}")

        return Response(
            text="success",
            content_type="application/text",
        )

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        return False


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
      # TODO: Check against the /api/local_raylet_health endpoint and the
      # /api/gcs_healthz endpoint. If one of those fail, then we can report bad
      # health.
      if False:
        return Response(status=503, text=f"Health check failed due to: {e}")

      return Response(
          text="success",
          content_type="application/text",
      )

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        return False

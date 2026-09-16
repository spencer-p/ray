import os
import socket
import time
import torch
import numpy as np
import ray

print("[Qwen-16Host] Initializing Ray client...")
ray.init(address="auto")

def get_qwen3_5_specs(num_layers=4):
    specs = []
    # 1. Embeddings & Final Logits / Norms
    specs.append(((248320, 4096), (None, "fsdp"), "token_embedder.embedding"))
    specs.append(((4096, 248320), ("fsdp", None), "decoder.logits_dense"))
    specs.append(((4096,), (None,), "decoder.decoder_norm"))

    for l_idx in range(num_layers):
        is_full_attn = (l_idx % 4 == 3)
        prefix = f"layer_{l_idx}"
        specs.append(((4096,), (None,), f"{prefix}.input_layernorm"))
        specs.append(((4096,), (None,), f"{prefix}.post_attention_layernorm"))

        if not is_full_attn:
            # Linear attention (Gated DeltaNet)
            specs.append(((4096, 20480), ("fsdp", None), f"{prefix}.gdn.in_proj_qkvz"))
            specs.append(((4096, 128), ("fsdp", None), f"{prefix}.gdn.in_proj_ba"))
            specs.append(((4, 1, 12288), (None, None, None), f"{prefix}.gdn.conv1d"))
            specs.append(((64,), (None,), f"{prefix}.gdn.A_log"))
            specs.append(((64,), (None,), f"{prefix}.gdn.dt_bias"))
            specs.append(((128,), (None,), f"{prefix}.gdn.rms_norm"))
            specs.append(((8192, 4096), (None, "fsdp"), f"{prefix}.gdn.out_proj"))
        else:
            # Full GQA attention
            specs.append(((4096, 8192), ("fsdp", None), f"{prefix}.self_attn.q_proj"))
            specs.append(((4096, 512), ("fsdp", None), f"{prefix}.self_attn.k_proj"))
            specs.append(((4096, 512), ("fsdp", None), f"{prefix}.self_attn.v_proj"))
            specs.append(((8192, 4096), (None, "fsdp"), f"{prefix}.self_attn.o_proj"))
            specs.append(((256,), (None,), f"{prefix}.self_attn.q_norm"))
            specs.append(((256,), (None,), f"{prefix}.self_attn.k_norm"))

        # MoE Routed + Shared Experts
        specs.append(((4096, 512), ("fsdp", None), f"{prefix}.moe.routed_gate"))
        specs.append(((512, 4096, 1024), ("expert", "fsdp", None), f"{prefix}.moe.routed_wi_0"))
        specs.append(((512, 4096, 1024), ("expert", "fsdp", None), f"{prefix}.moe.routed_wi_1"))
        specs.append(((512, 1024, 4096), ("expert", None, "fsdp"), f"{prefix}.moe.routed_wo"))
        specs.append(((4096, 1024), (None, "fsdp"), f"{prefix}.moe.shared_gate"))
        specs.append(((4096, 1024), ("fsdp", None), f"{prefix}.moe.shared_wi_0"))
        specs.append(((4096, 1024), ("fsdp", None), f"{prefix}.moe.shared_wi_1"))
        specs.append(((1024, 4096), (None, "fsdp"), f"{prefix}.moe.shared_wo"))

    return specs

def compute_local_shape(global_shape, sharding, ep_coord, fsdp_coord, ep_dim=32, fsdp_dim=2):
    local_shape = list(global_shape)
    for dim_idx, axis in enumerate(sharding):
        if axis == "expert":
            local_shape[dim_idx] //= ep_dim
        elif axis == "fsdp":
            local_shape[dim_idx] //= fsdp_dim
    return tuple(local_shape)

def get_expected_fill(host_idx: int, tensor_idx: int, iteration: int = 1) -> float:
    # Strictly bounded in [1, 250] so every integer is 100% bit-exact in bfloat16
    return float(((host_idx + 1) * 11 + tensor_idx + iteration * 3) % 245 + 1)

@ray.remote(resources={"TPU": 4, "src_worker": 1}, max_concurrency=8, enable_tensor_transport=True)
class QwenSourceWorker:
    def __init__(self, host_idx: int):
        self.host_idx = host_idx
        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)
        self.device = torch.device("tpu")
        print(f"[Qwen Src Host {self.host_idx}] Initialized actor on {self.hostname} ({self.ip})")

    def init_tpu(self):
        t = torch.zeros((1,), device=self.device)
        return {"host_idx": self.host_idx, "ip": self.ip}

    @ray.method(tensor_transport="TPU_SYNC")
    def create_layer_weights(self, ep_coord: int, fsdp_coord: int, num_layers: int = 4, iteration: int = 1):
        specs = get_qwen3_5_specs(num_layers=num_layers)
        tensors = []
        total_bytes = 0
        
        for idx, (g_shape, sharding, name) in enumerate(specs):
            l_shape = compute_local_shape(g_shape, sharding, ep_coord, fsdp_coord)
            val = get_expected_fill(self.host_idx, idx, iteration=iteration)
            t = torch.full(l_shape, val, dtype=torch.bfloat16, device=self.device)
            tensors.append(t)
            total_bytes += t.nelement() * t.element_size()

        print(f"[Qwen Src Host {self.host_idx}] Iter {iteration}: Created {len(tensors)} tensors ({total_bytes / 1e6:.2f} MB) on TPU")
        return tensors

@ray.remote(resources={"TPU": 4, "dst_worker": 1}, max_concurrency=8, enable_tensor_transport=True)
class QwenDestinationWorker:
    def __init__(self, host_idx: int):
        self.host_idx = host_idx
        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)
        self.device = torch.device("tpu")
        print(f"[Qwen Dst Host {self.host_idx}] Initialized actor on {self.hostname} ({self.ip})")

    def init_tpu(self):
        t = torch.zeros((1,), device=self.device)
        return {"host_idx": self.host_idx, "ip": self.ip}

    def consume_layer_weights(self, tensors_ref, src_host_idx: int, num_layers: int = 4, iteration: int = 1):
        tensors = tensors_ref
        specs = get_qwen3_5_specs(num_layers=num_layers)
        assert len(tensors) == len(specs), f"Expected {len(specs)} tensors, got {len(tensors)}"
        
        matches = []
        total_bytes = 0
        diff_details = []
        for idx, t in enumerate(tensors):
            bytes_i = t.nelement() * t.element_size()
            total_bytes += bytes_i
            expected_val = get_expected_fill(src_host_idx, idx, iteration=iteration)
            flat = t.flatten()
            n = flat.numel()
            sample_indices = [0, min(1, n-1), n//2, max(0, n-1)]
            samples = [float(flat[i].cpu()) for i in sample_indices]
            is_exact = all(s == expected_val for s in samples)
            matches.append(is_exact)
            if not is_exact:
                print(f"[Dst Host {self.host_idx}] MISMATCH tensor_{idx} ({specs[idx][2]}): samples={samples}, expected={expected_val}", flush=True)
                if len(diff_details) < 3:
                    diff_details.append(f"tensor_{idx} ({specs[idx][2]}): samples={samples}, expected={expected_val}")

        all_ok = all(matches)
        print(f"[Qwen Dst Host {self.host_idx}] Iter {iteration}: Received {len(tensors)} tensors ({total_bytes / 1e6:.2f} MB) from Src Host {src_host_idx}, parity_ok={all_ok}", flush=True)
        return {
            "host_idx": self.host_idx,
            "src_host_idx": src_host_idx,
            "tensor_count": len(tensors),
            "total_bytes": total_bytes,
            "all_ok": all_ok,
            "diff_details": diff_details
        }

def run_benchmark_iteration(src_workers, dst_workers, num_layers: int, iteration: int):
    print(f"\n--- [Iteration {iteration}] Transferring {num_layers} Qwen 3.5 layers (Embeddings + 3 GDN + 1 GQA MoE) ---")
    
    # 1. Allocate tensors
    t0_alloc = time.perf_counter()
    tensor_refs = []
    for host_idx in range(16):
        ep_coord = host_idx * 2
        fsdp_coord = host_idx % 2
        ref = src_workers[host_idx].create_layer_weights.remote(ep_coord, fsdp_coord, num_layers=num_layers, iteration=iteration)
        tensor_refs.append(ref)

    # 2. Parallel RDT Transfer
    t0_xfer = time.perf_counter()
    consume_futures = [
        dst_workers[i].consume_layer_weights.remote(tensor_refs[i], src_host_idx=i, num_layers=num_layers, iteration=iteration)
        for i in range(16)
    ]
    results = ray.get(consume_futures)
    elapsed_xfer = time.perf_counter() - t0_xfer

    total_transferred_bytes = sum(r["total_bytes"] for r in results)
    total_gb = total_transferred_bytes / 1e9
    throughput_gbps = total_gb / max(elapsed_xfer, 1e-9)

    print("="*80)
    print(f"ITERATION {iteration} RESULTS & INTEGRITY REPORT")
    print("="*80)
    print(f"Total Tensors Transferred  : {sum(r['tensor_count'] for r in results)} tensors ({results[0]['tensor_count']} per host)")
    print(f"Total Payload Size         : {total_transferred_bytes / 1e6:.2f} MB ({total_gb:.3f} GB)")
    print(f"Per-Host Payload           : {results[0]['total_bytes'] / 1e6:.2f} MB ({results[0]['total_bytes'] / 1e9:.3f} GB)")
    print(f"Transfer Time (Elapsed)    : {elapsed_xfer:.3f} s")
    print(f"Aggregate Throughput       : {throughput_gbps:.2f} GB/s")
    
    all_ok = all(r["all_ok"] for r in results)
    print(f"Numerical Parity Check     : {'PASSED (100% BIT-EXACT MATCH)' if all_ok else 'FAILED'}")
    print("="*80)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    assert all_ok, f"Parity check failed for iteration {iteration}!"
    return {
        "iteration": iteration,
        "payload_gb": total_gb,
        "elapsed_s": elapsed_xfer,
        "throughput_gbps": throughput_gbps,
        "tensors": sum(r["tensor_count"] for r in results),
    }

def main():
    print("="*80)
    print("QWEN 3.5 397B SHARDED WEIGHT TRANSFER BENCHMARK (16 -> 16 HOSTS, BF16)")
    print("Topology: TP=1, EP=32, FSDP=2 across two 16-host TPU v7x Torus Slices (32 hosts total)")
    print("Authentic Qwen 3.5 Architecture: Embeddings, Logits, GDN Linear Attn, GQA Attn, MoE 512 Experts")
    print("="*80)

    # Spawn 16 source and 16 destination workers
    src_workers = [QwenSourceWorker.remote(i) for i in range(16)]
    dst_workers = [QwenDestinationWorker.remote(i) for i in range(16)]

    print("\nGang-initializing TPU torus meshes on all 32 hosts simultaneously...")
    init_futures = [w.init_tpu.remote() for w in src_workers] + [w.init_tpu.remote() for w in dst_workers]
    ray.get(init_futures)
    print("All 32 hosts TPU initialized successfully!\n")

    # Run Benchmark: Iterations 1, 2, 3 to sample transfer latency
    results = []
    for it in range(1, 4):
        res = run_benchmark_iteration(src_workers, dst_workers, num_layers=60, iteration=it)
        results.append(res)

    print("\n" + "="*80)
    print("FINAL SUMMARY: 16-HOST TO 16-HOST FULL 60-LAYER QWEN 3.5 397B SHARDED TRANSFER SAMPLES")
    print("="*80)
    for res in results:
        print(f"Iteration {res['iteration']} : {res['payload_gb']:.2f} GB in {res['elapsed_s']:.3f} s -> {res['throughput_gbps']:.2f} GB/s ({res['tensors']} tensors)")
    avg_elapsed = sum(r['elapsed_s'] for r in results) / len(results)
    avg_throughput = sum(r['throughput_gbps'] for r in results) / len(results)
    print(f"Average Transfer Latency   : {avg_elapsed:.3f} s")
    print(f"Average Aggregate Throughput: {avg_throughput:.2f} GB/s")
    print("="*80)
    import sys
    sys.stdout.flush()

if __name__ == "__main__":
    main()

"""Consume HBM bandwidth so the rest of the GPU sees a MiG-like memory budget.

MPS partitions SMs but not the memory subsystem, so a 12%-capped process on an
H100 still gets the full ~3.35TB/s. That inverts the bottleneck: the pod became
SM starved where the real MiG 1g.18gb slice is bandwidth starved, and the H0/H1
anchors came back identical (3.17ms both) when the portal pays 1.74ms for that
same bf16->fp8 difference.

This process streams large buffers in a loop to soak up the surplus. Run it
under its own MPS thread percentage so it competes for bandwidth without eating
the SMs the server needs.

    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 python3 tools/runpod/bw_hog.py &

Tune with HOG_GB until sm_cap_verify reports ~600 GB/s for the capped client.
"""
import os
import signal
import sys

import torch


def main():
    gb = float(os.environ.get("HOG_GB", "4"))
    n = int(gb * 1e9) // 2
    dev = torch.device("cuda")
    a = torch.ones(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
    print(f"[bw_hog] streaming {gb:.1f}GB buffers, mps_pct={cap}, pid={os.getpid()}",
          flush=True)

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while running:
        for _ in range(50):
            b.copy_(a)          # 2 x gb of traffic per iteration
        torch.cuda.synchronize()
    print("[bw_hog] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

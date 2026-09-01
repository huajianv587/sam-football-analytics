# GPU acceptance notes

This directory keeps small, non-secret payload examples and the evidence summary for remote inference. Source videos and generated artifacts are excluded from Git.

## Initial smoke test

A synthetic three-second, 15 FPS clip with two first-frame rectangles verified the complete SAM 2.1 Hiera Large path: Slurm submission, CUDA model load, 45 propagated frames, two persistent IDs, RLE masks, track JSON, metrics JSON, and a decodable 720p H.264 foreground video.

The first remote attempt exposed a home-directory quota issue when EasyOCR downloaded its model. The runtime bootstrap now redirects Conda, pip, Torch, EasyOCR, Triton, and TorchInductor state into the configured scratch runtime.

## Full acceptance test

The final run used a real continuous 30-second clip normalized to 1280 x 720 at 15 FPS, with 11 first-frame subjects.

- Slurm job: `33152`
- State / exit: `COMPLETED` / `0:0`
- Slurm wall time: 3 minutes 9 seconds
- Worker time: 183.40 seconds
- Frames / objects: 450 / 11
- Propagation: 142.71 seconds, 3.15 FPS
- Effective pipeline throughput: 2.45 FPS
- GPU utilization: 69.29% average, 100% peak
- NVIDIA memory-used peak: 11,375 MB
- PyTorch allocated / reserved peak: 10,487.7 / 11,048 MB
- ID retention: 11/11 until image-boundary exit or video end
- Output validation: gzip/JSON parsed; H.264 MP4 decoded as 450 frames at 1280 x 720 and 15 FPS

## Performance finding

The dominant early bottleneck was not the A40. The first RLE implementation iterated over every full-resolution mask pixel in Python, forcing the GPU to wait between frames. Replacing it with NumPy run-boundary detection reduced 11-mask RLE encoding to milliseconds and increased real 11-object propagation to approximately 3.15 FPS.

Full VOS compilation was also tested on the installed PyTorch/CUDA stack. It increased first-run latency and reduced measured throughput, so the validated configuration keeps eager video propagation with BF16, TF32, the SAM 2 CUDA extension, GPU-resident state, batched mask transfer, vectorized RLE, and non-overlapping memory masks.

# resnet.s

ResNet-18 image classification in hand-written AArch64 assembly for Apple
Silicon.

The program calls no C code. It calls no libc functions. It does all I/O with
raw Darwin syscalls (`svc #0x80`). It reads time from the ARM hardware timer.
It contains its own NEON matrix kernel, im2col, pooling, `exp`, softmax and
top-5.

There is no conversion step. The program reads the original Hugging Face file
(`model.safetensors` from `timm/resnet18.tv_in1k`, the torchvision weights) as
it is. At startup it parses the header in assembly, folds every BatchNorm into
the conv before it, and repacks the weights for the matrix kernel.

```
$ ./resnet images/n01440764_0.ppm
[resnet.s] load 11.53 ms, forward 41.24 ms
 96.18%  tench, Tinca tinca
  0.48%  Staffordshire bullterrier, Staffordshire bull terrier
  0.36%  coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
  0.36%  hippopotamus, hippo, river horse, Hippopotamus amphibius
  0.30%  artichoke, globe artichoke
```

## Numbers (Apple M1, 8 GB)

Time for one forward pass of one 224x224 image, fp32. Median of 30 runs on
each of 3 images (`tools/bench.py`).

| | Time per image | Note |
|---|---|---|
| PyTorch, CPU, 4 threads (default) | 17.3 ms | Apple Accelerate, uses the AMX unit |
| PyTorch, CPU, 1 thread | 22.8 ms | Apple Accelerate, uses the AMX unit |
| **resnet.s** | **39.4 ms** | one thread, NEON only |

- Machine code: 8,232 bytes. Source: 1,695 lines of assembly.
- Startup (parse the header, fold BatchNorm, repack 46.8 MB of weights):
  about 12 ms. A whole run takes about 0.05 s.
- The output is exact to fp32 rounding. On 30 test images the top-5 classes
  match PyTorch in the same order. The largest logit difference is 1.6e-5
  (see below).

**Why PyTorch is faster.** This PyTorch build does its matrix math with
Apple Accelerate (`BLAS_INFO=accelerate`). Accelerate uses the AMX matrix
unit of the M1, not NEON. With one thread, a 512x512 matrix product in
PyTorch runs at about 1,200 GFLOPS. The NEON peak of one M1 P-core is about
102 GFLOPS (4 FMA pipes x 4 floats x 2 x 3.2 GHz). ResNet-18 needs 3.63
GFLOP, so even 100% of the NEON peak takes about 35 ms. resnet.s runs the
whole network at about 92 GFLOPS, which is 90% of that peak. The matrix
kernel alone runs at 93 to 97 GFLOPS on most layers.

## Build and run

You need macOS on Apple Silicon and the Xcode command line tools.

```bash
make weights
```

```bash
make
```

```bash
uv run python tools/prep.py photo.jpg photo.ppm
```

```bash
./resnet photo.ppm
```

`make weights` downloads `model.safetensors` (47 MB) from the
[timm/resnet18.tv_in1k repository](https://huggingface.co/timm/resnet18.tv_in1k)
with `curl`. Nothing changes it. The program reads it and `labels.txt` (the
1000 ImageNet class names, one per line) from the current folder.

The input must be a binary PPM (P6), 8-bit RGB, exactly 224x224.
`tools/prep.py` makes one from any image with the standard torchvision steps:
resize the shorter side to 256 (bilinear), then center crop to 224. The
program does the normalization (mean 0.485 0.456 0.406, std 0.229 0.224 0.225).
Python is only needed for the tools. It is never needed to run the program.

Options:

- `-d N` writes the raw float activations (f32, NHWC) after debug point `N`
  to stdout, then stops. 0 is the normalized input, 1 the stem, 2 the max pool,
  3 to 10 the 8 basic blocks, 11 the average pool, 12 the logits.
- `-b N` runs the forward pass `N` times and prints each time (benchmark).

## Verify

The tools need `uv`. They only check the result.

```bash
uv run python tools/fetch_images.py
```

```bash
uv run python tools/verify.py
```

`fetch_images.py` streams the Imagenette validation set (a 10-class subset of
ImageNet) and keeps 3 images per class, 30 in all, as PPM files in `images/`.
`verify.py` runs `./resnet` on each image and compares it with timm
`resnet18.tv_in1k` in PyTorch (fp32), loaded from the same `model.safetensors`:

- The top-5 class indices must match exactly, in the same order.
- The 5 printed lines must name the same classes with the same probabilities
  (to 0.01%).

The current result is `ALL PASS`. The largest logit difference is 1.57e-5.

To compare every stage of the network:

```bash
uv run python tools/ref.py images/n01440764_0.ppm --check
```

This prints the largest difference at each of the 13 debug points. The input is
bit-exact. Later stages differ by about 1e-6 of their largest value.

To time against PyTorch:

```bash
uv run python tools/bench.py
```

## How it works

| File | Contents |
|---|---|
| `src/main.s` | Argument parsing, `open`/`lseek`/`mmap` of the files, the PPM reader, timing (`cntvct_el0`), output of the top 5 with labels. |
| `src/load.s` | Startup. Parses the safetensors JSON header to find the 102 tensors it uses. Folds each BatchNorm into its conv and repacks all weights for the kernel. |
| `src/model.s` | The forward pass: normalization, im2col, the conv driver, max pool, the 8 basic blocks, average pool, fc, and the `-d` dump. |
| `src/gemm.s` | The matrix kernel: a 12x8 NEON tile, a 4x16 tile for the last rows, and bias, residual add and ReLU when it stores. |
| `src/math.s` | `expf` (range reduction and a degree-6 polynomial), softmax, top-5. |
| `tools/` | Python helpers, run with `uv`: `prep.py`, `fetch_images.py`, `labels.py`, `ref.py`, `verify.py`, `bench.py`. |

Activations are NHWC floats. Every conv, and the fc layer, is one matrix
product C = A x B:

- **A** is the im2col matrix. Each row is one output pixel. Each column is one
  (ky, kx, input channel). It is written in panels of 12 rows, so that the
  kernel reads 12 values in order for each k. For 4 or more channels, im2col
  reads 4 channels of 12 pixels at a time and turns them with three NEON 4x4
  transposes. The stem has 3 channels, so the normalized image has a zero
  border of 3 pixels, and one row of the 7x7 kernel is 21 floats in a row.
- **B** is the weights, with BatchNorm folded in:
  `w' = w * gamma / sqrt(var + eps)` and `b' = beta - mean * gamma / sqrt(var + eps)`.
  It is packed at startup in panels of 8 output channels, into memory that the
  program maps for itself (48 MB). The checkpoint stores conv weights as
  `[out][in][ky][kx]`.
- **C** in row-major order is NHWC, so it is the input of the next layer.

The kernel keeps a 12x8 tile of C in 24 NEON registers. For each k it loads
12 values of A and 8 of B and does 24 `fmla` by element. 24 independent sums
are enough to hide the 4-cycle FMA latency on the 4 FMA pipes. The last panel
of each conv has only 4 real rows (1 for layer4), so it uses a 4x16 tile that
costs a third as much. When it stores, the kernel adds the bias, then the
residual (if any), then applies ReLU (if any). A basic block is
`t = relu(conv1(x))` and `x = relu(conv2(t) + x)`, where `x` is first replaced
by the downsample conv if the block has one. The second conv writes over `x`
in place.

All other buffers (activations, the im2col matrix) are zero-fill sections in
the binary, so there is no allocator.

## Honest limits

- **libSystem is linked but not used.** macOS on Apple Silicon kills static
  binaries at launch (SIGKILL), so the binary must name `libSystem.dylib` for
  the loader. `nm -u resnet` is empty: the program imports zero symbols.
- **Slower than PyTorch on this Mac.** PyTorch uses the AMX unit through
  Accelerate. resnet.s uses NEON only, one thread, fp32.
- **The input must already be 224x224.** Resize and crop are in Python
  (`tools/prep.py`), not in assembly.
- **More memory than the weights alone.** The packed copy of the weights adds
  48 MB to the 47 MB file mapping. The peak resident size is about 108 MB.
- **im2col writes the whole matrix to memory** (up to 7.4 MB for the stem)
  before the kernel reads it. It takes about 1.7 ms of each forward pass.
- **The first run after the Mac is idle is slower** (for example a 17 ms load
  and a 57 ms forward pass), because the CPU clock rises only under load.
- **Two classes have the same name.** In `labels.txt` both 134 (the bird) and
  517 (the machine) are "crane", as in ImageNet.

# resnet.s

ResNet-18 image classification in hand-written AArch64 assembly for Apple
Silicon. No libc calls: all I/O is raw Darwin syscalls (`svc #0x80`). It reads
the original Hugging Face `model.safetensors` (`timm/resnet18.tv_in1k`) as it
is, with no conversion step.

```
$ ./resnet images/n01440764_0.ppm
[resnet.s] load 11.53 ms, forward 41.24 ms
 96.18%  tench, Tinca tinca
  0.48%  Staffordshire bullterrier, Staffordshire bull terrier
  0.36%  coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
  0.36%  hippopotamus, hippo, river horse, Hippopotamus amphibius
  0.30%  artichoke, globe artichoke
```

## Numbers (Apple M1)

One 224x224 image, fp32, median of 30 runs on 3 images (`tools/bench.py`).

| | Time per image |
|---|---|
| PyTorch, CPU, 4 threads | 17.3 ms |
| PyTorch, CPU, 1 thread | 22.8 ms |
| **resnet.s**, 1 thread, NEON | **39.4 ms** |

- 8,232 bytes of machine code, 1,695 lines of assembly. Startup takes about 12 ms.
- The top-5 classes match PyTorch on 30 test images. The largest logit difference is 1.6e-5.
- PyTorch uses Apple Accelerate, which runs on the AMX matrix unit, not NEON.
  resnet.s runs at about 92 GFLOPS, which is 90% of the NEON peak of one core.

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

The input must be a 224x224 binary PPM. `tools/prep.py` makes one from any
image (resize to 256, center crop to 224). Python is only for the tools.

- `-d N` writes the raw activations (f32, NHWC) after debug point `N` to
  stdout: 0 input, 1 stem, 2 max pool, 3 to 10 the 8 blocks, 11 average pool, 12 logits.
- `-b N` runs the forward pass `N` times and prints each time.

## How it works

| File | Contents |
|---|---|
| `src/main.s` | Arguments, `mmap` of the files, PPM reader, timing, top-5 output. |
| `src/load.s` | Parses the safetensors header, folds BatchNorm into each conv, packs the weights. |
| `src/model.s` | Forward pass: normalization, im2col, convs, pooling, the 8 basic blocks, fc. |
| `src/gemm.s` | The NEON matrix kernel. |
| `src/math.s` | `expf`, softmax, top-5. |

Every conv and the fc layer is im2col plus one matrix product. Activations are
NHWC. The kernel keeps a 12x8 tile in 24 registers and does 24 `fmla` by
element for each k. A 4x16 tile handles the last rows. Bias, the residual add
and ReLU happen when the tile is stored. All buffers are zero-fill sections, so
there is no allocator.

## Limits

- libSystem is linked only so the loader accepts the binary. `nm -u resnet` is empty.
- One thread, fp32, NEON only. PyTorch is faster here because it uses AMX.
- The input must already be 224x224. Resize and crop are in Python.
- Peak memory is about 108 MB: the file mapping plus a 48 MB packed copy of the weights.

## License

MIT. See [LICENSE](LICENSE).

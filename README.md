# HW/SW Co-Design: Vision Transformer MLP Kernel on Xtensa LX5

Hardware/software co-design project for the TU Dresden Vodafone Chair HW/SW
Co-Design Lab (SS 2026). Accelerates the Feed-Forward Network (MLP) block of
a Vision Transformer encoder on a Tensilica Xtensa LX5 core using custom TIE
instruction set extensions and FLIX (multi-slot bundled instructions).

## Result

**194.0x speedup** on the MLP kernel (`mlp_C`), from 552,087 cycles at
baseline down to 2,846 cycles, using 16-lane SIMD, FLIX multi-slot dispatch,
and a sliding-window register cache, for a 16.7% core area overhead.

| Build | `mlp_C` cycles | Speedup |
|---|---:|---:|
| Baseline (scalar C, `-O0`) | 552,087 | 1.00x |
| Pipelined FLIX (SIMD + FLIX, no cache) | 3,786 | 145.8x |
| Cache-based FLIX (final) | 2,846 | **194.0x** |

## What this is

The task: an MLP block (`FC1 -> ReLU -> FC2`, 64 -> 128 -> 64 int8) is the
compute bottleneck of a ViT encoder layer. Starting from a plain scalar C
implementation, the workload is profiled on the Xtensa ISS, and hardware
acceleration is added as custom TIE instructions, paired with the C kernel
changes needed to use them.

The final design (`src/mlp.c` + `src/Flix.tie`) uses:

- **`MAC_SIMD`**: a 16-lane, single-cycle signed 8x8 multiply-accumulate
  over a 128-bit input/weight buffer pair.
- **FLIX bundling**: a 3-slot, 64-bit instruction format that co-issues a
  cache-select, a weight load, and a MAC in a single cycle.
- **Sliding-window register cache** (`mlp_cache0..7`, 8x 128-bit): the
  optimization on top of the SIMD/FLIX design. The input (FC1) or hidden
  (FC2) activation vector is identical for every neuron in a layer, but the
  pre-cache design still reloaded it from memory once per neuron anyway.
  `MLP_FILL_Cx` loads it into registers once per layer instead;
  `MLP_SEL_Cx` then reads it with zero further memory access. Isolated
  effect: loads drop from 2,693 (pipelined FLIX, no cache) to 1,681 (final),
  about 37.6% fewer, on top of the much larger reduction SIMD and FLIX
  already deliver (159,431 loads in the plain scalar baseline).
- **Weight-pointer elimination**: `W1`/`W2` are row-major and contiguous, so
  after streaming a full weight row the pointer has already auto-incremented
  to the next row. A redundant per-neuron pointer reset is removed.

With the kernel this fast, the bottleneck shifts: fixed `_ResetHandler`/
C-runtime startup (3,906 cycles, constant across builds) is under 1% of the
552,087-cycle baseline but 43.4% of the final 8,996-cycle total runtime.

## Repo layout

```
src/
  mlp.c            final kernel (SIMD + FLIX + sliding-window cache)
  mlp_baseline.c   unoptimized scalar C reference
  mlp.h            problem-size constants (IN_SIZE=64, HIDDEN=128, OUT_SIZE=64)
  Flix.tie         TIE instruction set extension source (custom ISA)
report/
  hwswcd_report.pdf   full project report (LaTeX)
  hwswcd_report.tex   report source
```

`weights_w1.h`, `weights_w2.h`, `biases.h`, `activations.h`, and
`reference_output.h` (weight/bias data and the golden reference output) are
omitted since they are lab-provided data files, not part of the design.

## Toolchain

Built and profiled with Xtensa Xplorer / TDK (RF-2016.4) targeting the
`hwsw_codesign2019_aligned` Xtensa LX5 configuration, verified against a
golden reference on the Xtensa ISS (`xt-run`), which prints `Results match.`
on a successful run for every build listed above.


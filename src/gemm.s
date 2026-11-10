// The one matrix kernel: C = A * B + bias, then an optional residual add
// and an optional ReLU. Every conv (after im2col) and fc use it.
//
// A is packed in panels of 12 rows: A[M/12][Kp][12]. Each row is one output
// pixel. B is packed in panels of 8 columns: B[N/8][Kp][8]. Each column is
// one output channel. C is row-major [M][N], which is NHWC.
//
// Each step computes a 12x8 tile of C in 24 registers (v8-v31). For each k
// it loads 12 values of A and 8 values of B, and does 24 fmla by element.
.include "macros.inc"

.text
.p2align 2

// One k step: 12 rows x 8 columns.
.macro KSTEP
    ld1     {v0.4s, v1.4s, v2.4s}, [x10], #48
    ld1     {v3.4s, v4.4s}, [x11], #32
    fmla    v8.4s,  v3.4s, v0.s[0]
    fmla    v9.4s,  v4.4s, v0.s[0]
    fmla    v10.4s, v3.4s, v0.s[1]
    fmla    v11.4s, v4.4s, v0.s[1]
    fmla    v12.4s, v3.4s, v0.s[2]
    fmla    v13.4s, v4.4s, v0.s[2]
    fmla    v14.4s, v3.4s, v0.s[3]
    fmla    v15.4s, v4.4s, v0.s[3]
    fmla    v16.4s, v3.4s, v1.s[0]
    fmla    v17.4s, v4.4s, v1.s[0]
    fmla    v18.4s, v3.4s, v1.s[1]
    fmla    v19.4s, v4.4s, v1.s[1]
    fmla    v20.4s, v3.4s, v1.s[2]
    fmla    v21.4s, v4.4s, v1.s[2]
    fmla    v22.4s, v3.4s, v1.s[3]
    fmla    v23.4s, v4.4s, v1.s[3]
    fmla    v24.4s, v3.4s, v2.s[0]
    fmla    v25.4s, v4.4s, v2.s[0]
    fmla    v26.4s, v3.4s, v2.s[1]
    fmla    v27.4s, v4.4s, v2.s[1]
    fmla    v28.4s, v3.4s, v2.s[2]
    fmla    v29.4s, v4.4s, v2.s[2]
    fmla    v30.4s, v3.4s, v2.s[3]
    fmla    v31.4s, v4.4s, v2.s[3]
.endm

// Store one row of the tile (registers a, b): add the bias (v5, v6), add
// the residual row at x15 if x7 != 0, apply ReLU if x8 != 0.
.macro STOREROW a, b
    fadd    \a\().4s, \a\().4s, v5.4s
    fadd    \b\().4s, \b\().4s, v6.4s
    cbz     x7, 1f
    ld1     {v0.4s, v1.4s}, [x15]
    add     x15, x15, x9
    fadd    \a\().4s, \a\().4s, v0.4s
    fadd    \b\().4s, \b\().4s, v1.4s
1:  cbz     x8, 2f
    fmax    \a\().4s, \a\().4s, v7.4s
    fmax    \b\().4s, \b\().4s, v7.4s
2:  st1     {\a\().4s, \b\().4s}, [x14]
    add     x14, x14, x9
.endm

// gemm(x0 = C, x1 = A, x2 = B, x3 = bias, x4 = number of 12-row panels,
//      x5 = N, x6 = Kp, x7 = residual [M][N] or 0, x8 = 1 for ReLU)
// N must be a multiple of 8 and Kp a multiple of 4. C and the residual
// must have room for all panel rows. The residual may be C itself. Leaf.
.globl _gemm
_gemm:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    lsl     x9, x5, #2               // bytes in one row of C
    movi    v7.16b, #0               // for ReLU
.Lrow:
    mov     x11, x2                  // B panels follow each other
    mov     x13, #0                  // byte offset of the 8 columns
.Lcol:
    mov     x10, x1
    mov     x12, x6
.irp r, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi    v\r\().16b, #0
.endr
.Lk:
    KSTEP
    KSTEP
    KSTEP
    KSTEP
    subs    x12, x12, #4
    b.ne    .Lk
    add     x14, x0, x13             // C tile
    add     x15, x7, x13             // residual tile
    add     x16, x3, x13
    ld1     {v5.4s, v6.4s}, [x16]    // bias for the 8 columns
    STOREROW v8, v9
    STOREROW v10, v11
    STOREROW v12, v13
    STOREROW v14, v15
    STOREROW v16, v17
    STOREROW v18, v19
    STOREROW v20, v21
    STOREROW v22, v23
    STOREROW v24, v25
    STOREROW v26, v27
    STOREROW v28, v29
    STOREROW v30, v31
    add     x13, x13, #32
    cmp     x13, x9
    b.lo    .Lcol
    mov     x1, x10                  // next A panel
    add     x17, x9, x9, lsl #1
    lsl     x17, x17, #2             // 12 rows of C
    add     x0, x0, x17
    cbz     x7, 1f
    add     x7, x7, x17
1:  subs    x4, x4, #1
    b.ne    .Lrow
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

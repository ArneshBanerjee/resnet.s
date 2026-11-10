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

// One k step of the tail kernel: 4 rows x 16 columns (two B panels).
.macro TSTEP
    ldr     q0, [x10], #48           // the first 4 of the 12 rows
    ld1     {v1.4s, v2.4s}, [x11], #32
    ld1     {v3.4s, v4.4s}, [x17], #32
    fmla    v8.4s,  v1.4s, v0.s[0]
    fmla    v9.4s,  v2.4s, v0.s[0]
    fmla    v10.4s, v3.4s, v0.s[0]
    fmla    v11.4s, v4.4s, v0.s[0]
    fmla    v12.4s, v1.4s, v0.s[1]
    fmla    v13.4s, v2.4s, v0.s[1]
    fmla    v14.4s, v3.4s, v0.s[1]
    fmla    v15.4s, v4.4s, v0.s[1]
    fmla    v16.4s, v1.4s, v0.s[2]
    fmla    v17.4s, v2.4s, v0.s[2]
    fmla    v18.4s, v3.4s, v0.s[2]
    fmla    v19.4s, v4.4s, v0.s[2]
    fmla    v20.4s, v1.4s, v0.s[3]
    fmla    v21.4s, v2.4s, v0.s[3]
    fmla    v22.4s, v3.4s, v0.s[3]
    fmla    v23.4s, v4.4s, v0.s[3]
.endm

// Store one 16-column row of the tail tile (a, b, c, d): bias in v24-v27.
.macro TROW a, b, c, d
    fadd    \a\().4s, \a\().4s, v24.4s
    fadd    \b\().4s, \b\().4s, v25.4s
    fadd    \c\().4s, \c\().4s, v26.4s
    fadd    \d\().4s, \d\().4s, v27.4s
    cbz     x7, 1f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x15]
    add     x15, x15, x9
    fadd    \a\().4s, \a\().4s, v0.4s
    fadd    \b\().4s, \b\().4s, v1.4s
    fadd    \c\().4s, \c\().4s, v2.4s
    fadd    \d\().4s, \d\().4s, v3.4s
1:  cbz     x8, 2f
    fmax    \a\().4s, \a\().4s, v7.4s
    fmax    \b\().4s, \b\().4s, v7.4s
    fmax    \c\().4s, \c\().4s, v7.4s
    fmax    \d\().4s, \d\().4s, v7.4s
2:  st1     {\a\().4s, \b\().4s, \c\().4s, \d\().4s}, [x14]
    add     x14, x14, x9
.endm

// gemm(x0 = C, x1 = A, x2 = B, x3 = bias, x4 = M (rows), x5 = N, x6 = Kp,
//      x7 = residual [M][N] or 0, x8 = 1 for ReLU)
// N must be a multiple of 8 and Kp a multiple of 4. A has M rounded up to
// 12 rows. C and the residual must have room for that many rows. The
// residual may be C itself. When the last panel has at most 4 real rows
// and N is a multiple of 16, it uses a 4x16 kernel, so it costs 1/3 of a
// full panel. Leaf.
.globl _gemm
_gemm:
    stp     d8, d9, [sp, #-80]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    str     x19, [sp, #64]
    lsl     x9, x5, #2               // bytes in one row of C
    movi    v7.16b, #0               // for ReLU
    add     x4, x4, #11
    mov     x17, #12
    udiv    x16, x4, x17             // panels
    msub    x4, x16, x17, x4         // 11 - (empty rows in the last panel)
    mov     x19, #0                  // 1: use the tail kernel
    cmp     x4, #3                   // at most 4 real rows
    b.hi    1f
    tst     x5, #15
    b.ne    1f
    mov     x19, #1
    sub     x16, x16, #1
1:  mov     x4, x16                  // full panels
    cbz     x4, .Ltail
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

.Ltail:
    cbz     x19, .Ldone
    mov     x11, x2
    mov     x13, #0                  // byte offset of the 16 columns
.Ltcol:
    mov     x10, x1
    mov     x12, x6
    add     x17, x11, x6, lsl #5     // the second B panel
.irp r, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
    movi    v\r\().16b, #0
.endr
.Ltk:
    TSTEP
    TSTEP
    TSTEP
    TSTEP
    subs    x12, x12, #4
    b.ne    .Ltk
    mov     x11, x17                 // the next pair of B panels
    add     x14, x0, x13
    add     x15, x7, x13
    add     x16, x3, x13
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x16]
    TROW    v8, v9, v10, v11
    TROW    v12, v13, v14, v15
    TROW    v16, v17, v18, v19
    TROW    v20, v21, v22, v23
    add     x13, x13, #64
    cmp     x13, x9
    b.lo    .Ltcol
.Ldone:
    ldr     x19, [sp, #64]
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #80
    ret

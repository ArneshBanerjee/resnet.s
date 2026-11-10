// Softmax and top-5. No libm: exp() is range reduction plus a degree-6
// polynomial (the same as gpt2.s).
.include "macros.inc"

.text
.p2align 2

// s0 = exp(s0). Leaf. Clobbers s1, s2, w9, w10.
expf:
    FCONST  s1, 0x42b00000          // 88.0
    fminnm  s0, s0, s1
    FCONST  s1, 0xc2ae0000          // -87.0
    fmaxnm  s0, s0, s1
    FCONST  s1, 0x3fb8aa3b          // log2(e)
    fmul    s1, s0, s1
    frintn  s1, s1                  // k = round(x / ln2)
    fcvtzs  w10, s1
    FCONST  s2, 0x3f317200          // ln2 high part
    fmsub   s0, s1, s2, s0          // r = x - k*ln2_hi
    FCONST  s2, 0x35bfbe8e          // ln2 low part
    fmsub   s0, s1, s2, s0          //   - k*ln2_lo, |r| <= ln2/2
    FCONST  s1, 0x3ab60b61          // 1/720
    FCONST  s2, 0x3c088889          // 1/120
    fmadd   s1, s1, s0, s2
    FCONST  s2, 0x3d2aaaab          // 1/24
    fmadd   s1, s1, s0, s2
    FCONST  s2, 0x3e2aaaab          // 1/6
    fmadd   s1, s1, s0, s2
    fmov    s2, #0.5
    fmadd   s1, s1, s0, s2
    fmov    s2, #1.0
    fmadd   s1, s1, s0, s2
    fmadd   s1, s1, s0, s2          // p(r) ~= e^r
    add     w10, w10, #127
    lsl     w10, w10, #23
    fmov    s2, w10                 // 2^k built from raw exponent bits
    fmul    s0, s1, s2
    ret

// softmax(x0 x, x1 n), in place.
.globl _softmax
_softmax:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    ldr     s8, [x0]                // max
    mov     x2, #1
1:  cmp     x2, x20
    b.ge    2f
    ldr     s0, [x19, x2, lsl #2]
    fmax    s8, s8, s0
    add     x2, x2, #1
    b       1b
2:  movi    v9.16b, #0              // sum
    mov     x2, #0
3:  ldr     s0, [x19, x2, lsl #2]
    fsub    s0, s0, s8
    bl      expf
    str     s0, [x19, x2, lsl #2]
    fadd    s9, s9, s0
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    3b
    fmov    s0, #1.0
    fdiv    s9, s0, s9
    mov     x2, #0
4:  ldr     s0, [x19, x2, lsl #2]
    fmul    s0, s0, s9
    str     s0, [x19, x2, lsl #2]
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    4b
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// top5(x0 = x, x1 = n, x2 = out indices [5] (u32), x3 = out values [5])
// Five passes of argmax. Each pick is set to -inf in x, so x is changed. Leaf.
.globl _top5
_top5:
    FCONST  s1, 0xff800000          // -inf
    mov     x4, #5
1:  ldr     s0, [x0]                // best value
    mov     x5, #0                  // best index
    mov     x6, #1
2:  ldr     s2, [x0, x6, lsl #2]
    fcmp    s2, s0
    fcsel   s0, s2, s0, gt
    csel    x5, x6, x5, gt
    add     x6, x6, #1
    cmp     x6, x1
    b.lo    2b
    str     w5, [x2], #4
    str     s0, [x3], #4
    str     s1, [x0, x5, lsl #2]
    subs    x4, x4, #1
    b.ne    1b
    ret

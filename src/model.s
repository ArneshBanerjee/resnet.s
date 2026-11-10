// The forward pass. Activations are NHWC floats. Every conv is im2col plus
// one GEMM (src/gemm.s), with its BatchNorm folded in at load time.
.include "macros.inc"

.equ PADW, IMG + 6                   // img_pad width and height

.text
.p2align 2

// forward(): run the network on _image.
.globl _forward
_forward:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    ADDR    x0, img_pad
    MOV32   x9, (3*PADW + 3)*12      // inside the zero border
    add     x0, x0, x9
    ADDR    x1, _image
    ldr     x1, [x1]
    bl      normalize
    ADDR    x0, _debug               // -d 0: copy out the 224x224 image
    ldr     x0, [x0]
    cbnz    x0, 2f
    ADDR    x0, act_t
    ADDR    x1, img_pad
    MOV32   x9, (3*PADW + 3)*12
    add     x1, x1, x9
    mov     x2, #IMG
1:  mov     x3, #0
3:  ldr     q0, [x1, x3]
    str     q0, [x0], #16
    add     x3, x3, #16
    cmp     x3, #IMG*12
    b.lo    3b
    add     x1, x1, #PADW*12
    subs    x2, x2, #1
    b.ne    1b
    mov     x0, #0
    ADDR    x1, act_t
    MOV32   x2, IMG*IMG*3
    bl      debug

    // stem: conv 7x7 stride 2, BN, ReLU. 224 -> 112
2:  mov     x0, #0
    ADDR    x1, img_pad
    mov     x2, #IMG
    ADDR    x3, act_t
    mov     x4, #0
    mov     x5, #1
    bl      conv
    mov     x0, #1
    ADDR    x1, act_t
    MOV32   x2, 112*112*64
    bl      debug

    // max pool 3x3 stride 2. 112 -> 56
    ADDR    x0, act_x
    ADDR    x1, act_t
    mov     x2, #112
    mov     x3, #64
    bl      maxpool
    mov     x0, #2
    ADDR    x1, act_x
    MOV32   x2, 56*56*64
    bl      debug

    // 8 basic blocks (layer1 to layer4), each in place on act_x:
    //   t = relu(conv1(x)), x = relu(conv2(t) + x), where x is first
    //   replaced by downsample(x) (1x1 stride 2 conv + BN) if the block has one.
    mov     x19, #56                 // H
    mov     x20, #0                  // block
.Lblock:
    ADDR    x9, blocks
    add     x9, x9, x20, lsl #1
    ldrb    w21, [x9]                // conv1 index (conv2 is the next one)
    ldrb    w22, [x9, #1]            // downsample conv index, or 0
    mov     x0, x21
    ADDR    x1, act_x
    mov     x2, x19
    ADDR    x3, act_t
    mov     x4, #0
    mov     x5, #1
    bl      conv
    mov     x23, x0                  // OH
    ADDR    x24, act_x               // residual
    cbz     x22, 1f
    mov     x0, x22
    ADDR    x1, act_x
    mov     x2, x19
    ADDR    x3, act_d
    mov     x4, #0
    mov     x5, #0
    bl      conv
    ADDR    x24, act_d
1:  add     x0, x21, #1
    ADDR    x1, act_t
    mov     x2, x23
    ADDR    x3, act_x
    mov     x4, x24
    mov     x5, #1
    bl      conv
    mov     x19, x23
    ADDR    x9, _conv_shape          // debug point 3 + block: OH*OH*cout
    add     x9, x9, x21, lsl #3
    ldrh    w2, [x9, #2]
    mul     x2, x2, x19
    mul     x2, x2, x19
    add     x0, x20, #3
    ADDR    x1, act_x
    bl      debug
    add     x20, x20, #1
    cmp     x20, #8
    b.lo    .Lblock

    // global average pool: 7x7x512 -> 512
    ADDR    x0, feat
    ADDR    x1, act_x
    bl      avgpool
    mov     x0, #11
    ADDR    x1, feat
    mov     x2, #512
    bl      debug

    // fc: the same GEMM, with the 512 features as row 0 of one A panel
    ADDR    x0, col
    ADDR    x1, feat
    mov     x2, #512
1:  ldr     s0, [x1], #4             // one row of the panel: feature, 11 zeros
    st1     {v0.4s}, [x0], #16
    movi    v0.16b, #0
    stp     q0, q0, [x0], #32
    subs    x2, x2, #1
    b.ne    1b
    ADDR    x9, _convs
    ldp     x2, x3, [x9, #N_CONV*16] // packed fc weights, bias
    ADDR    x0, _logits
    ADDR    x1, col
    mov     x4, #1                   // M: one row
    mov     x5, #N_CLASSES
    mov     x6, #512
    mov     x7, #0
    mov     x8, #0
    bl      _gemm
    mov     x0, #12
    ADDR    x1, _logits
    mov     x2, #N_CLASSES
    bl      debug

    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// conv(x0 = conv index, x1 = in [H][H][cin], x2 = H, x3 = out,
//      x4 = residual or 0, x5 = 1 for ReLU) -> x0 = output size OH
// Shape from _conv_shape. Padding is k/2: OH = (H + 2*pad - k)/stride + 1.
conv:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    mov     x19, x0
    mov     x20, x3
    mov     x21, x4
    mov     x22, x5
    ADDR    x9, _conv_shape
    add     x9, x9, x19, lsl #3
    ldrh    w3, [x9]                 // cin
    ldrh    w23, [x9, #2]            // cout
    ldrh    w4, [x9, #4]             // k
    ldrh    w5, [x9, #6]             // stride
    lsr     x6, x4, #1               // pad
    add     x6, x2, x6, lsl #1
    sub     x6, x6, x4
    udiv    x6, x6, x5
    add     x24, x6, #1              // OH
    mov     x6, x24
    mul     x7, x4, x4
    mul     x7, x7, x3
    add     x7, x7, #3
    and     x25, x7, #~3             // Kp
    mov     x7, x25
    ADDR    x0, col
    tst     x3, #3                   // cin 3: the stem
    b.ne    1f
    bl      im2col4
    b       2f
1:  bl      im2col_stem
2:
    mul     x4, x24, x24             // M
    ADDR    x9, _convs
    add     x9, x9, x19, lsl #4
    ldp     x2, x3, [x9]             // packed weights, bias
    mov     x0, x20
    ADDR    x1, col
    mov     x5, x23
    mov     x6, x25
    mov     x7, x21
    mov     x8, x22
    bl      _gemm
    mov     x0, x24
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret

// im2col4(x0 = A, x1 = in [H][H][cin], x2 = H, x3 = cin, x4 = k,
//         x5 = stride, x6 = OH, x7 = Kp), for cin a multiple of 4 (Kp = K)
// Writes A in 12-row panels, A[M/12][Kp][12], for gemm. Row m is output
// pixel (oy, ox) = (m / OH, m % OH). Column (ky*k + kx)*cin + c holds
// in[oy*stride - pad + ky][ox*stride - pad + kx][c], or 0 outside the
// image. Rows past the last pixel are 0.
// Works on one 12-row panel at a time. For each tap it reads 4 channels
// of all 12 pixels, transposes them (three 4x4 transposes) and writes 4
// rows of the panel: 192 bytes in order. A pixel outside the image reads
// from a buffer of zeros.

// Point register xr at pixel r's input for this tap, or at the zeros.
// Pixel r has base pointer [sp + 16r], iy0 and ix0 at [sp + 16r + 8].
.macro PIX r, xr
    ldr     \xr, [sp, #16*\r]
    ldp     w1, w2, [sp, #16*\r + 8]
    add     w1, w1, w24              // iy
    add     w2, w2, w25              // ix
    cmp     w1, w21                  // unsigned: also catches negative
    ccmp    w2, w21, #2, lo          // not lo -> set C (hs)
    add     \xr, \xr, x26
    csel    \xr, \xr, x27, lo
.endm

// Transpose 4 pixels x 4 channels (a, b, c, d) into 4 channels x 4 pixels
// (o0..o3). Uses v4-v7.
.macro TR4 a, b, c, d, o0, o1, o2, o3
    trn1    v4.4s, \a\().4s, \b\().4s     // a0 b0 a2 b2
    trn2    v5.4s, \a\().4s, \b\().4s     // a1 b1 a3 b3
    trn1    v6.4s, \c\().4s, \d\().4s     // c0 d0 c2 d2
    trn2    v7.4s, \c\().4s, \d\().4s     // c1 d1 c3 d3
    zip1    \o0\().2d, v4.2d, v6.2d        // a0 b0 c0 d0
    zip1    \o1\().2d, v5.2d, v7.2d        // a1 b1 c1 d1
    zip2    \o2\().2d, v4.2d, v6.2d        // a2 b2 c2 d2
    zip2    \o3\().2d, v5.2d, v7.2d        // a3 b3 c3 d3
.endm

im2col4:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    sub     sp, sp, #208             // 12 pixels x 16 bytes, stride, pad
    mov     x19, x0                  // dst, written in order
    mov     x20, x1                  // in
    mov     x21, x2                  // H
    lsl     x22, x3, #2              // bytes in one input pixel
    mov     x23, x4                  // k
    lsr     x9, x4, #1
    stp     x5, x9, [sp, #192]       // stride, pad
    mov     x4, x6                   // OH
    mul     x3, x6, x6               // M
    add     x28, x3, #11
    mov     x9, #12
    udiv    x28, x28, x9
    mul     x28, x28, x9             // M rounded up to 12
    ADDR    x27, zeros
    mov     x0, #0                   // first pixel of the panel
.Lp4_panel:
    mov     x1, #0                   // pixel r of the panel
1:  add     x2, x0, x1               // m
    add     x5, sp, x1, lsl #4
    cmp     x2, x3
    b.hs    2f
    udiv    x10, x2, x4              // oy
    msub    x11, x10, x4, x2         // ox
    ldp     x12, x13, [sp, #192]
    mul     x10, x10, x12
    sub     x10, x10, x13            // iy0
    mul     x11, x11, x12
    sub     x11, x11, x13            // ix0
    madd    x14, x10, x21, x11
    madd    x14, x14, x22, x20       // in + (iy0*H + ix0) * pixel bytes
    str     x14, [x5]
    stp     w10, w11, [x5, #8]
    b       3f
2:  mov     w10, #0x40000000         // past the last pixel: never inside
    str     x20, [x5]
    stp     w10, w10, [x5, #8]
3:  add     x1, x1, #1
    cmp     x1, #12
    b.lo    1b
    mov     x24, #0                  // ky
.Lp4_ky:
    mov     x25, #0                  // kx
.Lp4_kx:
    madd    x26, x24, x21, x25
    mul     x26, x26, x22            // byte offset of this tap
    PIX     0, x6
    PIX     1, x7
    PIX     2, x8
    PIX     3, x9
    PIX     4, x10
    PIX     5, x11
    PIX     6, x12
    PIX     7, x13
    PIX     8, x14
    PIX     9, x15
    PIX     10, x16
    PIX     11, x17
    mov     x5, x22
.Lp4_c:
    ldr     q0, [x6], #16
    ldr     q1, [x7], #16
    ldr     q2, [x8], #16
    ldr     q3, [x9], #16
    TR4     v0, v1, v2, v3, v16, v19, v22, v25
    ldr     q0, [x10], #16
    ldr     q1, [x11], #16
    ldr     q2, [x12], #16
    ldr     q3, [x13], #16
    TR4     v0, v1, v2, v3, v17, v20, v23, v26
    ldr     q0, [x14], #16
    ldr     q1, [x15], #16
    ldr     q2, [x16], #16
    ldr     q3, [x17], #16
    TR4     v0, v1, v2, v3, v18, v21, v24, v27
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x19], #64
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x19], #64
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x19], #64
    subs    x5, x5, #16
    b.ne    .Lp4_c
    add     x25, x25, #1
    cmp     x25, x23
    b.lo    .Lp4_kx
    add     x24, x24, #1
    cmp     x24, x23
    b.lo    .Lp4_ky
    add     x0, x0, #12
    cmp     x0, x28
    b.lo    .Lp4_panel
    add     sp, sp, #208
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// im2col_stem(x0 = A, x1 = img_pad): im2col for the stem (7x7, stride 2,
// 3 channels, Kp = 148). img_pad has a zero border of 3 pixels, so no tap
// is outside it. For one ky, the 7 taps are 21 floats in a row. So each
// row of the kernel is 5 groups of 4 floats (transposed like im2col4)
// plus 1 float, and the 148th column is 0. Leaf.
.macro SPIX r, xr                    // xr = &img_pad[2*oy][2*ox] for pixel m0 + r
    add     x3, x0, #\r
    udiv    x4, x3, x5               // oy (past the last pixel: oy = 112, still inside img_pad)
    msub    x3, x4, x5, x3           // ox
    mov     \xr, #2*PADW*12
    madd    \xr, x4, \xr, x1
    add     x3, x3, x3, lsl #1
    add     \xr, \xr, x3, lsl #3     // + ox * 24 bytes
.endm
.macro SGROUP                        // 4 floats at offset x4 of each pixel
    ldr     q0, [x6, x4]
    ldr     q1, [x7, x4]
    ldr     q2, [x8, x4]
    ldr     q3, [x9, x4]
    TR4     v0, v1, v2, v3, v16, v19, v22, v25
    ldr     q0, [x10, x4]
    ldr     q1, [x11, x4]
    ldr     q2, [x12, x4]
    ldr     q3, [x13, x4]
    TR4     v0, v1, v2, v3, v17, v20, v23, v26
    ldr     q0, [x14, x4]
    ldr     q1, [x15, x4]
    ldr     q2, [x16, x4]
    ldr     q3, [x17, x4]
    TR4     v0, v1, v2, v3, v18, v21, v24, v27
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x2], #64
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    add     x4, x4, #16
.endm
.macro SONE xr                       // 1 float at offset x4
    ldr     s0, [\xr, x4]
    str     s0, [x2], #4
.endm
im2col_stem:
    mov     x2, x0                   // dst, written in order
    movi    v28.16b, #0
    mov     x0, #0                   // first pixel of the panel
.Lst_panel:
    mov     x5, #112                 // OH
    SPIX    0, x6
    SPIX    1, x7
    SPIX    2, x8
    SPIX    3, x9
    SPIX    4, x10
    SPIX    5, x11
    SPIX    6, x12
    SPIX    7, x13
    SPIX    8, x14
    SPIX    9, x15
    SPIX    10, x16
    SPIX    11, x17
    mov     x3, #0                   // byte offset of row ky
    mov     x5, #7
1:  mov     x4, x3
    SGROUP
    SGROUP
    SGROUP
    SGROUP
    SGROUP
    SONE    x6
    SONE    x7
    SONE    x8
    SONE    x9
    SONE    x10
    SONE    x11
    SONE    x12
    SONE    x13
    SONE    x14
    SONE    x15
    SONE    x16
    SONE    x17
    add     x3, x3, #PADW*12
    subs    x5, x5, #1
    b.ne    1b
    stp     q28, q28, [x2], #32      // column 148: 0
    str     q28, [x2], #16
    add     x0, x0, #12
    MOV32   x3, (112*112 + 11) / 12 * 12
    cmp     x0, x3
    b.lo    .Lst_panel
    ret

// normalize(x0 = out: the inside of img_pad, x1 = RGB bytes)
// out = (byte / 255 - mean) / std, the same steps as torchvision. Leaf.
normalize:
    ADDR    x2, norm_const
    ldp     s16, s17, [x2]           // mean
    ldr     s18, [x2, #8]
    ldp     s19, s20, [x2, #12]      // std
    ldr     s21, [x2, #20]
    FCONST  s22, 0x437f0000          // 255.0
    mov     x5, #IMG                 // rows
2:  mov     x3, #IMG                 // pixels in a row
1:  ldrb    w4, [x1], #1
    ucvtf   s0, w4
    fdiv    s0, s0, s22
    fsub    s0, s0, s16
    fdiv    s0, s0, s19
    ldrb    w4, [x1], #1
    ucvtf   s1, w4
    fdiv    s1, s1, s22
    fsub    s1, s1, s17
    fdiv    s1, s1, s20
    ldrb    w4, [x1], #1
    ucvtf   s2, w4
    fdiv    s2, s2, s22
    fsub    s2, s2, s18
    fdiv    s2, s2, s21
    stp     s0, s1, [x0], #8
    str     s2, [x0], #4
    subs    x3, x3, #1
    b.ne    1b
    add     x0, x0, #(PADW - IMG)*12 // skip the border
    subs    x5, x5, #1
    b.ne    2b
    ret

// maxpool(x0 = out, x1 = in [H][H][C], x2 = H, x3 = C)
// 3x3 window, stride 2, padding 1 (padding never wins). C multiple of 4. Leaf.
maxpool:
    lsr     x4, x2, #1               // OH
    lsl     x13, x3, #2              // bytes in one pixel
    FCONST  s1, 0xff800000           // -inf
    dup     v1.4s, v1.s[0]
    mov     x5, #0                   // oy
1:  mov     x6, #0                   // ox
2:  mov     x7, #0                   // byte offset of 4 channels
3:  mov     v0.16b, v1.16b
    mov     x8, #0                   // ky
4:  lsl     x10, x5, #1
    add     x10, x10, x8
    sub     x10, x10, #1             // iy
    cmp     x10, x2
    b.hs    7f
    mov     x9, #0                   // kx
5:  lsl     x11, x6, #1
    add     x11, x11, x9
    sub     x11, x11, #1             // ix
    cmp     x11, x2
    b.hs    6f
    madd    x12, x10, x2, x11
    madd    x12, x12, x13, x7
    ldr     q2, [x1, x12]
    fmax    v0.4s, v0.4s, v2.4s
6:  add     x9, x9, #1
    cmp     x9, #3
    b.lo    5b
7:  add     x8, x8, #1
    cmp     x8, #3
    b.lo    4b
    str     q0, [x0], #16
    add     x7, x7, #16
    cmp     x7, x13
    b.lo    3b
    add     x6, x6, #1
    cmp     x6, x4
    b.lo    2b
    add     x5, x5, #1
    cmp     x5, x4
    b.lo    1b
    ret

// avgpool(x0 = out [512], x1 = in [7][7][512]): mean over the 49 pixels. Leaf.
avgpool:
    FCONST  s2, 0x42440000           // 49.0
    dup     v2.4s, v2.s[0]
    mov     x2, #0                   // byte offset of 4 channels
1:  add     x3, x1, x2
    movi    v0.16b, #0
    mov     x4, #49
2:  ldr     q1, [x3]
    fadd    v0.4s, v0.4s, v1.4s
    add     x3, x3, #512*4
    subs    x4, x4, #1
    b.ne    2b
    fdiv    v0.4s, v0.4s, v2.4s
    str     q0, [x0, x2]
    add     x2, x2, #16
    cmp     x2, #512*4
    b.lo    1b
    ret

// debug(x0 = point, x1 = floats, x2 = count): if -d chose this point,
// write the floats to stdout and exit.
debug:
    ADDR    x3, _debug
    ldr     x3, [x3]
    cmp     x3, x0
    b.ne    2f
    mov     x19, x1                  // we exit after this, so no need to save
    lsl     x20, x2, #2
1:  mov     x0, #1
    mov     x1, x19
    mov     x2, x20
    SYSCALL SYS_WRITE                // may write less than asked: loop
    b.cs    3f
    add     x19, x19, x0
    subs    x20, x20, x0
    b.hi    1b
    mov     x0, #0
    SYSCALL SYS_EXIT
2:  ret
3:  mov     x0, #1
    SYSCALL SYS_EXIT

.section __TEXT,__const
// The 8 basic blocks: conv1 index in _conv_shape, downsample conv index or 0.
blocks:
    .byte  1, 0,   3, 0              // layer1
    .byte  5, 7,   8, 0              // layer2
    .byte 10, 12, 13, 0              // layer3
    .byte 15, 17, 18, 0              // layer4
.p2align 2
norm_const:
    .float 0.485, 0.456, 0.406       // ImageNet mean
    .float 0.229, 0.224, 0.225       // ImageNet std

// Activations: buffers big enough for the stem output
// (112*112 pixels rounded up to 12 rows, x 64 channels).
.zerofill __DATA,__bss,act_x,ACT_FLOATS*4,6
// act_x: the main path. act_t: inside a block. act_d: downsample output.
.zerofill __DATA,__bss,act_t,ACT_FLOATS*4,6
.zerofill __DATA,__bss,act_d,ACT_FLOATS*4,6
.zerofill __DATA,__bss,feat,512*4,4
// The normalized input with a zero border of 3 pixels for the stem conv.
// One extra row: the 8 padding rows of the last stem panel read row 230.
.zerofill __DATA,__bss,img_pad,(PADW+1)*PADW*12,6
// Read by im2col4 for pixels outside the image: 512 channels of 0.
.zerofill __DATA,__bss,zeros,512*4,4
// fc output: one 12-row panel. Row 0 is the logits.
.globl _logits
.zerofill __DATA,__bss,_logits,12*N_CLASSES*4,4
// im2col output, big enough for the stem (Kp = 148).
.zerofill __DATA,__bss,col,COL_FLOATS*4,6

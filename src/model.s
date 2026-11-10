// The forward pass. Activations are NHWC floats. Every conv is im2col plus
// one GEMM (src/gemm.s), with its BatchNorm folded in at load time.
.include "macros.inc"

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
    ADDR    x0, act_x
    ADDR    x1, _image
    ldr     x1, [x1]
    bl      normalize
    mov     x0, #0
    ADDR    x1, act_x
    MOV32   x2, IMG*IMG*3
    bl      debug

    // stem: conv 7x7 stride 2, BN, ReLU. 224 -> 112
    mov     x0, #0
    ADDR    x1, act_x
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
    mov     x4, #1
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
    bl      im2col
    mul     x4, x24, x24             // M
    add     x4, x4, #11
    mov     x9, #12
    udiv    x4, x4, x9               // row panels
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

// im2col(x0 = A, x1 = in [H][H][cin], x2 = H, x3 = cin, x4 = k,
//        x5 = stride, x6 = OH, x7 = Kp)
// Writes A in 12-row panels, A[M/12][Kp][12], for gemm. Row m is output
// pixel (oy, ox) = (m / OH, m % OH). Column (ky*k + kx)*cin + c holds
// in[oy*stride - pad + ky][ox*stride - pad + kx][c], or 0 outside the
// image. Rows past the last pixel and columns past K are 0. Leaf.
im2col:
    stp     x19, x20, [sp, #-32]!
    stp     x21, x22, [sp, #16]
    lsr     x19, x4, #1              // pad
    mul     x8, x6, x6               // M
    add     x20, x8, #11
    mov     x9, #12
    udiv    x20, x20, x9
    mul     x20, x20, x9             // M rounded up to 12
    mul     x21, x4, x4
    mul     x21, x21, x3
    sub     x21, x7, x21             // padding columns: Kp - K
    mov     x22, #48
    mul     x22, x7, x22             // bytes in one A panel
    mov     x9, #0                   // m
.Lic_row:
    mov     x10, #12                 // dst = A + (m/12)*panel + (m%12)*4
    udiv    x11, x9, x10
    msub    x12, x11, x10, x9
    madd    x10, x11, x22, x0
    add     x10, x10, x12, lsl #2
    cmp     x9, x8
    b.hs    .Lic_zero
    udiv    x13, x9, x6              // oy
    msub    x14, x13, x6, x9         // ox
    mul     x13, x13, x5
    sub     x13, x13, x19            // iy of the first tap
    mul     x14, x14, x5
    sub     x14, x14, x19            // ix of the first tap
    mov     x11, #0                  // ky
1:  add     x15, x13, x11            // iy
    mov     x12, #0                  // kx
2:  add     x16, x14, x12            // ix
    mov     x17, x3
    cmp     x15, x2                  // unsigned: also catches negative
    b.hs    4f
    cmp     x16, x2
    b.hs    4f
    madd    x16, x15, x2, x16
    mul     x16, x16, x3
    add     x16, x1, x16, lsl #2     // &in[iy][ix][0]
3:  ldr     s0, [x16], #4
    str     s0, [x10], #48
    subs    x17, x17, #1
    b.ne    3b
    b       5f
4:  str     wzr, [x10], #48          // outside the image
    subs    x17, x17, #1
    b.ne    4b
5:  add     x12, x12, #1
    cmp     x12, x4
    b.lo    2b
    add     x11, x11, #1
    cmp     x11, x4
    b.lo    1b
    mov     x17, x21                 // padding columns
    b       7f
.Lic_zero:
    mov     x17, x7                  // a row past the last pixel
7:  cbz     x17, 8f
6:  str     wzr, [x10], #48
    subs    x17, x17, #1
    b.ne    6b
8:  add     x9, x9, #1
    cmp     x9, x20
    b.lo    .Lic_row
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #32
    ret

// normalize(x0 = out [224][224][3] floats, x1 = RGB bytes)
// out = (byte / 255 - mean) / std, the same steps as torchvision. Leaf.
normalize:
    ADDR    x2, norm_const
    ldp     s16, s17, [x2]           // mean
    ldr     s18, [x2, #8]
    ldp     s19, s20, [x2, #12]      // std
    ldr     s21, [x2, #20]
    FCONST  s22, 0x437f0000          // 255.0
    MOV32   x3, IMG*IMG
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
// fc output: one 12-row panel. Row 0 is the logits.
.globl _logits
.zerofill __DATA,__bss,_logits,12*N_CLASSES*4,4
// im2col output, big enough for the stem (Kp = 148).
.zerofill __DATA,__bss,col,COL_FLOATS*4,6

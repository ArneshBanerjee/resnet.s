// Startup: read Hugging Face's model.safetensors as it is, with no
// conversion step.
//
// load_model: parse the JSON header and fill _tensors with a pointer to
//             each tensor we use. Then fold every BatchNorm into the conv
//             before it and repack all weights for the GEMM kernel.
.include "macros.inc"

.text
.p2align 2

// load_model(x0 = mapped model.safetensors, x1 = file size)
// File layout: u64 header length, JSON header, then raw tensor data.
// Note: this file's header is 10,830 bytes with no padding, so the float
// data is not 4-byte aligned. AArch64 allows unaligned loads.
.globl _load_model
_load_model:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    ldr     x2, [x0]                 // header length
    add     x3, x2, #8
    cmp     x3, x1
    b.hs    .Lmodel_bad
    add     x19, x0, #8              // cursor in the JSON header
    add     x20, x19, x2             // end of header = start of tensor data
    mov     x21, #0                  // tensors we use, found
    ADDR    x22, _tensors

.Lnext_key:
    cmp     x19, x20                 // find the next key: "name"
    b.hs    .Lheader_done
    ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.ne    .Lnext_key
    mov     x24, x19
1:  ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.ne    1b
    sub     x1, x19, x24
    sub     x1, x1, #1
    mov     x0, x24
    ADDR    x2, tensor_names
    bl      find_name
    mov     x25, x0                  // slot, or -1 for a tensor we do not use
    mov     x26, #-1                 // byte offset of the data
2:  ldrb    w0, [x19], #1            // walk the value object up to its '}'
    cmp     w0, #'}'
    b.eq    4f
    cmp     w0, #'['                 // only "data_offsets":[ is preceded by ts":
    b.ne    2b
    ldurb   w1, [x19, #-5]
    cmp     w1, #'t'
    b.ne    2b
    ldurb   w1, [x19, #-4]
    cmp     w1, #'s'
    b.ne    2b
    ldurb   w1, [x19, #-3]
    cmp     w1, #'"'
    b.ne    2b
    mov     x26, #0
    mov     x1, #10
3:  ldrb    w0, [x19]                // first number = start offset
    sub     w0, w0, #'0'
    cmp     w0, #9
    b.hi    2b
    madd    x26, x26, x1, x0
    add     x19, x19, #1
    b       3b
4:  tbnz    x25, #63, .Lnext_key
    tbnz    x26, #63, .Lmodel_bad
    ldr     x0, [x22, x25, lsl #3]   // a name seen twice is an error
    cbnz    x0, .Lmodel_bad
    add     x0, x20, x26
    str     x0, [x22, x25, lsl #3]
    add     x21, x21, #1
    b       .Lnext_key

.Lheader_done:
    cmp     x21, #N_TENSORS
    b.ne    .Lmodel_bad

    // Fold each BatchNorm into its conv and repack the weights for the
    // GEMM kernel, into memory that we map for ourselves:
    //   scale = gamma / sqrt(var + eps), w' = w * scale,
    //   b' = beta - mean * scale, eps = 1e-5.
    // fc is packed the same way, with scale 1 and its own bias.
    mov     x0, #0
    MOV32   x1, PACKED_BYTES
    mov     x2, #3                   // PROT_READ | PROT_WRITE
    mov     x3, #0x1002              // MAP_ANON | MAP_PRIVATE
    mov     x4, #-1
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cs    .Lnomem
    mov     x24, x0                  // next free byte in the new memory
    mov     x25, #0                  // conv index
    ADDR    x26, _convs
    ADDR    x28, scale_buf
.Lconv:
    ADDR    x9, _conv_shape
    add     x9, x9, x25, lsl #3
    ldrh    w19, [x9]                // cin
    ldrh    w20, [x9, #2]            // cout
    ldrh    w21, [x9, #4]            // k
    mul     x21, x21, x21            // taps = k*k
    mul     x23, x19, x21
    add     x23, x23, #3
    and     x23, x23, #~3            // Kp: K = taps*cin, rounded up to 4
    mul     x27, x23, x20
    add     x27, x24, x27, lsl #2    // the bias goes after the weights
    FCONST  s4, 0x3727c5ac           // 1e-5 (FCONST uses w9)
    mov     x9, #40
    madd    x9, x25, x9, x22         // this conv's 5 tensor pointers
    mov     x10, #0
    cmp     x25, #N_CONV
    b.eq    2f
    ldp     x11, x12, [x9, #8]       // BN weight (gamma), bias (beta)
    ldp     x13, x14, [x9, #24]      // running_mean, running_var
1:  ldr     s0, [x14, x10, lsl #2]
    fadd    s0, s0, s4
    fsqrt   s0, s0
    ldr     s1, [x11, x10, lsl #2]
    fdiv    s1, s1, s0               // scale
    str     s1, [x28, x10, lsl #2]
    ldr     s2, [x13, x10, lsl #2]
    ldr     s3, [x12, x10, lsl #2]
    fmsub   s3, s2, s1, s3           // beta - mean * scale
    str     s3, [x27, x10, lsl #2]
    add     x10, x10, #1
    cmp     x10, x20
    b.lo    1b
    b       3f
2:  ldr     x12, [x9, #8]            // fc: scale 1, copy fc.bias
    fmov    s1, #1.0
21: str     s1, [x28, x10, lsl #2]
    ldr     s3, [x12, x10, lsl #2]
    str     s3, [x27, x10, lsl #2]
    add     x10, x10, #1
    cmp     x10, x20
    b.lo    21b
3:  mov     x0, x24
    ldr     x1, [x9]                 // conv weight [cout][cin][k][k]
    mov     x2, x19
    mov     x3, x20
    mov     x4, x21
    mov     x5, x23
    bl      pack
    add     x9, x26, x25, lsl #4
    stp     x24, x27, [x9]           // packed weights, bias
    add     x24, x27, x20, lsl #2
    add     x25, x25, #1
    cmp     x25, #N_CONV
    b.ls    .Lconv

    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret
.Lnomem:
    ADDR    x1, s_nomem
    b       _fail
.Lmodel_bad:
    ADDR    x1, s_bad_model
    b       _fail

// pack(x0 = dst, x1 = w[cout][cin][taps], x2 = cin, x3 = cout, x4 = taps,
// x5 = Kp). Writes dst[cout/8][Kp][8]: row k = tap*cin + c of panel p holds
// w[8p + j][c][tap] * scale_buf[8p + j] for j = 0..7. Rows K..Kp-1 stay 0
// (new anonymous memory is zero). Leaf.
pack:
    ADDR    x6, scale_buf
    mul     x7, x2, x4               // K = floats per output channel in w
    lsl     x8, x7, #2               // bytes between output channels in w
    sub     x9, x5, x7
    lsl     x9, x9, #5               // bytes of the padding rows in a panel
    mov     x10, #0                  // first output channel of the panel
1:  mov     x11, #0                  // tap
2:  mov     x12, #0                  // input channel
3:  madd    x13, x12, x4, x11        // c*taps + tap
    madd    x13, x10, x7, x13        // + n*K
    add     x13, x1, x13, lsl #2
    add     x14, x6, x10, lsl #2
    mov     x15, #8
4:  ldr     s0, [x13]                // 8 output channels: one row of the panel
    add     x13, x13, x8
    ldr     s1, [x14], #4
    fmul    s0, s0, s1
    str     s0, [x0], #4
    subs    x15, x15, #1
    b.ne    4b
    add     x12, x12, #1
    cmp     x12, x2
    b.lo    3b
    add     x11, x11, #1
    cmp     x11, x4
    b.lo    2b
    add     x0, x0, x9
    add     x10, x10, #8
    cmp     x10, x3
    b.lo    1b
    ret

// find_name(x0 = text, x1 = length, x2 = list of NUL-terminated names,
// ended by an empty name) -> x0 = index in the list, or -1. Leaf.
find_name:
    mov     x6, #0                   // index
1:  ldrb    w3, [x2]
    cbz     w3, 4f
    mov     x3, #0                   // compare text[0..len) with the name
2:  ldrb    w4, [x2, x3]
    cmp     x3, x1
    b.eq    3f
    cbz     w4, 5f
    ldrb    w5, [x0, x3]
    cmp     w4, w5
    b.ne    5f
    add     x3, x3, #1
    b       2b
3:  cbnz    w4, 5f                   // text ended first
    mov     x0, x6
    ret
5:  ldrb    w3, [x2], #1             // skip to the next name
    cbnz    w3, 5b
    add     x6, x6, #1
    b       1b
4:  mov     x0, #-1
    ret

.section __TEXT,__const
// The 5 tensors of one conv and the BatchNorm after it.
.macro CONV conv, bn
    .asciz "\conv\().weight"
    .asciz "\bn\().weight"
    .asciz "\bn\().bias"
    .asciz "\bn\().running_mean"
    .asciz "\bn\().running_var"
.endm
.macro BLOCK b
    CONV \b\().conv1, \b\().bn1
    CONV \b\().conv2, \b\().bn2
.endm
// All tensors we use, in forward order. The index is the slot in _tensors.
tensor_names:
    CONV  conv1, bn1                                 // conv 0: stem
    BLOCK layer1.0                                   // convs 1-2
    BLOCK layer1.1                                   // convs 3-4
    BLOCK layer2.0                                   // convs 5-6
    CONV  layer2.0.downsample.0, layer2.0.downsample.1  // conv 7
    BLOCK layer2.1                                   // convs 8-9
    BLOCK layer3.0                                   // convs 10-11
    CONV  layer3.0.downsample.0, layer3.0.downsample.1  // conv 12
    BLOCK layer3.1                                   // convs 13-14
    BLOCK layer4.0                                   // convs 15-16
    CONV  layer4.0.downsample.0, layer4.0.downsample.1  // conv 17
    BLOCK layer4.1                                   // convs 18-19
    .asciz "fc.weight"
    .asciz "fc.bias"
    .byte 0
s_bad_model: .asciz "resnet: model.safetensors is not the ResNet-18 checkpoint\n"
s_nomem:     .asciz "resnet: out of memory\n"

// Shape of each conv (then fc): cin, cout, kernel size, stride.
// Padding is kernel size / 2.
.p2align 3
.globl _conv_shape
_conv_shape:
    .hword   3,  64, 7, 2            // 0  stem
    .hword  64,  64, 3, 1            // 1  layer1.0
    .hword  64,  64, 3, 1            // 2
    .hword  64,  64, 3, 1            // 3  layer1.1
    .hword  64,  64, 3, 1            // 4
    .hword  64, 128, 3, 2            // 5  layer2.0
    .hword 128, 128, 3, 1            // 6
    .hword  64, 128, 1, 2            // 7  layer2.0 downsample
    .hword 128, 128, 3, 1            // 8  layer2.1
    .hword 128, 128, 3, 1            // 9
    .hword 128, 256, 3, 2            // 10 layer3.0
    .hword 256, 256, 3, 1            // 11
    .hword 128, 256, 1, 2            // 12 layer3.0 downsample
    .hword 256, 256, 3, 1            // 13 layer3.1
    .hword 256, 256, 3, 1            // 14
    .hword 256, 512, 3, 2            // 15 layer4.0
    .hword 512, 512, 3, 1            // 16
    .hword 256, 512, 1, 2            // 17 layer4.0 downsample
    .hword 512, 512, 3, 1            // 18 layer4.1
    .hword 512, 512, 3, 1            // 19
    .hword 512, 1000, 1, 1           // 20 fc

.globl _tensors
.zerofill __DATA,__bss,_tensors,N_TENSORS*8,3
// For each conv (then fc): packed weights [cout/8][Kp][8], bias [cout].
.globl _convs
.zerofill __DATA,__bss,_convs,(N_CONV+1)*16,3
.zerofill __DATA,__bss,scale_buf,N_CLASSES*4,4

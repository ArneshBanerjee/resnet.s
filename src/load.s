// Startup: read Hugging Face's model.safetensors as it is, with no
// conversion step.
//
// load_model: parse the JSON header and fill _tensors with a pointer to
//             each tensor we use.
.include "macros.inc"

.text
.p2align 2

// load_model(x0 = mapped model.safetensors, x1 = file size)
//   -> x0 = number of tensors in the header
// File layout: u64 header length, JSON header, then raw tensor data.
// Note: this file's header is 10,830 bytes with no padding, so the float
// data is not 4-byte aligned. AArch64 allows unaligned loads.
.globl _load_model
_load_model:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    ldr     x2, [x0]                 // header length
    add     x3, x2, #8
    cmp     x3, x1
    b.hs    .Lmodel_bad
    add     x19, x0, #8              // cursor in the JSON header
    add     x20, x19, x2             // end of header = start of tensor data
    mov     x21, #0                  // tensors we use, found
    ADDR    x22, _tensors
    mov     x23, #0                  // all tensors in the header

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
    ldrb    w0, [x24]                // "__metadata__" is not a tensor
    cmp     w0, #'_'
    cinc    x23, x23, ne
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
    mov     x0, x23
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
.Lmodel_bad:
    ADDR    x1, s_bad_model
    b       _fail

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

.globl _tensors
.zerofill __DATA,__bss,_tensors,N_TENSORS*8,3

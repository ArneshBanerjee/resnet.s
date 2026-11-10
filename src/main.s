// resnet.s: ResNet-18 image classification in AArch64 assembly.
// No libc calls: all I/O is raw Darwin syscalls (svc #0x80).
//
// usage: resnet [-d N] image.ppm
//   -d N  write the raw float activations (f32, NHWC) after debug point N
//         to stdout: 0 input, 1 stem, 2 maxpool, 3-10 the 8 basic blocks,
//         11 avgpool, 12 logits
.include "macros.inc"

.text
.p2align 2

.globl _main
_main:
    mov     x19, x0                  // argc
    mov     x20, x1                  // argv
    mov     x21, #0                  // image path
    mov     x24, #1
.Largs:
    cmp     x24, x19
    b.ge    .Largs_done
    ldr     x0, [x20, x24, lsl #3]
    ldrb    w1, [x0]
    cmp     w1, #'-'
    b.ne    .Lpath
    ldrb    w1, [x0, #1]             // the only flag is -d N
    ldrb    w2, [x0, #2]
    cmp     w1, #'d'
    b.ne    .Lusage
    cbnz    w2, .Lusage
    add     x24, x24, #1
    cmp     x24, x19
    b.ge    .Lusage
    ldr     x0, [x20, x24, lsl #3]
    ldrb    w1, [x0]                 // N must start with a digit
    sub     w1, w1, #'0'
    cmp     w1, #9
    b.hi    .Lusage
    bl      parse_uint
    cmp     x0, #N_DEBUG
    b.hs    .Lusage
    ADDR    x1, _debug
    str     x0, [x1]
    b       .Lnext
.Lpath:
    cbnz    x21, .Lusage             // one image only
    mov     x21, x0
.Lnext:
    add     x24, x24, #1
    b       .Largs
.Largs_done:
    cbz     x21, .Lusage

    // the original Hugging Face file, read as it is
    ADDR    x0, path_model
    bl      map_file
    bl      _load_model
    mov     x22, x0                  // tensors in the header
    mov     x0, x21
    bl      map_file
    bl      read_ppm
    ADDR    x3, _image
    str     x0, [x3]
    mov     x23, x1                  // width
    mov     x24, x2                  // height

    // milestone 1: report what was found
    ADDR    x1, s_model
    mov     x0, #1
    bl      print_str
    mov     x1, x22
    mov     x0, #1
    bl      print_uint
    ADDR    x1, s_tensors
    mov     x0, #1
    bl      print_str
    mov     x1, #N_TENSORS
    mov     x0, #1
    bl      print_uint
    ADDR    x1, s_used
    mov     x0, #1
    bl      print_str
    ADDR    x1, s_image
    mov     x0, #1
    bl      print_str
    mov     x1, x23
    mov     x0, #1
    bl      print_uint
    ADDR    x1, s_x
    mov     x0, #1
    bl      print_str
    mov     x1, x24
    mov     x0, #1
    bl      print_uint
    ADDR    x1, s_newline
    mov     x0, #1
    bl      print_str
    mov     x0, #0
    SYSCALL SYS_EXIT

.Lusage:
    ADDR    x1, s_usage
    b       _fail

// read_ppm(x0 = mapped file, x1 = size) -> x0 = RGB bytes, x1 = width,
// x2 = height. Binary PPM: "P6", then width, height and maxval, each after
// whitespace or # comments, then one whitespace byte and the pixels.
// Needs 224x224 and maxval 255.
read_ppm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    add     x20, x0, x1              // end of file
    cmp     x1, #2
    b.lo    .Lppm_bad
    ldrh    w2, [x0]
    mov     w3, #0x3650              // "P6", little-endian
    cmp     w2, w3
    b.ne    .Lppm_bad
    add     x19, x0, #2              // cursor
    bl      ppm_uint
    mov     x21, x0                  // width
    bl      ppm_uint
    mov     x22, x0                  // height
    bl      ppm_uint
    cmp     x0, #255                 // maxval: 8 bits per channel
    b.ne    .Lppm_bad
    cmp     x19, x20                 // exactly one whitespace byte
    b.hs    .Lppm_bad
    ldrb    w1, [x19], #1
    cmp     w1, #' '
    b.hi    .Lppm_bad
    cmp     x21, #IMG
    b.ne    .Lppm_bad
    cmp     x22, #IMG
    b.ne    .Lppm_bad
    mul     x0, x21, x22
    add     x0, x0, x0, lsl #1       // width * height * 3 bytes
    sub     x1, x20, x19
    cmp     x1, x0
    b.lo    .Lppm_bad
    mov     x0, x19
    mov     x1, x21
    mov     x2, x22
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
.Lppm_bad:
    ADDR    x1, s_bad_ppm
    b       _fail

// ppm_uint: skip whitespace and # comments at x19 (end x20), then read a
// decimal number of 1 to 6 digits. -> x0, x19 moves past it. Leaf.
ppm_uint:
1:  cmp     x19, x20
    b.hs    .Lppm_bad
    ldrb    w1, [x19]
    cmp     w1, #'#'
    b.ne    3f
2:  add     x19, x19, #1             // a comment runs to the end of the line
    cmp     x19, x20
    b.hs    .Lppm_bad
    ldrb    w1, [x19]
    cmp     w1, #'\n'
    b.ne    2b
    b       1b
3:  cmp     w1, #' '                 // whitespace: any byte up to ' '
    b.hi    4f
    add     x19, x19, #1
    b       1b
4:  mov     x0, #0
    mov     x3, #10
    mov     x4, #0                   // digits read
5:  cmp     x19, x20
    b.hs    6f
    ldrb    w1, [x19]
    sub     w1, w1, #'0'
    cmp     w1, #9
    b.hi    6f
    madd    x0, x0, x3, x1
    add     x19, x19, #1
    add     x4, x4, #1
    cmp     x4, #6
    b.hi    .Lppm_bad
    b       5b
6:  cbz     x4, .Lppm_bad
    ret

// map_file(x0 = path) -> x0 = read-only mapping, x1 = size
map_file:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mov     x1, #0                   // O_RDONLY
    mov     x2, #0
    SYSCALL SYS_OPEN
    b.cs    1f
    mov     x21, x0                  // fd
    mov     x1, #0
    mov     x2, #2                   // SEEK_END -> file size
    SYSCALL SYS_LSEEK
    b.cs    1f
    cbz     x0, 1f                   // empty file
    mov     x20, x0                  // length
    mov     x1, x0
    mov     x0, #0
    mov     x2, #1                   // PROT_READ
    mov     x3, #2                   // MAP_PRIVATE
    mov     x4, x21
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cs    1f
    mov     x1, x20
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
1:  ADDR    x1, s_nofile
    mov     x0, #2
    bl      print_str
    mov     x1, x19
    mov     x0, #2
    bl      print_str
    ADDR    x1, s_newline
    mov     x0, #2
    bl      print_str
    mov     x0, #1
    SYSCALL SYS_EXIT

// fail(x1 = message): print it to stderr and exit 1
.globl _fail
_fail:
    mov     x0, #2
    bl      print_str
    mov     x0, #1
    SYSCALL SYS_EXIT

// print_str(x0 = fd, x1 = NUL-terminated string)
print_str:
    mov     x2, #0
1:  ldrb    w3, [x1, x2]
    cbz     w3, 2f
    add     x2, x2, #1
    b       1b
2:  SYSCALL SYS_WRITE
    ret

// print_uint(x0 = fd, x1 = value)
print_uint:
    sub     sp, sp, #32
    add     x2, sp, #32              // write digits backwards
    mov     x4, #10
1:  udiv    x3, x1, x4
    msub    x5, x3, x4, x1
    add     w5, w5, #'0'
    strb    w5, [x2, #-1]!
    mov     x1, x3
    cbnz    x1, 1b
    mov     x1, x2
    add     x2, sp, #32
    sub     x2, x2, x1
    SYSCALL SYS_WRITE
    add     sp, sp, #32
    ret

// parse_uint(x0 = string) -> x0
parse_uint:
    mov     x1, x0
    mov     x0, #0
    mov     x3, #10
1:  ldrb    w2, [x1], #1
    sub     w2, w2, #'0'
    cmp     w2, #9
    b.hi    2f
    madd    x0, x0, x3, x2
    b       1b
2:  ret

.section __TEXT,__cstring,cstring_literals
path_model:   .asciz "model.safetensors"
s_nofile:     .asciz "resnet: cannot load "
s_bad_ppm:    .asciz "resnet: the image must be a 224x224 binary PPM (P6, maxval 255)\n"
s_usage:      .asciz "usage: resnet [-d N] image.ppm\n"
s_model:      .asciz "model.safetensors: "
s_tensors:    .asciz " tensors, "
s_used:       .asciz " used\n"
s_image:      .asciz "image: "
s_x:          .asciz "x"
s_newline:    .asciz "\n"

.data
.p2align 3
.globl _debug
_debug:       .quad -1               // debug point from -d, or -1
.globl _image
_image:       .quad 0                // RGB bytes in the mapped PPM

// Shared macros and constants. AArch64, Darwin (Mach-O) syntax.

.macro ADDR reg, sym
    adrp    \reg, \sym@PAGE
    add     \reg, \reg, \sym@PAGEOFF
.endm

// Load a 32-bit constant (works for w and x registers).
.macro MOV32 reg, val
    movz    \reg, #((\val) & 0xFFFF)
    movk    \reg, #(((\val) >> 16) & 0xFFFF), lsl #16
.endm

// Load a float constant from its IEEE-754 bit pattern. Clobbers w9.
.macro FCONST sreg, bits
    MOV32   w9, \bits
    fmov    \sreg, w9
.endm

// Raw Darwin syscall: number in x16, trap with svc #0x80. Carry set = error.
.macro SYSCALL num
    mov     x16, #\num
    svc     #0x80
.endm

.equ SYS_EXIT,   1
.equ SYS_WRITE,  4
.equ SYS_OPEN,   5
.equ SYS_MMAP,   197
.equ SYS_LSEEK,  199

// ResNet-18, ImageNet
.equ IMG,        224                 // input width and height
.equ N_CLASSES,  1000
.equ N_CONV,     20                  // stem + 16 block convs + 3 downsample convs
.equ N_DEBUG,    13                  // debug points for -d (see src/main.s)

// _tensors (src/load.s): pointers into model.safetensors.
//   5 per conv, in forward order: conv weight, then BN weight, bias,
//   running_mean, running_var. Then fc.weight and fc.bias.
.equ N_TENSORS,  5*N_CONV + 2
.equ T_FC,       5*N_CONV

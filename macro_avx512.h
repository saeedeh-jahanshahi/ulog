#ifndef MACRO_AVX512_H
#define MACRO_AVX512_H

/* this files contain all macros use in all library */

#define BLOCK_LEN64 8 //number of 64 bits numbers occupy a 512 bits vector
#define BLOCK_LEN32 16 //number of 32 bits numbers occupy a 512 bits vector
#define BLOCK_LEN16 32 //number of 16 bits numbers occupy a 512 bits vector

#ifdef FLOAT //if original numbers are in float
#define LOG32_INT 0xff800000 //integer parts of ulogs32
#define LOG32_FRACTION 0x007fffff //fraction parts of ulogs32
#define LOG32_ONE 0x00800000 //one in ulogs32
#define LOG32_HALF 0x00400000 //0.5 in ulogs32
#define LOG32_NEG_BIAS 0xc0800000 //negative bias in ulogs32
#define LOG32_POS_BIAS 0x3f800000 //positive bias in ulogs32
#define LOG32_FRACTION_LEN 23 //length of fraction in ulogs32
#define LOG16_ONE 0x0080 //one in ulogs16
#define LOG16_HALF 0x0040 //0.5 in ulogs16
#define LOG16_NEG_BIAS 0xc080 //negative bias in ulogs16
#define LOG16_POS_BIAS 0x3f80 //positive bias in ulogs16
#define LOG16_FRACTION_LEN 5 //length of fraction in ulogs16

#else //if original numbers are in double
#define LOG32_INT 0xfff00000 //integer parts of ulogd32
#define LOG32_FRACTION 0x000fffff //fraction parts of ulogd32
#define LOG32_ONE 0x00100000 //one in ulogd32
#define LOG32_HALF 0x00080000 //0.5 in ulogd32
#define LOG32_NEG_BIAS 0xc0100000 //negative bias in ulogd32
#define LOG32_POS_BIAS 0x3ff00000 //positive bias in ulogd32
#define LOG32_FRACTION_LEN 20 //length of fraction in ulogd32
#define LOG16_ONE 0x0010 //one in ulogd16
#define LOG16_HALF 0x0008 //0.5 in ulogd16
#define LOG16_NEG_BIAS 0xc010 //negative bias in ulogd16
#define LOG16_POS_BIAS 0x3ff0 //positive bias in ulogd16
#define LOG16_FRACTION_LEN 4 //length of fraction in ulogd16
#endif

#define LOG32_SIGN 0x80000000 //sign position of ulog32
#define LOG32_ZERO_SIGN 0x7fffffff //zero the sign of ulog32

#define LOG16_SIGN 0x8000 //sign position of ulog16
#define LOG16_ZERO_SIGN 0x7fff //zero the sign of ulog16

#define SEC_HALF_32 0x0000ffff //second half of 32 bits numbers
#define FIRST_HALF_32 0xffff0000 //first half of 32 bits numbers
#define FIRST_HALF_64 0xffffffff00000000 //first half of 64 bits numbers

#define nr 8 //size of nr in gemm for double
#define GEMM_mr 31 //size of mr in gemm for double, float, ulog32, ulog16_bw
#define GEMM_mb 31 //size of mb in gemm for double, float, ulog32, ulog16_bw
#define GEMM_kb 38455 //size of kb in gemm for double

#define LOG32_nr 16 //size of nr in gemm for float and ulog32
#define GEMM_LOG32_kb 21266 //size of kb in gemm for float and ulog32

#define LOG16_nr 32 //size of nr in gemm for ulog16 and ulog16_bw
#define GEMM_LOG16_mr 15 //size of mr in gemm for ulog16
#define GEMM_LOG16_mb 30 //size of mb in gemm for ulog16
#define GEMM_LOG16_kb 96766 //size of kb in gemm for ulog16
#define GEMM_LOG16_bw_kb 95222 //size of kb in gemm for ulog16_bw

#define GEMV_mr 248 //mr size in gemv for double
#define GEMV_mb 248 //mb size in gemv for double
#define GEMV_kb 6023 //kb size in gemv for double

#define GEMV_LOG32_mr 496 //mr size in gemv for float and ulog32
#define GEMV_LOG32_mb 496 //mb size in gemv for float and ulog32
#define GEMV_LOG32_kb 6035 //kb size in gemv for float and ulog32

#define GEMV_LOG16_bw_mr 992 //mr size in gemv for ulog16_bw
#define GEMV_LOG16_bw_mb 992 //mb size in gemv for ulog16_bw
#define GEMV_LOG16_bw_kb 6041 //kb size in gemv for ulog16_bw

#define GEMV_LOG16_mr 448 //mr size in gemv for ulog16
#define GEMV_LOG16_mb 896 //mb size in gemv for ulog16
#define GEMV_LOG16_kb 6688 //kb size in gemv for ulog16

#define GEMV_ROW_LCM 27776 //lcm GEMV_LOG16_bw_mb and GEMV_LOG16_mb

#endif

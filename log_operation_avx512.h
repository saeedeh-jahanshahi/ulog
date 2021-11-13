#ifndef LOG_OPERATION_AVX512_H
#define LOG_OPERATION_AVX512_H

#include <immintrin.h>
#include <stdint.h>

#include "macro_avx512.h"

/* this file include implementation of all ulog operations */

uint32_t vec_reduce_ulog32(__m512i, __mmask16);
uint16_t vec_reduce_ulog16_bw(__m512i, __mmask32); 

//pairwise multiplication of 2 vectors in ulog32
static inline __m512i vec_mult_ulog32(__m512i a, __m512i b)
{
    //specify if either of 2 corresponding numbers in a and b are zero
    //because logarithm of zero in undefined if either of 2 numbers are zero put zero in corresponding position
    __m512i c = _mm512_and_epi32(a, b);
    __mmask16 mask = _mm512_cmpneq_epi32_mask(c, _mm512_setzero_epi32());

    //remove the bias of a by adding negative bias from it
    a = _mm512_maskz_add_epi32(mask, a, _mm512_set1_epi32(LOG32_NEG_BIAS));
    c = _mm512_maskz_add_epi32(mask, a, b);

    return c;
}

//pairwise mulitplication of 2 vectors in ulog16 when cpu supports AVX-512BW
static inline __m512i vec_mult_ulog16_bw(__m512i a, __m512i b)
{
    //specify if either of 2 corresponding numbers in a and b are zero
    //because logarithm of zero in undefined if either of 2 numbers are zero put zero in corresponding position
    __m512i temp = _mm512_setzero_epi32();
    __mmask32 mask = _mm512_cmpneq_epi16_mask(a, temp);
    mask &= _mm512_cmpneq_epi16_mask(b, temp);

    //remove the bias of a by adding negative bias from it
    a = _mm512_maskz_add_epi16(mask, a, _mm512_set1_epi16(LOG16_NEG_BIAS));
    temp = _mm512_maskz_add_epi16(mask, a, b);

    return temp;
}

//multiplication of 2 single numbers in ulog32
static inline uint32_t scalar_mult_ulog32(uint32_t a, uint32_t b)
{
    //test if either of 2 numbers are zero
    if((a == 0) || (b == 0))
    {
        return 0;
    }

    uint32_t c = (a + LOG32_NEG_BIAS) + b;

    return c;
}

//multiplication of 2 single numbers in ulog16
static inline uint16_t scalar_mult_ulog16(uint16_t a, uint16_t b)
{
    //test if either of 2 numbers are zero
    if((a == 0) || (b == 0))
    {
        return 0;
    }

    uint16_t c = (a + LOG16_NEG_BIAS) + b;

    return c;
}

//pairwise addition of 2 vectors in ulog32
static inline __m512i vec_add_ulog32(__m512i a, __m512i b)
{
    //specify the maximum and minimum of each pair number
    __m512i max512i = _mm512_max_epi32(a, b);
    __m512i min512i = _mm512_min_epi32(a, b);

    min512i = _mm512_sub_epi32(max512i, min512i);

    //specify the position of zero in each pair
    __mmask16 mask = _mm512_cmpneq_epi32_mask(min512i, max512i);

    //if minimum is nonzero round it to nearest integer otherwise the result is maximum.
    min512i = _mm512_mask_add_epi32(max512i, mask, min512i, _mm512_set1_epi32(LOG32_HALF));
    min512i = _mm512_mask_srli_epi32(max512i, mask, min512i, LOG32_FRACTION_LEN);

    //shift right one with the amount of corresponding rounded result.
    min512i = _mm512_mask_srlv_epi32(max512i, mask, _mm512_set1_epi32(LOG32_ONE), min512i);

    min512i = _mm512_mask_add_epi32(max512i, mask, max512i, min512i);

    //zero the sign
    min512i = _mm512_mask_and_epi32(max512i, mask, min512i, _mm512_set1_epi32(LOG32_ZERO_SIGN));

    return min512i;
}

//pairwise addition of 2 vectors in ulog16 when cpu support AVX-512BW
static inline __m512i vec_add_ulog16_bw(__m512i a, __m512i b)
{
    //specify the maximum and minimum of each pair number
    __m512i max512i = _mm512_max_epi16(a, b);
    __m512i min512i = _mm512_min_epi16(a, b);

    min512i = _mm512_sub_epi16(max512i, min512i);

    //specify the position of zero in each pair
    __mmask32 mask = _mm512_cmpneq_epi16_mask(min512i, max512i);

    //if minimum is nonzero round it to nearest integer otherwise the result is maximum.
    min512i = _mm512_mask_add_epi16(max512i, mask, min512i, _mm512_set1_epi16(LOG16_HALF));
    min512i = _mm512_mask_srli_epi16(max512i, mask, min512i, LOG16_FRACTION_LEN);

    //shift right one with the amount of corresponding rounded result.
    min512i = _mm512_mask_srlv_epi16(max512i, mask, _mm512_set1_epi16(LOG16_ONE), min512i);

    min512i = _mm512_mask_add_epi16(max512i, mask, max512i, min512i);

    //zero the sign
    min512i = _mm512_mask_and_epi32(max512i, mask, min512i, _mm512_set1_epi16(LOG16_ZERO_SIGN));

    return min512i;
}

//addition of 2 single numbers in ulog32.
static inline uint32_t scalar_add_ulog32(uint32_t a, uint32_t b)
{
    //specify which number is maximum.
    uint32_t temp = 0;
    if(a < b)
    {
        temp = a;
        a = b;
        b = temp;
    }

    temp = a - b;

    //test if minimum is zero or not.
    if(temp == a)
    {
        return a;
    }

    //round the result of subtract to nearest integer.
    temp += LOG32_HALF;
    temp >>= LOG32_FRACTION_LEN;

    //shift right one by the amound of rounded result.
    a += (LOG32_ONE >> temp);

    //zero the sign
    a &= LOG32_ZERO_SIGN;

    return a;
}

//addition of 2 single numbers in ulog16
static inline uint16_t scalar_add_ulog16(uint16_t a, uint16_t b)
{
    //specify which number is maximum.
    uint16_t temp = 0;
    if(a < b)
    {
        temp = a;
        a = b;
        b = temp;
    }

    temp = a - b;

    //test if minimum is zero or not.
    if(temp == a)
    {
        return a;
    }

    //round the result of subtract to nearest integer.
    temp += LOG16_HALF;
    temp >>= LOG16_FRACTION_LEN;
    //printf("the result of rounding is %lX\n", temp);

    //shift right one by the amound of rounded result.
    a += (LOG16_ONE >> temp);

    //zero the sign
    a &= LOG16_ZERO_SIGN;

    return a;
}

//the largest exponent of 2 lower than x
static inline int low_exponent(int x)
{
    int exp = 0;
    while(x > 1)
    {
        x >>= 1;
        exp ++;
    }
    return exp;
}

//pairwise sqrt of 2 vectors in ulog32
static inline __m512i vec_sqrt_ulog32(__m512i a)
{
    //remove the bias of a
    __m512i result = _mm512_add_epi32(a, _mm512_set1_epi32(LOG32_NEG_BIAS));

    result = _mm512_srli_epi32(result, 1);

    //add bias again
    result = _mm512_add_epi32(result, _mm512_set1_epi32(LOG32_POS_BIAS));

    //zero the sign
    result = _mm512_and_epi32(result, _mm512_set1_epi32(LOG32_ZERO_SIGN));

    return result;
}

//pairwise sqrt of 2 vectors in ulog16 when cpu support AVX-512BW
static inline __m512i vec_sqrt_ulog16_bw(__m512i a)
{
    //remove the bias of a
    __m512i result = _mm512_add_epi16(a, _mm512_set1_epi16(LOG16_NEG_BIAS));

    result = _mm512_srli_epi16(result, 1);

    //add bias again
    result = _mm512_add_epi16(result, _mm512_set1_epi16(LOG16_POS_BIAS));

    //zero the sign
    result = _mm512_and_epi32(result, _mm512_set1_epi16(LOG16_ZERO_SIGN));

    return result;
}

//sqrt of 2 single numbers in ulog32
static inline uint32_t scalar_sqrt_ulog32(uint32_t a)
{
    uint32_t result = ((a + LOG32_NEG_BIAS) / 2 + LOG32_POS_BIAS) & LOG32_ZERO_SIGN;
    return result;
}

//sqrt of 2 single numbers in ulog16
static inline uint16_t scalar_sqrt_ulog16(uint16_t a)
{
    uint16_t result = ((a + LOG16_NEG_BIAS) / 2 + LOG16_POS_BIAS) & LOG16_ZERO_SIGN;
    return result;
}

//pairwise exponent of 2 vectors in ulog32
static inline __m512i vec_pow_ulog32(__m512i a, uint32_t p)
{
    //remove the bias of a
    __m512i result = _mm512_add_epi32(a, _mm512_set1_epi32(LOG32_NEG_BIAS));

    result = _mm512_mul_epu32(result, _mm512_set1_epi32(p));

    //add bias again
    result = _mm512_add_epi32(result, _mm512_set1_epi32(LOG32_POS_BIAS));

    //zero the sign
    result = _mm512_and_epi32(result, _mm512_set1_epi32(LOG32_ZERO_SIGN));

    return result;
}

//pairwise exponent of 2 vectors in ulog16 when cpu support AVX-512BW
static inline __m512i vec_pow_ulog16_bw(__m512i a, uint16_t p)
{
    //remove the bias of a
    __m512i result = _mm512_add_epi16(a, _mm512_set1_epi16(LOG16_NEG_BIAS));

    result = _mm512_mulhrs_epi16(result, _mm512_set1_epi16(p));

    //add bias again
    result = _mm512_add_epi16(result, _mm512_set1_epi16(LOG16_POS_BIAS));

    //zero the sign
    result = _mm512_and_epi32(result, _mm512_set1_epi16(LOG16_ZERO_SIGN));

    return result;
}

//exponent of 2 single numbers in ulog32
static inline uint32_t scalar_pow_ulog32(uint32_t a, uint32_t p)
{
    uint32_t result = ((a + LOG32_NEG_BIAS) * p + LOG32_POS_BIAS) & LOG32_ZERO_SIGN;
    return result;
}

//exponent of 2 single numbers in ulog16
static inline uint32_t scalar_pow_ulog16(uint16_t a, uint16_t p)
{
    uint16_t result = ((a + LOG16_NEG_BIAS) * p + LOG16_POS_BIAS) & LOG16_ZERO_SIGN;
    return result;
}

#endif

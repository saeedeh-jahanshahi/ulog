#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

#include "log_operation_avx512.h"
#include "general_avx512.h"
#include "macro_avx512.h"

//reduction by addition of a vector in ulog32
uint32_t vec_reduce_ulog32(__m512i arr, __mmask16 mask)
{
    //specify the maximum of whole vector
    uint32_t maxi = _mm512_mask_reduce_max_epu32(mask, arr);

    //subtract maximum of vector from all ulog32 numbers in vector
    __m512i temp_512 = _mm512_maskz_set1_epi32(mask, maxi);
    arr = _mm512_maskz_sub_epi32(mask, temp_512, arr);
    
    //rounding the result of subtract to nearest integer
    arr = _mm512_maskz_add_epi32(mask, arr, _mm512_set1_epi32(LOG32_HALF));
    arr = _mm512_maskz_srli_epi32(mask, arr, LOG32_FRACTION_LEN);

    //shift to right one by the amount of the corresponding nearest integer
    temp_512 = _mm512_maskz_set1_epi32(mask, LOG32_ONE);
    arr = _mm512_maskz_srlv_epi32(mask, temp_512, arr);

    uint32_t result = _mm512_reduce_add_epi32(arr);

    //seperate the fraction part
    uint32_t fr = result & LOG32_FRACTION;

    //seperate the integer part
    result &= LOG32_INT;
    result >>= LOG32_FRACTION_LEN;

    //find the largest exponent of 2 smaller than result
    result = low_exponent(result);

    fr >>= result;

    //move the integer part to its position to add the fraction part to it
    result <<= LOG32_FRACTION_LEN;

    maxi = maxi + result + fr;

    return maxi;
}

//reduction by addition of a vector in ulog16
uint16_t vec_reduce_ulog16_bw(__m512i arr, __mmask32 mask)
{
    arr = _mm512_mask_blend_epi16(mask, _mm512_setzero_epi32(), arr);

    //specify the maximum of whole vector
    uint32_t maxi1 = _mm512_reduce_max_epu32(arr);
    uint16_t maxi = (uint16_t)(maxi1 >> BLOCK_LEN32);
    __m512i temp = _mm512_and_epi32(arr, _mm512_set1_epi32(SEC_HALF_32));
    uint16_t maxi2 = (uint16_t)_mm512_reduce_max_epu32(temp);
    maxi = (maxi > maxi2) ? maxi : maxi2;

    temp = _mm512_maskz_set1_epi16(mask, maxi);

    arr = _mm512_maskz_sub_epi16(mask, temp, arr);

    //rounding the result of subtract to nearest integer
    arr = _mm512_maskz_add_epi16(mask, arr, _mm512_set1_epi16(LOG16_HALF));
    arr = _mm512_maskz_srli_epi16(mask, arr, LOG16_FRACTION_LEN);

    //shift to right one by the amount of the corresponding nearest integer
    temp = _mm512_maskz_set1_epi16(mask, LOG16_ONE);
    arr = _mm512_maskz_srlv_epi16(mask, temp, arr);

    //reduction by addition the total ulog16 elements in arr
    temp = _mm512_and_epi32(arr, _mm512_set1_epi32(FIRST_HALF_32));
    uint32_t result = _mm512_reduce_add_epi32(temp);
    arr = _mm512_slli_epi32(arr, BLOCK_LEN32);
    result += _mm512_reduce_add_epi32(arr);

    //seperate the fraction part
    uint32_t fr = result & LOG32_FRACTION;

    //seperate the integer part
    result &= LOG32_INT;
    result >>= LOG32_FRACTION_LEN;

    //find the largest exponent of 2 smaller than result
    result = low_exponent(result);

    fr >>= (result + BLOCK_LEN32);

    //move the integer part to its position to add the fraction part to it
    result <<= (LOG32_FRACTION_LEN - BLOCK_LEN32);

    maxi = maxi + result + fr;

    return (uint16_t)maxi;
}

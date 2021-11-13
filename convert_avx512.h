#ifndef CONVERT_AVX512_H
#define CONVERT_AVX512_H

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

#include "macro_avx512.h"
/* this file contains all definitions and inline conversions functions for ulog data types */

//convert a vector of 64 bits integers to ulog16
static inline __m128i vec_convert_log64_log16(__m512i src)
{
    //shift right each 64 bits integer 48 bits to right
    src = _mm512_srli_epi64(src, 3 * BLOCK_LEN32);

    //truncate the zero bits of each 64 bits numbers and convert it to 16 bits.
    __m128i temp = _mm512_cvtepi64_epi16(src);

    return temp;
}

//convert a vector of 64 bits integer to ulog16
static inline __m256i vec_convert_log64_log32(__m512i src)
{
    //shift right each 64 bits integer 32 bits to right
    src = _mm512_srli_epi64(src, 2 * BLOCK_LEN32);

    //truncate the zero bits of each 64 bits numbers and convert it to 32 bits.
    __m256i temp = _mm512_cvtepi64_epi32(src);

    return temp;
}

//convert a vector of 32 bits integer to ulog16
static inline __m256i vec_convert_log32_log16(__m512i src)
{
    //shift right each 32 bits integer 16 bits to right
    src = _mm512_srli_epi32(src, BLOCK_LEN32);

    //truncate the zero bits of each 32 bits numbers and convert it to 16 bits.
    __m256i temp = _mm512_cvtepi32_epi16(src);

    return temp;
}

//convert a vector of 8 double numbers to float
static inline __m256 vec_convert_double_float(__m512d x)
{
    return _mm512_cvtpd_ps(x);
}

//convert a vector of ulog16 to 64 bits integer
static inline __m512i vec_convert_log16_log64(__m128i part_128i)
{
    //zero extend each ulog16
    __m512i part_512i = _mm512_cvtepu16_epi64(part_128i);

    //shift left each ulog16 48 bits and convert it to 64 bits integers
    part_512i = _mm512_slli_epi64(part_512i, 3 * BLOCK_LEN32);

    return  part_512i;
}

//convert a vector of 8 ulog32 numbers to 64 bits integer
static inline __m512i vec_convert_log32_log64(__m256i part_256i)
{
    //zero extend each ulog32
    __m512i part_512i = _mm512_cvtepu32_epi64(part_256i);

    //shift left each ulog32 32 bits and convert it to 64 bits integers
    part_512i = _mm512_slli_epi64(part_512i, BLOCK_LEN16);

    return part_512i;
}

//convert a vector of 16 ulog16 numbers to 32 bits integer
static inline __m512i vec_convert_log16_log32(__m256i part_256i)
{
    //zero extend each ulog16
    __m512i part_512i = _mm512_cvtepu16_epi32(part_256i);

    //shift left each ulog16 32 bits and convert it to 32 bits integers
    part_512i = _mm512_slli_epi32(part_512i, BLOCK_LEN16);

    return part_512i;
}

//converst a single ulog16 number to double
static inline double scalar_convert_log16_double(uint16_t x)
{
    uint64_t x64 = (uint64_t)x;
    x64 <<= 48;
    double *p = (double *)&x64;

    return *p;
}

//convert a single ulog32 number to double
static inline double scalar_convert_log32_double(uint32_t x)
{
    uint64_t x64 = (uint64_t)x;
    x64 <<= 32;
    double *p = (double *)&x64;

    return *p;
}

//convert a single ulog16 number to float
static inline float scalar_convert_log16_float(uint16_t x)
{
    uint32_t x32 = (uint32_t)x;
    x32 <<= 16;
    float *p = (float *)&x32;

    return *p;
}

//convert a single float number to double
static inline double scalar_convert_float_double(float x)
{
    return (double)x;
}

static inline __m512i vec_zero_extend_log16_log32(__m256i part_256i)
{
    return _mm512_cvtepu16_epi32(part_256i);
}

static inline __m512i vec_zero_extend_log16_log64(__m128i part_128i)
{
    return _mm512_cvtepu16_epi64(part_128i);
}

//convert a vector of ulog16 numbers to 2 vectors with conversion of each 16 bits ulog16 to 32 bits by zero extending
static inline __m512i vec_zero_extend512i_log16_log32(__m512i *a1)
{
    //seperate the second part
    __m512i a2 = _mm512_and_epi32(*a1, _mm512_set1_epi32(FIRST_HALF_32));

    //seperate the first part
    *a1 = _mm512_slli_epi32(*a1, 16);

    return a2;
}

//merge 2 vectors with 32 bits zero extended ulog16 into one vector with 16 bits ulog16 numbers
static inline __m512i vec_merge512i_log16(__m512i x1, __m512i x2)
{
    x1 = _mm512_and_epi32(x1, _mm512_set1_epi32(FIRST_HALF_32));
    x2 = _mm512_srli_epi32(x2, 16);
    return _mm512_or_epi32(x1, x2);
}

static inline float scalar_convert_double_float(double x)
{
    return (float)x;
}

static inline uint32_t scalar_convert_double_log32(double x)
{
    uint32_t *r = (uint32_t *)&x;
    return r[1];
}

static inline uint16_t scalar_convert_double_log16(double x)
{
    uint16_t *r = (uint16_t *)&x;
    return r[3];
}

static inline uint16_t scalar_convert_float_log16(float x)
{
    uint16_t *r = (uint16_t *)&x;
    return r[1];
}

void array_convert_double_log16(double *, uint16_t *, uint32_t);
void array_convert_double_log32(double *, uint32_t *, uint32_t);
void array_convert_log16_double(double *, uint16_t *, uint32_t);
void array_convert_log32_double(double *, uint32_t *, uint32_t);
void array_convert_double_float(double *, float *, uint32_t);
void array_convert_float_log16(float *, uint16_t *, uint32_t);
void array_convert_float_double(double *, float *, uint32_t);
void array_convert_log16_float(float *, uint16_t *, uint32_t);

#endif

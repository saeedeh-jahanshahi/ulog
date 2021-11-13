#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

#include "convert_avx512.h"
#include "macro_avx512.h"

//function for conversion of an array of double to ulog16
void array_convert_double_log16(double *x, uint16_t *parts, uint32_t size)
{
    __m512i src;
    __m128i part_128i;

    for(int i = 0; i < size; i += BLOCK_LEN64)
    {
        src = _mm512_loadu_si512((void *)(x + i));

        //a vector of 8 double numbers send to vec_convert_log64_log16 function to convert to ulog16
        part_128i = vec_convert_log64_log16(src);
        _mm_storeu_si128((__m128i *)(parts + i), part_128i);
    }
}

//function for conversion of an array of double to ulog32
void array_convert_double_log32(double *x, uint32_t *parts, uint32_t size)
{
    __m512i src;
    __m256i part_256i;

    for(int i = 0; i < size; i += BLOCK_LEN64)
        src = _mm512_loadu_si512((void *)(x + i));

        //a vector of 8 double numbers send to vec_convert_log64_log32 function to convert to ulog32
        part_256i = vec_convert_log64_log32(src);
        _mm256_storeu_si256((__m256i *)(parts + i), part_256i);
    }
}

//function for conversion of an array of float to double
void array_convert_float_double(double *result, float *x, uint32_t size)
{
    __m256 src;
    __m512d result_512d;

    for(int i = 0; i < size; i += BLOCK_LEN64)
    {
        src = _mm256_loadu_ps((void *)(x + i));

        //a vector of 16 float numbers send to _mm512_cvtps_pd which is an intel intrinsic function to convert to double
        result_512d = _mm512_cvtps_pd(src);
        _mm512_storeu_pd((void *)(result + i), result_512d);
    }
}

//function for conversion of float to ulog16
void array_convert_float_log16(float *x, uint16_t *parts, uint32_t size)
{
    __m512i src;
    __m256i part_256i;

    for(int i = 0; i < size; i += BLOCK_LEN32)
    {
        src = _mm512_loadu_si512((void *)(x + i));

        //a vector of 16 float numbers send to vec_convert_log32_log16 to convert to ulog16
        part_256i = vec_convert_log32_log16(src);
        _mm256_storeu_si256((__m256i *)(parts + i), part_256i);
    }
}

//function for converting an array of ulog16 to double
void array_convert_log16_double(double *d, uint16_t *parts, uint32_t size)
{
    int i = 0;
    __m128i part_128i;
    __m512i part_512i;
    __m512d part_512d;

    for(i = 0; i < size; i+= BLOCK_LEN64)
    {
        //double occupy 4 times memory than ulog16. therefore for reverse conversion just 8*16=128 bits data read.
        part_128i = _mm_loadu_si128((__m128i *)(parts + i));

        //vec_convert_log16_log64 converts each vector of 8 ulog16 numbers to ulog64. the output has the type __m512i.
        part_512i = vec_convert_log16_log64(part_128i);

        //this intel intrinsic function has no cost and just convert __m512i to __m512d
        part_512d = _mm512_castsi512_pd(part_512i);
        _mm512_storeu_pd((double *)(d + i), part_512d);

    }
}

//function for converting an array of ulog32 to double
void array_convert_log32_double(double *d, uint32_t *parts, uint32_t size)
{
    int i = 0;
    __m256i part_256i;
    __m512i part_512i;
    __m512d part_512d;

    for(i = 0; i < size; i+= BLOCK_LEN64)
    {
        //double occupy 2 times memory than ulog32. therefore for reverse conversion just 8*32=256 bits data read.
        part_256i = _mm256_loadu_si256((__m256i *)(parts + i));

        //vec_convert_log32_log64 converts each vector of 8 ulog16 numbers to ulog64. the output has the type __m512i.
        part_512i = vec_convert_log32_log64(part_256i);

        //this intel intrinsic function has no cost and just convert __m512i to __m512d
        part_512d = _mm512_castsi512_pd(part_512i);
        _mm512_storeu_pd((double *)(d + i), part_512d);

    }
}

//function for converting an array of ulog16 to float.
void array_convert_log16_float(float *d, uint16_t *parts, uint32_t size)
{
    int i = 0;
    __m256i part_256i;
    __m512i part_512i;
    __m512 part_512;

    for(i = 0; i < size; i+= BLOCK_LEN32)
    {
        //float occupy 2 times memory than ulog16. therefore for reverse conversion just 16*16=256 bits data read.
        part_256i = _mm256_loadu_si256((__m256i *)(parts + i));

        //vec_convert_log16_log32 converts each vector of 8 ulog16 numbers to ulog32. the output has the type __m512i.
        part_512i = vec_convert_log16_log32(part_256i);

        //this intel intrinsic function has no cost and just convert __m512i to __m512d
        part_512 = _mm512_castsi512_ps(part_512i);
        _mm512_storeu_ps((float *)(d + i), part_512);

    }
}

//function for converting an array of double to float
void array_convert_double_float(double *x, float *r, uint32_t size)
{
    __m512d src;
    __m256 float_data;

    for(int i = 0; i < size; i += BLOCK_LEN64)
    {
        src = _mm512_loadu_pd((void *)(x + i));

        //vec_convert_double_float converts a vector of 8 double numbers to float numbers
        float_data = vec_convert_double_float(src);
        _mm256_storeu_ps((float *)(r + i), float_data);
    }
}

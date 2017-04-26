
#ifndef BLISLAB_DGEMM_KERNEL_H
#define BLISLAB_DGEMM_KERNEL_H

#include "bl_config.h"

#include <stdio.h>
#include <immintrin.h> // AVX


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long dim_t;

typedef union {
    __m256d v;
    __m256i u;
    double d[ 4 ];
} v4df_t;


typedef union {
    __m128i v;
    int d[ 4 ];
} v4li_t;

struct aux_s {
    double *b_next;
    float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

void bl_dgemm_asm_8x4( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_8x4_beta0( int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data );

void bl_dgemm_asm_8x4_mulstrassen( int k,
                                   double *a,
                                   double *b,
                                   unsigned long long len_c, unsigned long long ldc,
                                   double **c_list, double *alpha_list,
                                   aux_t* data );

static void (*bl_micro_kernel) (
        int k,
        double *a,
        double *b,
        double *c,
        unsigned long long ldc,
        aux_t* data
        ) = {
        BL_MICRO_KERNEL
};

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

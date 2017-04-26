#ifndef BLISLAB_DGEMM_H
#define BLISLAB_DGEMM_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <immintrin.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>


// Determine the target operating system
#if defined(_WIN32) || defined(__CYGWIN__)
#define BL_OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define BL_OS_OSX 1
#elif defined(__ANDROID__)
#define BL_OS_ANDROID 1
#elif defined(__linux__)
#define BL_OS_LINUX 1
#elif defined(__bgq__)
#define BL_OS_BGQ 1
#elif defined(__bg__)
#define BL_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__)
#define BL_OS_BSD 1
#else
#error "Cannot determine operating system"
#endif

// gettimeofday() needs this.
#if BL_OS_WINDOWS
  #include <time.h>
#elif BL_OS_OSX
  #include <mach/mach_time.h>
#else
  #include <sys/time.h>
  #include <time.h>
#endif

#include "bl_config.h"

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define A( i, j )     A[ (j)*lda + (i) ]
#define B( i, j )     B[ (j)*ldb + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]
#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]
#define C( i, j )     C[ (j)*ldc + (i) ]

void bl_dgemm(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_beta0(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_abc(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_ab(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

void bl_dgemm_strassen_naive(
        int    m,
        int    n,
        int    k,
        double *A,
        int    lda,
        double *B,
        int    ldb,
        double *C,
        int    ldc
        );

// XB = XA
void mkl_copym(
    int m,
    int n,
    double *XA,
    int lda,
    double *XB,
    int ldb
    );

// XB = XB + alpha * XA
void mkl_axpym(
        int m,
        int n,
        double *buf_alpha,
        double *XA,
        int lda,
        double *XB,
        int ldb
        );

double *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        );

void bl_free( void *p );

void bl_printmatrix(
        double *A,
        int    lda,
        int    m,
        int    n
        );

double bl_clock( void );
double bl_clock_helper();

void bl_dgemm_ref(
    int    m,
    int    n,
    int    k,
    double *XA,
    int    lda,
    double *XB,
    int    ldb,
    double *XC,
    int    ldc
    );

void bl_get_range( int n, int bf, int* start, int* end );

void bl_acquire_mpart( 
        int m,
        int n,
        double *src_buff,
        int lda,
        int x,
        int y,
        int i,
        int j,
        double **dst_buff
        );


void bl_compare_error( int ldc, int ldc_ref, int m, int n, double *C, double *C_ref );

void bl_dynamic_peeling( int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc, int dim1, int dim2, int dim3 );

int bl_read_nway_from_env( char* env );

static double *glob_packA=NULL, *glob_packB=NULL;

void bl_finalize();

void bl_malloc_packing_pool( double **packA, double **packB, int n, int bl_ic_nt );

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

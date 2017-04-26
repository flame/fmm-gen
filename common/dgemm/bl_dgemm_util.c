#include "bl_dgemm.h"


// XB = XA
void mkl_copym(
    int m,
    int n,
    double *XA,
    int lda,
    double *XB,
    int ldb
    )
{
    int ii;
    int incx = 1;
    int incy = 1;

    #pragma omp parallel for schedule( dynamic )
    for ( ii = 0; ii < n; ii ++ ) {
        double *cur_buf_a, *cur_buf_b;
        cur_buf_a = &XA[ ii * lda ];
        cur_buf_b = &XB[ ii * ldb ];
        dcopy_( &m, cur_buf_a, &incx, cur_buf_b, &incy);
    }

}

// XB = XB + alpha * XA
void mkl_axpym(
    int m,
    int n,
    double *buf_alpha,
    double *XA,
    int lda,
    double *XB,
    int ldb
    )
{
    int ii, jj;
    int incx = 1;
    int incy = 1;

    #pragma omp parallel for schedule( dynamic )
    for ( ii = 0; ii < n; ii ++ ) {
        double *cur_buf_a, *cur_buf_b;
        cur_buf_a = &XA[ ii * lda ];
        cur_buf_b = &XB[ ii * ldb ];
        daxpy_( &m, buf_alpha, cur_buf_a, &incx, cur_buf_b, &incy );
    }

    //for ( ii = 0; ii < n; ii ++ ) {
    //    for ( jj = 0; jj < m; jj ++ ) {
    //        XB[ ii * ldb + jj ] += *buf_alpha * XA[ ii * lda + jj ];
    //    }
    //}


}

double *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    double *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}

void bl_free( void *p ) {
    free( p );
}

void bl_dgemm_printmatrix(
        double *A,
        int    lda,
        int    m,
        int    n
        )
{
    int    i, j;
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            printf("%2.2lf\t", A[j * lda + i]);
        }
        printf("\n");
    }
}


/*
 * The timer functions are copied directly from BLIS 0.2.0
 *
 */
static double gtod_ref_time_sec = 0.0;

double bl_clock( void )
{
	return bl_clock_helper();
}

#if BL_OS_WINDOWS
// --- Begin Windows build definitions -----------------------------------------

double bl_clock_helper()
{
    LARGE_INTEGER clock_freq = {0};
    LARGE_INTEGER clock_val;
    BOOL          r_val;

    r_val = QueryPerformanceFrequency( &clock_freq );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    r_val = QueryPerformanceCounter( &clock_val );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    return ( ( double) clock_val.QuadPart / ( double) clock_freq.QuadPart );
}

// --- End Windows build definitions -------------------------------------------
#elif BL_OS_OSX
// --- Begin OSX build definitions -------------------------------------------

double bl_clock_helper()
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info( &timebase );

    uint64_t nsec = mach_absolute_time();

    double the_time = (double) nsec * 1.0e-9 * timebase.numer / timebase.denom;

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = the_time;

    return the_time - gtod_ref_time_sec;
}

// --- End OSX build definitions ---------------------------------------------
#else
// --- Begin Linux build definitions -------------------------------------------

double bl_clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#endif





// Code for work assignments
void bl_get_range( int n, int bf, int* start, int* end )
{
	//int      n_way      = thread->n_way;
	//int      work_id    = thread->work_id;
    int      n_way      = omp_get_num_threads();
    int      work_id    = omp_get_thread_num();


    //printf( "n: %d, bf: %d, start: %d, end: %d, n_way: %d, work_id: %d\n,", n, bf, *start, *end, n_way, work_id );

	int      all_start  = 0;
	int      all_end    = n;

	int      size       = all_end - all_start;

	int      n_bf_whole = size / bf;
	int      n_bf_left  = size % bf;

	int      n_bf_lo    = n_bf_whole / n_way;
	int      n_bf_hi    = n_bf_whole / n_way;

	// In this function, we partition the space between all_start and
	// all_end into n_way partitions, each a multiple of block_factor
	// with the exception of the one partition that recieves the
	// "edge" case (if applicable).
	//
	// Here are examples of various thread partitionings, in units of
	// the block_factor, when n_way = 4. (A '+' indicates the thread
	// that receives the leftover edge case (ie: n_bf_left extra
	// rows/columns in its sub-range).
	//                                        (all_start ... all_end)
	// n_bf_whole  _left  hel  n_th_lo  _hi   thr0  thr1  thr2  thr3
	//         12     =0    f        0    4      3     3     3     3
	//         12     >0    f        0    4      3     3     3     3+
	//         13     >0    f        1    3      4     3     3     3+
	//         14     >0    f        2    2      4     4     3     3+
	//         15     >0    f        3    1      4     4     4     3+
	//         15     =0    f        3    1      4     4     4     3 
	//
	//         12     =0    t        4    0      3     3     3     3
	//         12     >0    t        4    0      3+    3     3     3
	//         13     >0    t        3    1      3+    3     3     4
	//         14     >0    t        2    2      3+    3     4     4
	//         15     >0    t        1    3      3+    4     4     4
	//         15     =0    t        1    3      3     4     4     4

	// As indicated by the table above, load is balanced as equally
	// as possible, even in the presence of an edge case.

	// First, we must differentiate between cases where the leftover
	// "edge" case (n_bf_left) should be allocated to a thread partition
	// at the low end of the index range or the high end.

		// Notice that if all threads receive the same number of
		// block_factors, those threads are considered "high" and
		// the "low" thread group is empty.
		int n_th_lo = n_bf_whole % n_way;
		//int n_th_hi = n_way - n_th_lo;

		// If some partitions must have more block_factors than others
		// assign the slightly larger partitions to lower index threads.
		if ( n_th_lo != 0 ) n_bf_lo += 1;

		// Compute the actual widths (in units of rows/columns) of
		// individual threads in the low and high groups.
		int size_lo = n_bf_lo * bf;
		int size_hi = n_bf_hi * bf;

		// Precompute the starting indices of the low and high groups.
		int lo_start = all_start;
		int hi_start = all_start + n_th_lo * size_lo;

		// Compute the start and end of individual threads' ranges
		// as a function of their work_ids and also the group to which
		// they belong (low or high).
		if ( work_id < n_th_lo )
		{
			*start = lo_start + (work_id  ) * size_lo;
			*end   = lo_start + (work_id+1) * size_lo;
		}
		else // if ( n_th_lo <= work_id )
		{
			*start = hi_start + (work_id-n_th_lo  ) * size_hi;
			*end   = hi_start + (work_id-n_th_lo+1) * size_hi;

			// Since the edge case is being allocated to the high
			// end of the index range, we have to advance the last
			// thread's end.
			if ( work_id == n_way - 1 ) *end += n_bf_left;
		}
	
}


// Split into m x n, get the subblock starting from i th row and j th column.
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
        )
{
    //printf( "m: %d, n: %d, lda: %d, x: %d, y: %d, i: %d, j: %d\n", m, n, lda, x, y, i, j );
    *dst_buff = &src_buff[ m / x * i + ( n / y * j ) * lda ]; //src( m/x*i, n/y*j )
}

#define USE_SET_DIFF 1
#define TOLERANCE 1E-10
void bl_compare_error( int ldc, int ldc_ref, int m, int n, double *C, double *C_ref )
{
    int    i, j;
    #pragma omp parallel for private( j )
    for ( i = 0; i < m; i ++ ) {
        for ( j = 0; j < n; j ++ ) {
            if ( fabs( C( i, j ) - C_ref( i, j ) ) > TOLERANCE ) {
                printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C( i, j ), C_ref( i, j ) );
                break;
            }
        }
    }
}

void bl_dynamic_peeling( int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc, int dim1, int dim2, int dim3 ) {
    int mr = m % dim1;
    int kr = k % dim2;
    int nr = n % dim3;
    int ms = m - mr;
    int ns = n - nr;
    int ks = k - kr;
    double *A_extra, *B_extra, *C_extra;

    // Adjust part handled by fast matrix multiplication.
    // Add far column of A outer product bottom row B
    if ( kr > 0 ) {
        // In Strassen, this looks like C([1, 2], [1, 2]) += A([1, 2], 3) * B(3, [1, 2])
        A_extra = &A[ 0  + ks * lda ];//ms * kr
        B_extra = &B[ ks + 0  * ldb ];//kr * ns
        C_extra = &C[ 0  + 0  * ldc ];//ms * ns
        if ( ms > 0 && ns > 0 )
            bl_dgemm( ms, ns, kr, A_extra, lda, B_extra, ldb, C_extra, ldc );
    }

    // Adjust for far right columns of C
    if ( nr > 0 ) {
        // In Strassen, this looks like C(:, 3) = A * B(:, 3)
        B_extra = &B[ 0 + ns * ldb ];//k * nr
        C_extra = &C[ 0 + ns * ldc ];//m * nr
        bl_dgemm( m, nr, k, A, lda, B_extra, ldb, C_extra, ldc );
    }

    // Adjust for bottom rows of C
    if ( mr > 0 ) {
        // In Strassen, this looks like C(3, [1, 2]) = A(3, :) * B(:, [1, 2])
        double *A_extra = &A[ ms + 0 * lda ];// mr * k
        double *B_extra = &B[ 0  + 0 * ldb ];// k  * ns
        double *C_extra = &C[ ms + 0 * ldc ];// mr * ns
        if ( ns > 0 )
            bl_dgemm( mr, ns, k, A_extra, lda, B_extra, ldb, C_extra, ldc );
    }
}

int bl_read_nway_from_env( char* env )
{
    int number = 1;
    char* str = getenv( env );
    if( str != NULL )
    {
        number = strtol( str, NULL, 10 );
    }
    return number;
}

void bl_finalize( ) {
    if ( glob_packA != NULL ) {
        bl_free( glob_packA );
    }
    if ( glob_packB != NULL ) {
        bl_free( glob_packB );
    }
}

void bl_malloc_packing_pool( double **packA, double **packB, int n, int bl_ic_nt ) {
    *packA  = NULL;
    if ( glob_packA == NULL ) {
        *packA = bl_malloc_aligned( DGEMM_KC, ( ( DGEMM_MC + 1 ) * bl_ic_nt ), sizeof(double) );
        glob_packA = *packA;
    } else {
        *packA = glob_packA;
    }
    *packB  = NULL;
    if ( glob_packB == NULL ) {
        if ( DGEMM_NC > n ) {
            *packB  = bl_malloc_aligned( DGEMM_KC, ( ( n + DGEMM_NR )            ), sizeof(double) );
        } else {
            *packB  = bl_malloc_aligned( DGEMM_KC, ( ( DGEMM_NC + 1 )            ), sizeof(double) );
        }
        glob_packB = *packB;
    } else {
        *packB = glob_packB;
    }
}



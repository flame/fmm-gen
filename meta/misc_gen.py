'''
   fmm-gen    
   Generating Families of Practical Fast Matrix Multiplication Algorithms

   Copyright (C) 2017, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import sys

from common_mix import write_line, write_break

def gen_test_makefile( file_name, lib_name, test_name ):
    myfile = open( file_name, 'w' )
    myfile.write( \
'''\
export BLISLAB_DIR=..

ifeq ($(BLISLAB_USE_INTEL),true)
include $(BLISLAB_DIR)/make.inc.files/make.intel.inc
else
include $(BLISLAB_DIR)/make.inc.files/make.gnu.inc
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/{0}.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/{0}.so

BLISLAB_TEST_CC_SRC= \\
									 {1}

BLISLAB_TEST_CPP_SRC=\\


OTHER_DEP = \\
			                $(LIBBLISLAB) \\

BLISLAB_TEST_EXE= $(BLISLAB_TEST_CC_SRC:.c=.x) $(BLISLAB_TEST_CPP_SRC:.cpp=.x)

all: $(BLISLAB_TEST_EXE)

clean:
	rm -f $(BLISLAB_TEST_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

%.x: %.cpp $(OTHER_DEP)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
'''.format( lib_name, test_name ) )

def gen_project_makefile( file_name, lib_name, src_filename, kernel_filename, ker_header_filename, test_makefile_name, is_par=True ):
    if is_par:
        par_cflags='CFLAGS += -D_PARALLEL_'
    else:
        par_cflags=''
    myfile = open( file_name, 'w' )
    myfile.write( \
'''\
ifeq ($(BLISLAB_USE_INTEL),true)
include $(BLISLAB_DIR)/make.inc.files/make.intel.inc
else
include $(BLISLAB_DIR)/make.inc.files/make.gnu.inc
endif

$(info * Using CFLAGS=$(CFLAGS))
$(info * Using LDFLAGS=$(LDFLAGS))
$(info * Using LDLIBS=$(LDLIBS))

{5}

LIBBLISLAB = $(BLISLAB_DIR)/lib/{0}.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/{0}.so

FRAME_CC_SRC=  \\
							 dgemm/my_dgemm.c \\
							 dgemm/bl_dgemm_ref.c \\
							 dgemm/bl_dgemm_util.c \\
                             {1}

FRAME_CPP_SRC= \\

KERNEL_SRC=    \\
							 kernels/bl_dgemm_asm_8x4.c \\
							 kernels/bl_dgemm_asm_8x4_mulstrassen.c \\
                             {2}

OTHER_DEP = \\
			                 include/bl_dgemm.h \\
                             {3}

BLISLAB_OBJ=$(FRAME_CC_SRC:.c=.o) $(FRAME_CPP_SRC:.cpp=.o) $(KERNEL_SRC:.c=.o) $(FRAME_CC_SRC_S:.c=.os) $(KERNEL_SRC_S:.c=.os)

all: $(LIBBLISLAB) $(SHAREDLIBBLISLAB) TESTBLISLAB

TESTBLISLAB: $(LIBBLISLAB)
	cd $(BLISLAB_DIR)/test && $(MAKE) -f {4} && cd $(BLISLAB_DIR)

$(LIBBLISLAB): $(BLISLAB_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(BLISLAB_OBJ)
	$(RANLIB) $@

$(SHAREDLIBBLISLAB): $(BLISLAB_OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $(BLISLAB_OBJ) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
# ---------------------------------------------------------------------------

clean:
	-rm $(BLISLAB_OBJ) $(LIBBLISLAB) $(SHAREDLIBBLISLAB) dgemm/*~ kernels/*~ kernels/*.o test/*~ include/*~ *~ make.inc.files/*~
	$(MAKE) clean -f {4} -C test
'''.format( lib_name, src_filename, kernel_filename, ker_header_filename, test_makefile_name, par_cflags ) )

def gen_testfile( file_name, pack_type ):
    myfile = open( file_name, 'w' )
    myfile.write( \
'''\
#include "bl_dgemm.h"

void test_bl_dgemm( int m, int n, int k )
{{
    int    i, j, p, nx;
    double *A, *B, *C, *C_ref;
    double tmp, error, flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;

    A    = (double*)malloc( sizeof(double) * m * k );
    B    = (double*)malloc( sizeof(double) * k * n );

    lda     = m;
    ldb     = k;
    ldc     = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    ldc_ref = ( ( m - 1 ) / DGEMM_MR + 1 ) * DGEMM_MR;
    C     = bl_malloc_aligned( ldc, n + 4, sizeof(double) );
    C_ref = bl_malloc_aligned( ldc, n + 4, sizeof(double) );

    nrepeats = 3;

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {{
        for ( i = 0; i < m; i ++ ) {{
            A( i, p ) = (double)( drand48() );
        }}
    }}
    for ( j = 0; j < n; j ++ ) {{
        for ( p = 0; p < k; p ++ ) {{
            B( p, j ) = (double)( drand48() );
        }}
    }}

    for ( j = 0; j < n; j ++ ) {{
        for ( i = 0; i < m; i ++ ) {{
            C_ref( i, j ) = (double)( 0.0 );
                C( i, j ) = (double)( 0.0 );
        }}
    }}

    for ( i = 0; i < nrepeats; i ++ ) {{
        bl_dgemm_beg = bl_clock();
        {{
            bl_dgemm_strassen_{0}( m, n, k, A, lda, B, ldb, C, ldc );
        }}
        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

        if ( i == 0 ) {{
            bl_dgemm_rectime = bl_dgemm_time;
        }} else {{
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }}
    }}

    for ( i = 0; i < nrepeats; i ++ ) {{
        ref_beg = bl_clock();
        {{
            bl_dgemm( m, n, k, A, lda, B, ldb, C_ref, ldc_ref );
        }}
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {{
            ref_rectime = ref_time;
        }} else {{
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }}
    }}

    bl_compare_error( ldc, ldc_ref, m, n, C, C_ref );

    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\\t %5d\\t %5d\\t %5.2lf\\t %5.2lf\\n",
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime );
    //printf( "%5d\\t %5d\\t %5d\\t %5.2lf\\n",
    //        m, n, k, flops / bl_dgemm_rectime );



    fflush(stdout);

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
}}

int main( int argc, char *argv[] )
{{
    int    m, n, k;

    if ( argc != 4 ) {{
        printf( "Error: require 3 arguments, but only %d provided.\\n", argc - 1 );
        exit( 0 );
    }}

    sscanf( argv[ 1 ], "%d", &m );
    sscanf( argv[ 2 ], "%d", &n );
    sscanf( argv[ 3 ], "%d", &k );

    test_bl_dgemm( m, n, k );

    return 0;
}}
'''.format( pack_type ) )





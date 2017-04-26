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

from common_mix import is_one, is_negone, is_nonzero, write_line, write_break, data_access, transpose, printmat, writeCoeffs, phantomMatMul, parse_coeff, read_coeffs, writePartition, writeEquation, getBlockName, getName, generateCoeffs, exp_dim_mix, num_nonzero

def create_packm_functions(myfile, coeffs):
    ''' Generate all of the custom add functions.

    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    '''
    def all_adds(coeffs, name):
        for i, coeff_set in enumerate(coeffs):
            if len(coeff_set) > 0:
                write_packm_func(myfile, coeff_set, i, name)
                write_break(myfile)

    # S matrices formed from A subblocks
    all_adds(transpose(coeffs[0]), 'A')

    # T matrices formed from B subblocks
    all_adds(transpose(coeffs[1]), 'B')

    # Output C formed from multiplications
    ##all_adds( coeffs[2], 'M' )
    #all_adds(transpose(coeffs[2]), 'C' )


def create_macro_functions( myfile, coeffs ):
    add = 'inline void bl_macro_kernel_stra_ab( int m, int n, int k, double *packA, double *packB, double *C, int ldC ) {' 
    write_line(myfile, 0, add)

    write_line( myfile, 1, 'int i, j;' )
    write_line( myfile, 1, 'aux_t aux;' )
    write_line( myfile, 1, 'aux.b_next = packB;' )

    write_line( myfile, 1, 'for ( j = 0; j < n; j += DGEMM_NR ) {' )
    write_line( myfile, 1, '    aux.n  = min( n - j, DGEMM_NR );' )
    write_line( myfile, 1, '    for ( i = 0; i < m; i += DGEMM_MR ) {' )
    write_line( myfile, 1, '        aux.m = min( m - i, DGEMM_MR );' )
    write_line( myfile, 1, '        if ( i + DGEMM_MR >= m ) {' )
    write_line( myfile, 1, '            aux.b_next += DGEMM_NR * k;' )
    write_line( myfile, 1, '        }' )

    add = '( *bl_micro_kernel )( k, &packA[ i * k ], &packB[ j * k ], &C[ j * ldC + i ], (unsigned long long) ldC, &aux );'
    #add = '( bl_dgemm_asm_8x4_beta0 )( k, &packA[ i * k ], &packB[ j * k ], &C[ j * ldC + i ], (unsigned long long) ldC, &aux );'
    write_line(myfile, 3, add)

    write_line(myfile, 2, '}')
    write_line(myfile, 1, '}')
    write_line(myfile, 0, '}')  # end of function


def write_packm_func( myfile, coeffs, index, mat_name ):
    ''' Write the add function for a set of coefficients.  This is a custom add
    function used for a single multiply in a single fast algorithm.

    coeffs is the set of coefficients used for the add
    '''
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    add = 'inline void pack%s_add_stra_ab%d( int m, int n, ' % (mat_name, index)
    add += ', '.join(['double *%s%d' % ( mat_name, i ) for i in range(nnz)])
    add += ', int ld%s, double *pack%s ' % (mat_name, mat_name)
    add += ') {'
    write_line(myfile, 0, add)

    write_line( myfile, 1, 'int i, j;' )


    add = 'double '
    add += ', '.join(['*%s%d_pntr' % ( mat_name, i ) for i in range(nnz)])
    add += ', *pack%s_pntr;' % mat_name
    write_line( myfile, 1, add )

    if ( mat_name == 'A' ):
        ldp  = 'DGEMM_MR'
        incp = '1'
        ldm  = 'ld%s' % mat_name
        incm = '1'
    elif ( mat_name == 'B' ):
        ldp  = 'DGEMM_NR'
        incp = '1'
        ldm  = '1'
        incm = 'ld%s' % mat_name
    else:
        print "Wrong mat_name!"
    #ldp = 'DGEMM_MR' if (mat_name == 'A') else 'DGEMM_NR'

    write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {' )
    write_line( myfile, 2, 'pack%s_pntr = &pack%s[ %s * j ];' % (mat_name, mat_name, ldp) )
    if ldm == '1':
        add = ''.join(['%s%d_pntr = &%s%d[ j ]; ' % ( mat_name, i, mat_name, i ) for i in range(nnz)])
    else:
        add = ''.join(['%s%d_pntr = &%s%d[ %s * j ]; ' % ( mat_name, i, mat_name, i, ldm ) for i in range(nnz)])
    write_line( myfile, 2, add )

    write_line( myfile, 2, 'for ( i = 0; i < %s; ++i ) {' % ldp )

    add = 'pack%s_pntr[ i ]' % mat_name + ' ='
    for j, coeff in enumerate(nonzero_coeffs):
        ind = j
        add += arith_expression_pntr(coeff, mat_name, ind, incm )
    
    add += ';'
    write_line(myfile, 3, add)

    write_line(myfile, 2, '}')
    write_line(myfile, 1, '}')

    write_line(myfile, 0, '}')  # end of function

def arith_expression_pntr(coeff, mat_name, ind, incm):
    ''' Return the arithmetic expression needed for multiplying coeff by value
    in a string of expressions.

    coeff is the coefficient
    value is a string representing the value to be multiplied by coeff
    place is the place in the arithmetic expression
    '''
    if incm == '1':
        value = '%s%d_pntr[ i ]'% ( mat_name, ind )
    else:
        value = '%s%d_pntr[ i * %s ]'% ( mat_name, ind, incm )
    if is_one(coeff):
         expr = ' %s' % value
    elif is_negone(coeff):
        expr = ' - %s' % value
    else:
        #print "coeff is not +-1!"
        expr = ' (double)(%s) * %s' % (coeff, value)

    if ind != 0 and not is_negone(coeff):
        return ' +' + expr
    return expr


def create_straprim_caller( myfile, coeffs, dims_mix, num_multiplies ):
    ''' Generate all of the function callers.
    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    dims is a 3-tuple (m, k, n) of the dimensions of the problem
    '''
    for i in xrange(len(coeffs[0][0])):
        a_coeffs = [c[i] for c in coeffs[0]]
        b_coeffs = [c[i] for c in coeffs[1]]
        c_coeffs = [c[i] for c in coeffs[2]]
        write_straprim_caller(myfile, i, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies )


def create_straprim_ab_functions( myfile, coeffs, dims_mix ):
    ''' Generate all of the AB fmm primitive functions.
    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    dims is a 3-tuple (m, k, n) of the dimensions of the problem
    '''
    for i in xrange(len(coeffs[0][0])):
        a_coeffs = [c[i] for c in coeffs[0]]
        b_coeffs = [c[i] for c in coeffs[1]]
        c_coeffs = [c[i] for c in coeffs[2]]
        write_straprim_ab_function( myfile, i, a_coeffs, b_coeffs, c_coeffs, dims_mix )

def write_straprim_caller(myfile, index, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies ):
    comment = '// M%d = (' % (index)
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 0, i, dims_mix ) \
                               for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    comment += ') * ('
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 1, i, dims_mix ) \
                               for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    comment += '); '
    comment += '; '.join([' %s += %s * M%d' % ( getBlockName( 2, i, dims_mix ), c, index ) for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    comment += ';'
    write_line(myfile, 1, comment)

    add = 'bl_dgemm_straprim_ab%d( ms, ns, ks, ' % index

    add += ', '.join(['%s' % getBlockName( 0, i, dims_mix ) \
                      for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    add += ', lda, '
    add += ', '.join(['%s' % getBlockName( 1, i, dims_mix ) \
                      for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    add += ', ldb, '
    add += ', '.join(['%s' % getBlockName( 2, i, dims_mix ) \
                      for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    add += ', ldc, packA, packB, bl_ic_nt );'
    write_line( myfile, 1, add )

def write_straprim_ab_function( myfile, index, a_coeffs, b_coeffs, c_coeffs, dims_mix ):
    comment = '// M%d = (' % (index)
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 0, i, dims_mix ) \
                               for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    comment += ') * ('
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 1, i, dims_mix ) \
                               for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    comment += '); '
    comment += '; '.join([' %s += %s * M%d' % ( getBlockName( 2, i, dims_mix ), c, index ) for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    comment += ';'
    write_line(myfile, 0, comment)

    add = 'void bl_dgemm_straprim_ab%d( int m, int n, int k, ' % index

    add += ', '.join(['double* %s%d' % ( 'a', i ) for i in range( num_nonzero(a_coeffs) )])
    add += ', int lda, '
    add += ', '.join(['double* %s%d' % ( 'b', i ) for i in range( num_nonzero(b_coeffs) )])
    add += ', int ldb, '
    add += ', '.join(['double* %s%d' % ( 'c', i ) for i in range( num_nonzero(c_coeffs) )])
    add += ', int ldc, double *packA, double *packB, int bl_ic_nt ) {'

    write_line( myfile, 0, add )
    write_line( myfile, 1, 'int i, j, p, ic, ib, jc, jb, pc, pb;' )
    write_line( myfile, 1, 'int ldM = m, nM = n;' )
    write_line( myfile, 1, 'double *M = bl_malloc_aligned( ldM, nM, sizeof(double) );' )
    #####################
    write_line( myfile, 1, 'memset( M, 0, sizeof(double) * ldM * nM );' )
    write_line( myfile, 1, 'for ( jc = 0; jc < n; jc += DGEMM_NC ) {' )
    write_line( myfile, 2, 'jb = min( n - jc, DGEMM_NC );' )
    write_line( myfile, 2, 'for ( pc = 0; pc < k; pc += DGEMM_KC ) {' )
    write_line( myfile, 3, 'pb = min( k - pc, DGEMM_KC );' )
    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 3, '#pragma omp parallel for num_threads( bl_ic_nt ) private( j )' )
    write_line( myfile, 0, '#endif')
    write_line( myfile, 3, 'for ( j = 0; j < jb; j += DGEMM_NR ) {' )

    add = 'packB_add_stra_ab%d( min( jb - j, DGEMM_NR ), pb, ' % index
    add += ', '.join(['&%s%d[ pc + (jc+j)*ldb ]' % ( 'b', i ) for i in range( num_nonzero(b_coeffs) )])
    add += ', ldb, &packB[ j * pb ] );'
    write_line( myfile, 4, add )

    write_line( myfile, 3, '}' )
    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 3, '#pragma omp parallel num_threads( bl_ic_nt ) private( ic, ib, i )' )
    write_line( myfile, 0, '#endif')
    write_line( myfile, 3, '{' )
    write_line( myfile, 4, 'int tid = omp_get_thread_num();' )
    write_line( myfile, 4, 'int my_start;' )
    write_line( myfile, 4, 'int my_end;' )
    write_line( myfile, 4, 'bl_get_range( m, DGEMM_MR, &my_start, &my_end );' )
    write_line( myfile, 4, 'for ( ic = my_start; ic < my_end; ic += DGEMM_MC ) {' )
    write_line( myfile, 5, 'ib = min( my_end - ic, DGEMM_MC );' )
    write_line( myfile, 5, 'for ( i = 0; i < ib; i += DGEMM_MR ) {' )

    add = 'packA_add_stra_ab%d( min( ib - i, DGEMM_MR ), pb, ' % index
    add += ', '.join(['&%s%d[ pc*lda + (ic+i) ]' % ( 'a', i ) for i in range( num_nonzero(a_coeffs) )])
    add += ', lda, &packA[ tid * DGEMM_MC * pb + i * pb ] );'
    write_line( myfile, 6, add )

    write_line( myfile, 5, '}' )

    add = 'bl_macro_kernel_stra_ab( ib, jb, pb, packA + tid * DGEMM_MC * pb, packB, &M[ jc * ldM + ic ], ldM );'
    write_line( myfile, 5, add )

    write_line( myfile, 4, '}' )
    write_line( myfile, 3, '}' )
    write_line( myfile, 2, '}' )
    write_line( myfile, 1, '}' )

    ############################
    add = 'M_Add%d( m, n, ' % ( index )

    if len( c_coeffs ) > 0:
        nonzero_coeffs = [coeff for coeff in c_coeffs if is_nonzero(coeff)]
        nnz = len( nonzero_coeffs )
        add += ', '.join( ['%s%d' % ( 'c', i ) for i in range( nnz )] )

    add += ', ldc, M, ldM, bl_ic_nt );'

    write_line( myfile, 1, add )

    write_line( myfile, 1, 'free( M );' )

    write_line( myfile, 0, '}' )
    write_break( myfile )

def write_ab_strassen_header( myfile ):
    write_line( myfile, 1, 'double *packA, *packB;' );
    write_line( myfile, 1, 'char *str;' );
    write_line( myfile, 1, 'int  bl_ic_nt;' )
    write_break( myfile )
    write_line( myfile, 1, '// Early return if possible' );
    write_line( myfile, 1, 'if ( m == 0 || n == 0 || k == 0 ) {' );
    write_line( myfile, 1, '    printf( "bl_dgemm_strassen_ab(): early return\\n" );' );
    write_line( myfile, 1, '    return;' );
    write_line( myfile, 1, '}' );
    write_break( myfile )
    write_line( myfile, 1, '// sequential is the default situation' );
    write_line( myfile, 1, 'bl_ic_nt = 1;' );
    write_line( myfile, 1, '// check the environment variable' );
    write_line( myfile, 1, 'str = getenv( "BLISLAB_IC_NT" );' );
    write_line( myfile, 1, 'if ( str != NULL ) {' );
    write_line( myfile, 1, '    bl_ic_nt = (int)strtol( str, NULL, 10 );' );
    write_line( myfile, 1, '}' );
    write_break( myfile )
    write_line( myfile, 1, '// Allocate packing buffers' );
    write_line( myfile, 1, 'packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );' );
    write_line( myfile, 1, 'packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );' );

    write_break( myfile )

def create_M_add_functions(myfile, coeffs):
    def all_adds(coeffs, name):
        for i, coeff_set in enumerate(coeffs):
            if len(coeff_set) > 0:
                write_M_add_func(myfile, coeff_set, i, name)
                write_break(myfile)

    # Output C formed from multiplications
    #all_adds( coeffs[2], 'M' )
    all_adds(transpose(coeffs[2]), 'M' )

def write_M_add_func( myfile, coeffs, index, mat_name ):
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    add = 'void %s_Add%d( int m, int n, ' % (mat_name, index)
    add += ', '.join(['double* %s%d' % ( mat_name, i ) for i in range(nnz)])
    add += ', int ld%s, double* R, int ldR, int bl_ic_nt ' % (mat_name)
    # Handle the C := alpha A * B + beta C
    is_output = (mat_name == 'M')
    #is_output = False 
    #if is_output:
    #    add += ', double beta'
    add += ') {'
    write_line(myfile, 0, add)

    # Handle the C := alpha A * B + beta C

    write_line( myfile, 1, 'int i, j;' )
    #write_line( myfile, 1, '#pragma omp parallel for schedule( dynamic )' )
    write_line( myfile, 0, '#ifdef _PARALLEL_')
    write_line( myfile, 1, '#pragma omp parallel for num_threads( bl_ic_nt )' )
    write_line( myfile, 0, '#endif')
    write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {')
    write_line( myfile, 2, 'for ( i = 0; i < m; ++i ) {')
    for j, coeff in enumerate(nonzero_coeffs):
        ind = j
        add = data_access( mat_name, str(ind) )  + ' += '
        add += arith_expression( coeff, 'R', '' )
        add += ';'
        write_line(myfile, 3, add)
    write_line(myfile, 2, '}')
    write_line(myfile, 1, '}')
    write_line(myfile, 0, '}')  # end of function


def arith_expression(coeff, mat_name, ind):
    ''' Return the arithmetic expression needed for multiplying coeff by value
    in a string of expressions.

    coeff is the coefficient
    value is a string representing the value to be multiplied by coeff
    place is the place in the arithmetic expression
    '''
    value = data_access( mat_name, str( ind ) )
    if is_one(coeff):
         expr = ' %s' % value
    elif is_negone(coeff):
        expr = ' -%s' % value
    else:
        #print "coeff is not +-1!"
        expr = ' (double)(%s) * %s' % (coeff, value)

    if ind != 0 and not is_negone(coeff):
        return ' +' + expr
    return expr


def gen_ab_fmm( coeff_filename_mix, dims_mix, level_mix, outfilename ):

    coeffs_mix = []
    idx = 0
    for coeff_file in coeff_filename_mix:
        coeffs = read_coeffs( coeff_file )
        level = level_mix[idx]
        for level_id in range( level ):
            coeffs_mix.append( coeffs )
        idx += 1

    dims_level_mix = []
    idx = 0
    for dims in dims_mix:
        level = level_mix[idx]
        for level_id in range( level ):
            dims_level_mix.append( dims )
        idx += 1

    #coeffs2= [ transpose( U2 ), transpose( V2 ), transpose( W2 ) ]

    with open( outfilename, 'w' ) as myfile:
        write_line( myfile, 0, '#include "bl_dgemm_kernel.h"' )
        write_line( myfile, 0, '#include "bl_dgemm.h"' )
        write_break( myfile )

        cur_coeffs = generateCoeffs( coeffs_mix )
        #writeCoeffs( cur_coeffs )
        #writeEquation( cur_coeffs, dims_level_mix )

        num_multiplies = len(cur_coeffs[0][0])

        create_M_add_functions( myfile, cur_coeffs )

        create_packm_functions( myfile, cur_coeffs )

        create_macro_functions( myfile, cur_coeffs )

        create_straprim_ab_functions( myfile, cur_coeffs, dims_level_mix )

        write_line( myfile, 0, 'void bl_dgemm_strassen_ab( int m, int n, int k, double *XA, int lda, double *XB, int ldb, double *XC, int ldc )' )
        write_line( myfile, 0, '{' )

        write_ab_strassen_header( myfile )

        writePartition( myfile, dims_level_mix )

        write_break( myfile )

        create_straprim_caller( myfile, cur_coeffs, dims_level_mix, num_multiplies )

        write_break( myfile )
        level_dim = exp_dim_mix( dims_level_mix )
        write_line( myfile, 1, 'bl_dynamic_peeling( m, n, k, XA, lda, XB, ldb, XC, ldc, %d * DGEMM_MR, %d, %d * DGEMM_NR );' % ( level_dim[0], level_dim[1], level_dim[2] ) )

        write_break( myfile )
        write_line( myfile, 1, 'free( packA );' )
        write_line( myfile, 1, 'free( packB );' )

        write_line( myfile, 0, '}' )

def main():
    try:
        coeff_file_mix = [ ]
        dims_mix = [ ]
        level_mix = [ ]
        num_file    = int( sys.argv[1] )
        for file_id in range( num_file ):
            coeff_file = sys.argv[ 2 + file_id * 3 ]
            dims  = tuple([int(d) for d in sys.argv[ 3 + file_id * 3 ].split(',')])
            level = int( sys.argv[ 4 + file_id * 3 ] )
            coeff_file_mix.append( coeff_file )
            dims_mix.append( dims )
            level_mix.append( level )

        outfile = 'a.c'

        if len(sys.argv) > 2 + 3 * num_file:
            outfile = sys.argv[ 2 + 3 * num_file ]

    except:
        raise Exception('USAGE: python ab_gen_mix.py 2 coeff_file1 m1,n1,p1 L1 coeff_file2 m2,n2,p2 L2 out_file')

    gen_ab_fmm( coeff_file_mix, dims_mix, level_mix, outfile )



if __name__ == '__main__':
    main()


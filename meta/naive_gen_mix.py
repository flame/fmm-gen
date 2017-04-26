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

from common_mix import is_one, is_negone, is_nonzero, write_line, write_break, data_access, transpose, printmat, writeCoeffs, phantomMatMul, parse_coeff, read_coeffs, writePartition, writeEquation, getBlockName, getName, generateCoeffs, exp_dim_mix


def create_add_functions(myfile, coeffs):
    ''' Generate all of the custom add functions.

    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    '''
    def all_adds(coeffs, name):
        for i, coeff_set in enumerate(coeffs):
            if len(coeff_set) > 0:
                write_add_func(myfile, coeff_set, i, name)
                write_break(myfile)

    # S matrices formed from A subblocks
    all_adds(transpose(coeffs[0]), 'S')

    # T matrices formed from B subblocks
    all_adds(transpose(coeffs[1]), 'T')

    # Output C formed from multiplications
    #all_adds( coeffs[2], 'M' )
    all_adds(transpose(coeffs[2]), 'M' )

def write_add_func( myfile, coeffs, index, mat_name ):
    ''' Write the add function for a set of coefficients.  This is a custom add
    function used for a single multiply in a single fast algorithm.

    coeffs is the set of coefficients used for the add
    '''
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
    if is_output:
        #write_line( myfile, 1, 'int i, j;' )
        #write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {')
        #write_line( myfile, 2, 'for ( i = 0; i < m; ++i ) {')
        #add = data_access('R') + ' ='
        #for j, coeff in enumerate(nonzero_coeffs):
        #    ind = j
        #    add += arith_expression(coeff, mat_name, ind )
        #add += ' + %s;' % (data_access('R'))
        #write_line(myfile, 3, add)
        #write_line(myfile, 2, '}')
        #write_line(myfile, 1, '}')

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
            add += arith_expression(coeff, 'R', '' )
            add += ';'
            write_line(myfile, 3, add)
        write_line(myfile, 2, '}')
        write_line(myfile, 1, '}')

        #write_line( myfile, 1, 'int i, j;' )
        #for j, coeff in enumerate(nonzero_coeffs):
        #    write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {')
        #    write_line( myfile, 2, 'for ( i = 0; i < m; ++i ) {')
        #    ind = j
        #    add = data_access( mat_name, str(ind) )  + ' += '
        #    add += arith_expression(coeff, 'R', '' )
        #    add += ';'
        #    write_line(myfile, 3, add)
        #    write_line(myfile, 2, '}')
        #    write_line(myfile, 1, '}')
    else:
        write_line( myfile, 1, 'int i, j;' )
        write_line( myfile, 0, '#ifdef _PARALLEL_')
        write_line( myfile, 1, '#pragma omp parallel for num_threads( bl_ic_nt )' )
        write_line( myfile, 0, '#endif')
        write_line( myfile, 1, 'for ( j = 0; j < n; ++j ) {' )
        write_line( myfile, 2, 'for ( i = 0; i < m; ++i ) {' )
        add = data_access('R') + ' ='
        for j, coeff in enumerate(nonzero_coeffs):
            ind = j
            add += arith_expression(coeff, mat_name, ind )
    
        add += ';'
        write_line(myfile, 3, add)
        write_line(myfile, 2, '}')
        write_line(myfile, 1, '}')

    write_line(myfile, 0, '}')  # end of function

def arith_expression(coeff, mat_name, ind):
    ''' Return the arithmetic expression needed for multiplying coeff by value
    in a string of expressions.
    '''
    value = data_access( mat_name, str( ind ) )
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


def write_output_add( myfile, index, coeffs, dims, rank ):
    add = 'M_Add%d( ' % (index)
    add += 'ms, ns, '
    for i, coeff in enumerate(coeffs):
        if is_nonzero(coeff):
            suffix = i
            #if suffix > rank:
            #    suffix = '_X%d' % (suffix - rank)
            add += 'M%s, ' % suffix
    add += 'ldM, '
    #output_mat = getBlockName( 2, index, dims, level )
    output_mat = getBlockName( 2, index, dims )
    add += '%s, ldc, bl_ic_nt );' % output_mat
    write_line(myfile, 1, add)

def create_output( myfile, coeffs, dims  ):
    num_multiplies = len(coeffs[0][0])
    for i, row in enumerate(coeffs[2]):
        write_output_add(myfile, i, row, dims,
                         num_multiplies  )


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
        write_multiply_caller(myfile, i, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies )


def create_straprim_naive_functions( myfile, coeffs, dims_mix, num_multiplies ):
    ''' Generate all of the Naive fmm primitive functions.
    myfile is the file to which we are writing
    coeffs is the set of all coefficients
    dims is a 3-tuple (m, k, n) of the dimensions of the problem
    '''
    for i in xrange(len(coeffs[0][0])):
        a_coeffs = [c[i] for c in coeffs[0]]
        b_coeffs = [c[i] for c in coeffs[1]]
        c_coeffs = [c[i] for c in coeffs[2]]
        write_straprim_naive_function( myfile, i, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies )

def is_nonone(x):
    return not is_one(x)

def num_nonone(arr):
    ''' Returns number of non-one entries in the array arr. '''
    return len(filter(is_nonone, arr))

def need_tmp_mat(coeffs):
    return num_nonone(coeffs) > 1

def instantiate_tmp(myfile, tmp_name, mult_index):
    inst = 'double* %s%d = bl_malloc_aligned( ld%s, n%s, sizeof(double) );' % (tmp_name, mult_index, tmp_name, tmp_name)
    write_line(myfile, 1, inst)

def write_multiply_caller(myfile, index, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies ):
    comment = '// M%d = (' % (index)
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 0, i, dims_mix ) \
                               for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    comment += ') * ('
    comment += ' + '.join([str(c) + ' * %s' % getBlockName( 1, i, dims_mix  ) \
                               for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    comment += '); '
    comment += '; '.join([' %s += %s * M%d' % ( getBlockName( 2, i, dims_mix ), c, index ) for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    comment += ';'
    write_line(myfile, 1, comment)


    add = 'bl_dgemm_straprim_naive%d( ms, ns, ks, ' % index

    add += ', '.join(['%s' % getBlockName( 0, i, dims_mix ) \
                      for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    add += ', lda, '
    add += ', '.join(['%s' % getBlockName( 1, i, dims_mix ) \
                      for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    add += ', ldb, '
    add += ', '.join(['%s' % getBlockName( 2, i, dims_mix ) \
                      for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    add += ', ldc, bl_ic_nt );'
    write_line( myfile, 1, add )

def write_straprim_naive_function(myfile, index, a_coeffs, b_coeffs, c_coeffs, dims_mix, num_multiplies ):
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

    add = 'void bl_dgemm_straprim_naive%d( int ms, int ns, int ks, ' % index

    add += ', '.join(['double* %s' % getBlockName( 0, i, dims_mix ) \
                      for i, c in enumerate(a_coeffs) if is_nonzero(c)])
    add += ', int lda, '
    add += ', '.join(['double* %s' % getBlockName( 1, i, dims_mix ) \
                      for i, c in enumerate(b_coeffs) if is_nonzero(c)])
    add += ', int ldb, '
    add += ', '.join(['double* %s' % getBlockName( 2, i, dims_mix ) \
                      for i, c in enumerate(c_coeffs) if is_nonzero(c)])
    add += ', int ldc, int bl_ic_nt ) {'

    #add += ', '.join(['double* %s%d' % ( 'a', i ) for i in range( num_nonzero(a_coeffs) )])
    #add += ', lda, '
    #add += ', '.join(['double* %s%d' % ( 'b', i ) for i in range( num_nonzero(b_coeffs) )])
    #add += ', ldb, '
    #add += ', '.join(['double* %s%d' % ( 'c', i ) for i in range( num_nonzero(c_coeffs) )])
    #add += ', ldc ) {'

    write_line( myfile, 0, add )
    write_line( myfile, 1, 'int ldS = ms, nS = ks, ldT = ks, nT = ns, ldM = ms, nM = ns;' )

    def para_ld( coeff_index  ):
        if( coeff_index == 0 ):
            mm = 'ms'
            nn = 'ks'
        elif( coeff_index == 1 ):
            mm = 'ks'
            nn = 'ns' 
        elif( coeff_index == 2 ):
            mm = 'ms' 
            nn = 'ns'
        else:
            print "Wrong coeff_index\n"
        return str(mm) + ', ' + str(nn) + ', '

    def addition_str(coeffs, coeff_index, mat_name, tmp_name, dims_mix ):
        tmp_mat = '%s%d' % (tmp_name, index)
        add = '%s_Add%d( %s' % ( tmp_name, index, para_ld( coeff_index ) )
        for i, coeff in enumerate(coeffs):
            if is_nonzero(coeff):
                add += getBlockName( coeff_index, i, dims_mix ) + ', '
        add += 'ld%s, ' % ( mat_name )
        add += tmp_mat + ', ld%s, bl_ic_nt );' % tmp_name
        return add

    # Write the adds to temps if necessary
    if need_tmp_mat(a_coeffs):
        instantiate_tmp(myfile, 'S', index)
        write_line(myfile, 1, addition_str(a_coeffs, 0, 'a', 'S', dims_mix ))

    if need_tmp_mat(b_coeffs):
        instantiate_tmp(myfile, 'T', index)
        write_line(myfile, 1, addition_str(b_coeffs, 1, 'b', 'T', dims_mix ))

    inst = 'double* M%d = bl_malloc_aligned( ldM, nM, sizeof(double) );' % ( index )
    write_line( myfile, 1, inst )
    write_line( myfile, 1, 'memset( M%d, 0, sizeof(double) * ldM * nM );' % ( index ) )

    res_mat = 'M%d' % (index)

    ## Handle the case where there is one non-zero coefficient and it is
    ## not equal to one.  We need to propagate the multiplier information.
    #a_nonzero_coeffs = filter(is_nonzero, a_coeffs)
    #b_nonzero_coeffs = filter(is_nonzero, b_coeffs)
    #if len(a_nonzero_coeffs) == 1 and a_nonzero_coeffs[0] != 1:
    #    write_line(myfile, 1, '%s.UpdateMultiplier(Scalar(%s));' % (res_mat,
    #                                                                a_nonzero_coeffs[0]))
    #if len(b_nonzero_coeffs) == 1 and b_nonzero_coeffs[0] != 1:
    #    write_line(myfile, 1, '%s.UpdateMultiplier(Scalar(%s));' % (res_mat,
    #                                                                b_nonzero_coeffs[0]))

    def subblock_name( coeffs, coeff_index, mat_name, tmp_name, dims_mix ):
        if need_tmp_mat(coeffs):
            return '%s%d' % (tmp_name, index)
        else:
            loc = [i for i, c in enumerate(coeffs) if is_nonzero(c)]
            return getBlockName( coeff_index, loc[0], dims_mix )

    def subblock_ld( coeffs, mat_name, tmp_name ):
        if need_tmp_mat(coeffs):
            return '%s' % (tmp_name)
        else:
            return mat_name

    # Finally, write the actual call to matrix multiply.
    write_line(myfile, 1,
               'bl_dgemm( ms, ns, ks, %s, ld%s, %s, ld%s, %s, ldM );' % (
            subblock_name(a_coeffs, 0, 'a', 'S', dims_mix ),
            subblock_ld(a_coeffs, 'a', 'S' ),
            subblock_name(b_coeffs, 1, 'b', 'T', dims_mix ),
            subblock_ld(b_coeffs, 'b', 'T' ),
            res_mat
               ))

    write_line( myfile, 1, addition_str(c_coeffs, 2, 'c', 'M', dims_mix ))

    # If we are not in parallel mode, de-allocate the temporary matrices
    if need_tmp_mat(a_coeffs):
        write_line(myfile, 1, 'free( S%d );' % (index))
    if need_tmp_mat(b_coeffs):
        write_line(myfile, 1, 'free( T%d );' % (index))

    write_line(myfile, 1, 'free( M%d );' % (index))
    write_line( myfile, 0, '}' )
    write_break( myfile )

def write_naive_strassen_header( myfile ):
    write_line( myfile, 1, 'char *str;' );
    write_line( myfile, 1, 'int  bl_ic_nt;' )
    write_break( myfile )
    write_line( myfile, 1, '// Early return if possible' );
    write_line( myfile, 1, 'if ( m == 0 || n == 0 || k == 0 ) {' );
    write_line( myfile, 1, '    printf( "bl_dgemm_strassen_abc(): early return\\n" );' );
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


def gen_naive_fmm( coeff_filename_mix, dims_mix, level_mix, outfile ):

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

    with open(outfile, 'w') as myfile:
        write_line( myfile, 0, '#include "bl_dgemm.h"' )
        write_break( myfile )

        cur_coeffs = generateCoeffs( coeffs_mix )

        num_multiplies = len(cur_coeffs[0][0])

        create_add_functions( myfile, cur_coeffs )
        create_straprim_naive_functions( myfile, cur_coeffs, dims_level_mix, num_multiplies )

        write_line( myfile, 0, 'void bl_dgemm_strassen_naive( int m, int n, int k, double *XA, int lda, double *XB, int ldb, double *XC, int ldc )' )
        write_line( myfile, 0, '{' )

        write_naive_strassen_header( myfile )

        writePartition( myfile, dims_level_mix )

        write_break( myfile )

        create_straprim_caller( myfile, cur_coeffs, dims_level_mix, num_multiplies )

        write_break( myfile )
        level_dim = exp_dim_mix( dims_level_mix )
        write_line( myfile, 1, 'bl_dynamic_peeling( m, n, k, XA, lda, XB, ldb, XC, ldc, %d * DGEMM_MR, %d, %d * DGEMM_NR );' % ( level_dim[0], level_dim[1], level_dim[2] ) )


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

    gen_naive_fmm( coeff_file_mix, dims_mix, level_mix, outfile )

if __name__ == '__main__':
    main()



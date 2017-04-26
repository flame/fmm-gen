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

from common_util import is_one, is_negone, is_nonzero, contain_nontrivial, write_line, write_break, data_access, transpose, parse_coeff, read_coeffs, num_nonzero

def printmat( X ):
    for jj in range( len(X[0]) ):
        mystr = ""
        for ii in range( len(X) ):
            #mystr += '{:04.2f}'.format( float(X[ii][jj]) ) + " "
            mystr += '%5.2f' % ( float(X[ii][jj]) ) + " "
        print mystr

def writeCoeffs( coeffs ):
    U = transpose( coeffs[ 0 ] )
    V = transpose( coeffs[ 1 ] )
    W = transpose( coeffs[ 2 ] )
    print ( "U:" )
    printmat( U )
    print ( "V:" )
    printmat( V )
    print ( "W:" )
    printmat( W )
    print ""

def genSubmatID( submat_id_queue, split_num ):
    res_submat_id_queue = []
    for elem in submat_id_queue:
        for idx in range( split_num ):
            res_submat_id_queue.append( elem + '_' + str(idx) )
    return res_submat_id_queue

# composition operation? Kronecker Product
def phantomMatMul( A, B ):
    m_A = len( A[0] )
    n_A = len( A    )
    m_B = len( B[0] )
    n_B = len( B    )

    m_C = m_A * m_B
    n_C = n_A * n_B

    C = [ [0 for x in range( m_C )] for y in range( n_C ) ]
    #print C

    for colid_A in range( n_A ):
        vec_A = A[ colid_A ]
        for rowid_A in range( m_A ):
            elem_A = vec_A[ rowid_A ]
            if ( elem_A != 0 ):
                for colid_B in range( n_B ):
                    vec_B = B[ colid_B ]
                    for rowid_B in range( m_B ):
                        elem_B = vec_B[ rowid_B ]
                        if ( elem_B != 0 ):
                            rowid_C = rowid_A * m_B + rowid_B
                            colid_C = colid_A * n_B + colid_B
                            elem_C = str( float(elem_A) * float(elem_B) )
                            C[ colid_C ][ rowid_C ] = elem_C

    return C

def writeSubmat( myfile, mat_name, dim1, dim2, split1, split2, src_mat_id ): 
    decl = "double *"
    sep_symbol = ""
    for ii in range( split1 ):
        for jj in range( split2 ):
            decl+=sep_symbol+mat_name+str(src_mat_id)+'_'+str(ii * split2 + jj)
            sep_symbol=", *"
    decl+=";"
    write_line( myfile, 1, decl )

    for ii in range( split1 ):
        for jj in range( split2 ):
            write_line( myfile, 1, "bl_acquire_mpart( {0}, {1}, {2}, ld{3}, {4}, {5}, {6}, {7}, &{2}{8} );".format( dim1, dim2, mat_name+str(src_mat_id), mat_name, split1, split2, ii, jj, '_'+str(ii * split2 + jj) ) )

def exp_dim( dims, level ):
    res = [ 1, 1, 1 ]
    for i in range( level ):
        res[ 0 ] = res[ 0 ] * dims[ 0 ]
        res[ 1 ] = res[ 1 ] * dims[ 1 ]
        res[ 2 ] = res[ 2 ] * dims[ 2 ]
    return tuple( res )

def exp_dim_mix( dims_mix ):
    res = [ 1, 1, 1 ]
    for dims in dims_mix:
        res[ 0 ] = res[ 0 ] * dims[ 0 ]
        res[ 1 ] = res[ 1 ] * dims[ 1 ]
        res[ 2 ] = res[ 2 ] * dims[ 2 ]
    return tuple( res )

def writePartition( myfile, dims_mix ):

    write_line( myfile, 1, "int ms, ks, ns;" )
    write_line( myfile, 1, "int md, kd, nd;" )
    write_line( myfile, 1, "int mr, kr, nr;" )
    write_line( myfile, 1, "double *a = XA, *b= XB, *c = XC;" )
    write_break( myfile )

    level_dim = exp_dim_mix( dims_mix )

    write_line( myfile, 1, "mr = m %% ( %d * DGEMM_MR ), kr = k %% ( %d ), nr = n %% ( %d * DGEMM_NR );" % ( level_dim[0], level_dim[1], level_dim[2] ) )
    write_line( myfile, 1, "md = m - mr, kd = k - kr, nd = n - nr;" )

    write_break( myfile )

    triple_combinations = [
        ( "a", "ms", "ks", 0, 1 ),
        ( "b", "ks", "ns", 1, 2 ),
        ( "c", "ms", "ns", 0, 2 )
    ]

    level = len( dims_mix )

    for ( mat_name, dim1, dim2, split1, split2 ) in triple_combinations:
        write_line( myfile, 1, "ms=md, ks=kd, ns=nd;" )
        submat_id_queue = [""]

        for level_id in range( level ):

            cur_dims = dims_mix[ level_id ]

            for src_mat_id in submat_id_queue:
                writeSubmat( myfile, mat_name, dim1, dim2, cur_dims[split1], cur_dims[split2], src_mat_id )

            #Generate next level myqueue
            submat_id_queue = genSubmatID( submat_id_queue, cur_dims[split1] * cur_dims[split2] )

            # Get the current submat size
            if ( level_id != level - 1 ):
                write_line( myfile, 1, "ms=ms/{0}, ks=ks/{1}, ns=ns/{2};".format( cur_dims[0], cur_dims[1], cur_dims[2] ) )

            write_break( myfile )

        write_break( myfile )

    write_line( myfile, 1, "ms=ms/{0}, ks=ks/{1}, ns=ns/{2};".format( cur_dims[0], cur_dims[1], cur_dims[2] ) )
           
    write_break( myfile )

def getActualMatName( idx ):
    if ( idx == 0 ):
        matname = "A"
    elif( idx == 1 ):
        matname = "B"
    elif( idx == 2 ):
        matname = "C"
    else:
        print "Not supported!\n"
    return matname

def getActualBlockName( coeff_index, item_index, dims_mix ):
    my_mat_name = getActualMatName( coeff_index )

    if( coeff_index == 0 ):
        mm = 0
        nn = 1
    elif( coeff_index == 1 ):
        mm = 1
        nn = 2
    elif( coeff_index == 2 ):
        mm = 0
        nn = 2
    else:
        print "Wrong coeff_index\n"

    #my_partition_ii = item_index / nn
    #my_partition_jj = item_index % nn
    submat_index = ""
    dividend = item_index
    mm_base = 1
    nn_base = 1
    ii_index = 0
    jj_index = 0
    for dims in reversed(dims_mix):
        remainder = dividend % ( dims[mm] * dims[nn] )
        #remainder -> i, j (m_axis, n_axis)
        ii = remainder / dims[nn]
        jj = remainder % dims[nn]
        ii_index = ii * mm_base + ii_index
        jj_index = jj * nn_base + jj_index
        #submat_index = str(remainder) + submat_index
        dividend = dividend / ( dims[mm] * dims[nn] )
        mm_base = mm_base * dims[mm]
        nn_base = nn_base * dims[nn]

    return my_mat_name + "(" + str( ii_index ) + "," + str( jj_index ) + ")"

def writeEquation( coeffs, dims_mix ):
    for eq_index in range( len( coeffs[0][0] ) ):
        m_mat_name = "M"+str(eq_index)

        my_eq_str = ""
        for coeff_index in range( len(coeffs) ):
            #print "coeff_index:" + str(coeff_index)
            name_list = getName( coeff_index ) # 0: a, gamma; 1: b, delta; 2: c, alpha
            coeff_list = transpose( coeffs[ coeff_index ] )
            my_eq_coeff_list = coeff_list[ eq_index ]

            if ( coeff_index == 0 ): #A
                my_eq_str = my_eq_str + m_mat_name + "=( "
            elif ( coeff_index == 1 ): #B
                my_eq_str = my_eq_str + " )( "
            elif ( coeff_index == 2 ): #C
                my_eq_str += " );\n  "
            else:
                print "Coeff_index not supported!\n"

            nz_index = 0
            for item_index in range( len(my_eq_coeff_list) ):
                if ( is_nonzero( my_eq_coeff_list[ item_index ] ) ):

                    mat_name = getActualBlockName( coeff_index, item_index, dims_mix )
                    if ( coeff_index == 0 or coeff_index == 1 ): # A or B
                        mat_prefix = ""
                        if ( is_negone( my_eq_coeff_list[ item_index ] ) ):
                            mat_prefix = "-"
                        elif ( is_one( my_eq_coeff_list[ item_index ] ) ):
                            if ( nz_index == 0 ):
                                mat_prefix = ""
                            else:
                                mat_prefix = "+"
                        else:
                            mat_prefix = "+(" + str( my_eq_coeff_list[ item_index ] )+")"
                            #print "%d:%s" % ( item_index, my_eq_coeff_list[ item_index ] )
                            #print "entry should be either 1 or -1!"
                        my_eq_str += mat_prefix + mat_name
                    elif ( coeff_index == 2 ):
                        mat_suffix = ""
                        if ( is_negone( my_eq_coeff_list[ item_index ] ) ):
                            mat_suffix = "-="
                        elif ( is_one( my_eq_coeff_list[ item_index ] ) ):
                            mat_suffix = "+="
                        else:
                            mat_suffix = "+=(" + str( my_eq_coeff_list[ item_index ] ) + ") "
                            #print "%d:%s" % ( item_index, my_eq_coeff_list[ item_index ] )
                            #print "entry should be either 1 or -1!"
                        my_eq_str += mat_name + mat_suffix + m_mat_name + ";"
                    else:
                        print "Coeff_index not support!\n"
                    #write_line( myfile, 0, str( coeff_index ) + " " + str( item_index ) )
                    #write_line( myfile, 0, "{0}_list[{1}] = {2}; {3}_list[{1}] = {4};".format( name_list[0], str(nz_index), getBlockName( coeff_index, item_index, dims, level ), name_list[1], my_eq_coeff_list[ item_index ] ) )
                    nz_index += 1
        print my_eq_str
        #print ""

def getNNZ ( coeffs ):
    nonzero_coeffs = [coeff for coeff in coeffs if is_nonzero(coeff)]
    nnz = len( nonzero_coeffs )
    return nnz

def getBlockName( coeff_index, item_index, dims_mix, level=1 ):
    my_mat_name = (getName( coeff_index )) [ 0 ]

    if( coeff_index == 0 ):
        mm = 0
        nn = 1
    elif( coeff_index == 1 ):
        mm = 1
        nn = 2
    elif( coeff_index == 2 ):
        mm = 0
        nn = 2
    else:
        print "Wrong coeff_index\n"

    #my_partition_ii = item_index / nn
    #my_partition_jj = item_index % nn
    submat_index = ""
    dividend = item_index
    #for ii in range( level ):
    for dims in reversed(dims_mix):
        remainder = dividend % ( dims[mm] * dims[nn] )
        submat_index = '_' + str(remainder) + submat_index
        #submat_index = submat_index + str(remainder) 
        dividend = dividend / ( dims[mm] * dims[nn] )

    return my_mat_name + str( submat_index )

def getName( idx ):
    if ( idx == 0 ):
        my_list = [ 'a', 'gamma' ]
    elif( idx == 1 ):
        my_list = [ 'b', 'delta' ]
    elif( idx == 2 ):
        my_list = [ 'c', 'alpha' ]
    else:
        my_list = []
        print "Not supported!\n"
    return my_list

def generateCoeffs( coeffs_mix ):
    coeffs = coeffs_mix[ 0 ]
    UM = transpose( coeffs[ 0 ] )
    VM = transpose( coeffs[ 1 ] )
    WM = transpose( coeffs[ 2 ] )

    for ii in range( len(coeffs_mix) - 1 ):
        coeffs = coeffs_mix[ ii + 1 ]
        U = transpose( coeffs[ 0 ] )
        V = transpose( coeffs[ 1 ] )
        W = transpose( coeffs[ 2 ] )
        UM = phantomMatMul( UM, U )
        VM = phantomMatMul( VM, V )
        WM = phantomMatMul( WM, W )

    res_coeffs = [ transpose( UM ), transpose( VM ), transpose( WM ) ]

    return res_coeffs
 

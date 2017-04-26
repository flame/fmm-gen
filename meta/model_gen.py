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

import numpy as np
from common_mix import exp_dim
import math
from model_coefficient import gen_model_coefficient
from alg_dic import alg_dic

alg_set=[
'222',
'232',
'234',
'243',
'252',
'322',
'323',
'324',
'332',
'333',
'336',
'342',
'343',
'353',
'363',
'422',
'423',
'424',
'432',
'433',
'442',
'522',
'633',
]

def get_model_gflops( m, n, k, dims, level, coefficient ):

    clock_rate = 3.54
    tau = 1 / ( 8 * clock_rate )
    channels = 1
    alpha = 0.5 / channels
    mc = 96
    nc = 4096
    kc = 256

    [ M_total_mul, M_A_add,  M_B_add, M_C_add, N_A_mul, N_B_mul, N_C_mul, N_A_add, N_B_add, N_C_add ] = coefficient

    virtual_flops = 2 * m * k * n

    level_dims = exp_dim( dims, level )

    ms = m / level_dims[0]
    ks = k / level_dims[1]
    ns = n / level_dims[2]

    actual_flops = M_total_mul * 2 * ms * ks * ns + M_A_add * 2 * ms * ks + M_B_add * 2 * ks * ns + M_C_add * 2 * ms * ns

    mops_Bc = 1 * ns * ks
    mops_Ac = 1 * ms * ks * math.ceil( (float)(ns) / (float)(nc) )
    mops_Cc = 1 * math.ceil( (float)(ks) / (float)(kc) ) * ms * ns
    mops_B  = 1 * ns * ks
    mops_A  = 1 * ns * ks
    mops_C  = 1 * ms * ns

    mops = N_A_mul * mops_Ac + N_B_mul * mops_Bc + N_C_mul * 2 * mops_Cc \
         + N_A_add * mops_Bc + N_B_add * mops_B  + N_C_add * mops_C
    # Using a function to describe the prefetching efficiency: penalty -> punishment

    time_total = actual_flops * tau + mops * alpha

    effective_gflops = virtual_flops / time_total

    return effective_gflops

def generate_coefficient_mat( ):

    coeff_mat_size = 1 + 6 * len(alg_set) 

    coefficient_mat   = np.zeros( (coeff_mat_size, 10) )
    # Generate 139 * 10 Coefficient matrix for the model
    coefficient_mat[0,:] = np.asarray( [1, 0, 0, 0, 1, 1, 1, 1, 1, 1] ) # DGEMM modeling
    level_num_list = [ 1, 2 ]
    alg_id = 0
    for alg in alg_set:
        coeff_file = alg_dic[ alg ][ 0 ]
        dims = alg_dic[ alg ][ 1 ]

        for level_num in level_num_list:
            coeff_file_list = [ coeff_file ]
            level_list = [ level_num ]
            [ comp_counter, abc_counter, ab_counter, naive_counter ] = gen_model_coefficient( coeff_file_list, level_list )

            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 0 ) * len(alg_set), 0:4  ] = np.asarray( comp_counter )
            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 1 ) * len(alg_set), 0:4  ] = np.asarray( comp_counter )
            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 2 ) * len(alg_set), 0:4  ] = np.asarray( comp_counter )
            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 0 ) * len(alg_set), 4:10 ] = np.asarray( abc_counter  )
            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 1 ) * len(alg_set), 4:10 ] = np.asarray( ab_counter   )
            coefficient_mat[ 1 + alg_id * 1 + ( 3 * (level_num-1) + 2 ) * len(alg_set), 4:10 ] = np.asarray( naive_counter)
        alg_id += 1
    np.savetxt("coefficient.csv", coefficient_mat, delimiter=",")
    return coefficient_mat

step_size = 240
sample_list = range( step_size, 12000 + 1, step_size )

def generate_result_mat( coefficient_mat, test_type ):
    my_result_size = 1 + 6 * len(alg_set) 
    sample_size = len( sample_list )
    my_result_mat = np.zeros( (my_result_size, sample_size) )
    ## Iterate/Traverse the m, k, n dimension
    for var_id in range( my_result_size ):
        #dims  = alg_to_dims[ ( var_id - 1 ) % len(alg_set) ]
        if var_id == 0:
            dims = tuple( [ 1, 1, 1 ] )
        else:
            dims = alg_dic[ alg_set[ ( var_id - 1 ) % len(alg_set)  ] ] [ 1 ]
        #level = var_id - 1 < 69 ? 1 : 2
        if var_id - 1 < len(alg_set)*3:
            level = 1
        else:
            level = 2
        coefficient = coefficient_mat[ var_id, : ].tolist()
        sample_id = 0
        for my_size in sample_list:
            if test_type == 'square':
                m = my_size
                n = my_size
                k = my_size
                my_result_mat[ var_id ][ sample_id  ] = get_model_gflops( m, n, k, dims, level, coefficient ) 
            elif test_type == 'rankk':
                m = 14400
                n = 14400
                k = my_size
                my_result_mat[ var_id ][ sample_id  ] = get_model_gflops( m, n, k, dims, level, coefficient ) 
            elif test_type == 'fixk':
                m = my_size
                n = my_size
                k = 1024
                my_result_mat[ var_id ][ sample_id  ] = get_model_gflops( m, n, k, dims, level, coefficient ) 
            else:
                print 'wrong test type!'
                exit( 0 )
            sample_id += 1
            #exit( 0 )
    np.savetxt("%s_result.csv" % (test_type), my_result_mat, delimiter=",")
    return my_result_mat

my_header = 'dim,gemm,222,232,234,243,252,322,323,324,332,333,336,342,343,353,363,422,423,424,432,433,442,522,633';
prefix='./' 

def generate_csv( offset, my_filename, square_result_mat ):
    square_x = np.asarray( [ sample_list ] )
    gemm_result = square_result_mat[0:1,:]
    #filename = prefix + 'model_square_1level_abc_1core.csv'
    full_filename = prefix + my_filename
    my_result = np.concatenate( ( square_x, gemm_result, square_result_mat[1+offset:1+len(alg_set)+offset,:] ), axis = 0 )
    #my_result = my_result.T
    my_result = np.transpose( my_result )
    #np.savetxt(full_filename, my_result, delimiter=",", header=my_header, comments="")
    np.savetxt(full_filename, my_result, delimiter=",", header=my_header, comments="", fmt="%.2lf")

def main():
    coefficient_mat = generate_coefficient_mat( )
    #coefficient_mat = np.loadtxt( 'coefficient.csv', dtype='int',delimiter=',')
    #coefficient_mat = np.loadtxt( 'coefficient.csv', dtype='float',delimiter=',')

    square_result_mat = generate_result_mat( coefficient_mat, 'square' )
    #square_result_mat = np.loadtxt( 'square_result.csv', dtype='float',delimiter=',')

    alg_num = len( alg_set )
    offset_list   = [ i * alg_num for i in range(6) ]
    filename_list = [
        'model_square_1level_abc_1core.csv',
        'model_square_1level_ab_1core.csv',
        'model_square_1level_naive_1core.csv',
        'model_square_2level_abc_1core.csv',
        'model_square_2level_ab_1core.csv',
        'model_square_2level_naive_1core.csv',
    ]
    for i in range(6):
        offset = offset_list[ i ]
        my_filename = filename_list[ i ]
        generate_csv( offset, my_filename, square_result_mat )


    rankk_result_mat = generate_result_mat( coefficient_mat, 'rankk' )
    #rankk_result_mat = np.loadtxt( 'rankk_result.csv', dtype='float',delimiter=',')

    alg_num = len( alg_set )
    offset_list   = [ i * alg_num for i in range(6) ]
    filename_list = [
        'model_rankk_1level_abc_1core.csv',
        'model_rankk_1level_ab_1core.csv',
        'model_rankk_1level_naive_1core.csv',
        'model_rankk_2level_abc_1core.csv',
        'model_rankk_2level_ab_1core.csv',
        'model_rankk_2level_naive_1core.csv',
    ]
    for i in range(6):
        offset = offset_list[ i ]
        my_filename = filename_list[ i ]
        generate_csv( offset, my_filename, rankk_result_mat )

    fixk_result_mat = generate_result_mat( coefficient_mat, 'fixk' )
    #fixk_result_mat = np.loadtxt( 'fixk_result.csv', dtype='float',delimiter=',')

    alg_num = len( alg_set )
    offset_list   = [ i * alg_num for i in range(6) ]
    filename_list = [
        'model_fixk_1level_abc_1core.csv',
        'model_fixk_1level_ab_1core.csv',
        'model_fixk_1level_naive_1core.csv',
        'model_fixk_2level_abc_1core.csv',
        'model_fixk_2level_ab_1core.csv',
        'model_fixk_2level_naive_1core.csv',
    ]
    for i in range(6):
        offset = offset_list[ i ]
        my_filename = filename_list[ i ]
        generate_csv( offset, my_filename, fixk_result_mat )

if __name__ == '__main__':
    main()


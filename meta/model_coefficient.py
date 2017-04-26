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
from common_mix import is_nonzero, read_coeffs, generateCoeffs, transpose

def gen_model_coefficient( coeff_filename_mix, level_mix ):

    coeffs_mix = []
    idx = 0
    for coeff_file in coeff_filename_mix:
        coeffs = read_coeffs( coeff_file )
        level = level_mix[idx]
        for level_id in range( level ):
            coeffs_mix.append( coeffs )
        idx += 1

    cur_coeffs = generateCoeffs( coeffs_mix )

    #N_A_mul = 0
    #N_B_mul = 0
    #N_C_mul = 0
    #N_A_add = 0
    #N_B_add = 0
    #N_C_add = 0

    abc_counter   = [ 0 for i in range(6) ]
    ab_counter    = [ 0 for i in range(6) ]
    naive_counter = [ 0 for i in range(6) ]

    #N_mul   = 0
    #N_A_add = 0
    #N_B_add = 0
    #N_C_add = 0

    comp_counter  = [ 0 for i in range(4) ]

    for i, coeff_set in enumerate( transpose( cur_coeffs[0] ) ):
        nonzero_coeffs = [coeff for coeff in coeff_set if is_nonzero(coeff)]
        nnz = len( nonzero_coeffs )
        #if ( nnz == 1 ):
        abc_counter[0]   += nnz
        ab_counter[0]    += nnz
        naive_counter[0] += 1
        naive_counter[3] += nnz + 1 # if nnz == 1, naive_counter[3] += 0
        comp_counter[1]  += nnz - 1

    for i, coeff_set in enumerate( transpose( cur_coeffs[1] ) ):
        nonzero_coeffs = [coeff for coeff in coeff_set if is_nonzero(coeff)]
        nnz = len( nonzero_coeffs )
        #if ( nnz == 1 ):
        abc_counter[1]   += nnz
        ab_counter[1]    += nnz
        naive_counter[1] += 1
        naive_counter[4] += nnz + 1 # if nnz == 1, naive_counter[4] += 0
        comp_counter[2]  += nnz - 1

    for i, coeff_set in enumerate( transpose( cur_coeffs[2] ) ):
        nonzero_coeffs = [coeff for coeff in coeff_set if is_nonzero(coeff)]
        nnz = len( nonzero_coeffs )
        #if ( nnz == 1 ):
        abc_counter[2]   += nnz
        ab_counter[2]    += 1
        ab_counter[5]    += 3 * nnz
        naive_counter[2] += 1
        naive_counter[5] += 3 * nnz

        comp_counter[0]  += 1
        comp_counter[3]  += nnz

    return [ comp_counter, abc_counter, ab_counter, naive_counter ]

def main():
    try:
        coeff_file_mix = [ ]
        level_mix = [ ]
        num_file    = int( sys.argv[1] )
        for file_id in range( num_file ):
            coeff_file = sys.argv[ 2 + file_id * 2 ]
            level = int( sys.argv[ 3 + file_id * 2 ] )
            coeff_file_mix.append( coeff_file )
            level_mix.append( level )
    except:
        raise Exception('USAGE: python model_coefficient.py 2 coeff_file1 L1 coeff_file2 L2')

    print gen_model_coefficient( coeff_file_mix, level_mix )

if __name__ == '__main__':
    main()


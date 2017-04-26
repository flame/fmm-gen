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
import os
from ab_gen_mix import gen_ab_fmm
from abc_gen_mix import gen_abc_fmm
from naive_gen_mix import gen_naive_fmm
from misc_gen import gen_test_makefile, gen_project_makefile, gen_testfile
from alg_dic import alg_dic

# Get the executable file by alg, level, and pack_type
def generateCode( alg_dims_mix, level_mix, pack_type, path_prefix='../' ):
    if path_prefix[-1] != '/':
        path_prefix += '/'

    #jobname = alg + '_' + str(level) + '_' + pack_type
    jobname = '_'.join( [ alg_dims_mix[i]+'-'+str(level_mix[i]) for i in range(len(alg_dims_mix)) ] ) + '_' + pack_type

    gen_dirname = path_prefix + jobname

    if not os.path.exists(gen_dirname):
        os.makedirs(gen_dirname)

    os.system( 'cp -r {0}common/* {1}'.format( path_prefix, gen_dirname ) )
    os.chdir( gen_dirname )
    #os.system( 'make clean' )

    #generate the code for alg, level, pack_type

    # level -> level
    # pack_type, alg, level -> outfile 
    outfile = './dgemm/my_dgemm_' + jobname + '.c'

    # pack_type, alg, level -> lib_name
    lib_name = 'lib' + jobname
    # pack_type, alg, level -> makefile_name
    project_makefile_name = './makefile_' + jobname 
    test_makefile_name    = './test/makefile_' + jobname 
    test_name = './test/test_' + jobname + '.c'

    # alg -> coeff_file, dims
    coeff_file_list = [ alg_dic[ alg ][0] for alg in alg_dims_mix ]
    dims_list = [ alg_dic[ alg ][1] for alg in alg_dims_mix ]
    level_list = level_mix

    # Generate the code
    if ( pack_type == 'abc' ):
        # pack_type, alg, level -> micro_kernel_file
        # pack_type, alg, level -> kernel_header_file
        micro_kernel_file  = './kernels/bl_kernel_' + jobname + '.c'
        kernel_header_file = './include/bl_kernel_' + jobname + '.h'

        gen_abc_fmm( coeff_file_list, dims_list, level_list, outfile, micro_kernel_file, kernel_header_file )


        gen_project_makefile( project_makefile_name+'_st', lib_name, outfile[2:] + ' \\', micro_kernel_file[2:] + ' \\', kernel_header_file[2:] + ' \\', test_makefile_name[7:], False ) 
        gen_project_makefile( project_makefile_name+'_mt', lib_name, outfile[2:] + ' \\', micro_kernel_file[2:] + ' \\', kernel_header_file[2:] + ' \\', test_makefile_name[7:], True ) 


    elif ( pack_type == 'naive' ):

        gen_naive_fmm( coeff_file_list, dims_list, level_list, outfile )

        gen_project_makefile( project_makefile_name+'_st', lib_name, outfile[2:], '', '', test_makefile_name[7:], False ) 
        gen_project_makefile( project_makefile_name+'_mt', lib_name, outfile[2:], '', '', test_makefile_name[7:], True ) 

    elif ( pack_type == 'ab' ):
        
        gen_ab_fmm( coeff_file_list, dims_list, level_list, outfile )

        gen_project_makefile( project_makefile_name+'_st', lib_name, outfile[2:], '', '', test_makefile_name[7:], False ) 
        gen_project_makefile( project_makefile_name+'_mt', lib_name, outfile[2:], '', '', test_makefile_name[7:], True ) 

    else:
        print 'pack_type not supported'

    gen_test_makefile( test_makefile_name, lib_name, test_name[7:] + ' \\' ) 
    gen_testfile( test_name, pack_type )

    # compile and get the lib.a, test.x
    # Compile the code ( make -f **.mk )
    os.system( 'source sourceme.sh; make -f %s_st;' % ( project_makefile_name[2:]) )
    os.system( 'mv test/test_' + jobname + '.x test/test_' + jobname + '_st.x' )
    os.system( 'source sourceme.sh; make clean -f %s_st;' % ( project_makefile_name[2:] ) )
    os.system( 'source sourceme.sh; make -f %s_mt;' % ( project_makefile_name[2:] ) )
    os.system( 'mv test/test_' + jobname + '.x test/test_' + jobname + '_mt.x' )

    os.chdir( path_prefix )

def main():
    try:
        alg_dims_mix = [ ]
        level_mix = [ ]
        num_file    = int( sys.argv[1] )
        for file_id in range( num_file ):
            alg_dims = sys.argv[ 2 + file_id * 2 ]
            level = int( sys.argv[ 3 + file_id * 2 ] )
            alg_dims_mix.append( alg_dims )
            level_mix.append( level )
            
        pack_type = 'abc'
        gen_path  = '../'

        if len(sys.argv) > 2 + 2 * num_file:
            pack_type = sys.argv[ 2 + 2 * num_file ]
        if len(sys.argv) > 3 + 2 * num_file:
            gen_path = sys.argv[ 3 + 2 * num_file ]

    except:
        raise Exception('USAGE: python control.py 2 m1,n1,p1 L1 m2,n2,p2, L2 pack_type gen_path')

    generateCode( alg_dims_mix, level_mix, pack_type, gen_path )

if __name__ == '__main__':
    main()


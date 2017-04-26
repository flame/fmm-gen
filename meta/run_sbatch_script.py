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

from sbatch_util import run_job
import subprocess
import os
import shutil
from control import generateCode

from config import path_prefix

orig_directory = os.getcwd()

test_exe_dir = path_prefix + 'experiments/'

if not os.path.exists(test_exe_dir):
    os.makedirs(test_exe_dir)

# change to the working test_exe_dir
os.chdir( path_prefix )

#1. core number: 1, 10
#2. matrix type: rankk (m=n=8192, k vary from 1024..1024..8192 ), fixk (k=1024, m=n from 1024..1024..20480), square(m=n=k, vary from [256..256..20480] )
#3. range: [256, 256, 20480], 
#4. routine: stra (1level), mulstra(2level)
#ibrun tacc_affinity ./run_stra_square_4core_mnk_256_256_20480.sh
#ibrun tacc_affinity ./run_mulstra_rankk_2core_mn_8192_k_256_256_8192.sh
arg_list = ['nodes', 'procspernode', 'matrix_type', 'test_routine', 'test_label', 'core_num',
            'jobname', 'test_exe_dir']

num_nodes=1
procspernode=1

#alg_set=['strassen']
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

level_num_list=[ 1 ]
#level_num_list=[ 1,2 ]
#pack_type_list=['naive','abc', 'ab']
pack_type_list=['abc']
test_routine_list = []
test_label_list   = []

generate_code_flag   = True
copy_executable_flag = True
test_executable_flag = True

for alg in alg_set:
    for level in level_num_list:
        for pack_type in pack_type_list:
            if generate_code_flag:
                alg_list=[ alg ]
                level_list=[ level ]
                generateCode( alg_list, level_list, pack_type, path_prefix )
            test_routine_list.append( 'test_' + alg + '-' + str(level) + '_' + pack_type )
            test_label_list.append( alg + '_' + str(level) + '_' + pack_type )

#test_routine_list.extend( ['test_dgemm_mkl', 'austin_strassen_1_mkl', 'austin_strassen_2_mkl'] )
#test_label_list.extend( ['mkl_gemm', 'austin_strassen_1_mkl', 'austin_strassen_2_mkl'] )
#test_routine_list.extend( ['austin_fast323_1_mkl', 'austin_fast323_2_mkl'] )
#test_label_list.extend( ['austin_fast323_1_mkl', 'austin_fast323_2_mkl'] )

#matrix_type_list = ['square', 'rankk', 'fixk'] # determine the test_range
matrix_type_list = ['square'] # determine the test_range

core_num_list=[1]
#core_num_list=[1,10]

## put a copy of the executable in the run test_exe_dir to use with core files
if copy_executable_flag:
    for alg in alg_set:
        for level in level_num_list:
            for pack_type in pack_type_list:
                jobname = alg + '-' + str(level) + '_' + pack_type
                os.system( 'cp ' + jobname + '/test/test_' + jobname + '_st.x ' + test_exe_dir )
                os.system( 'cp ' + jobname + '/test/test_' + jobname + '_mt.x ' + test_exe_dir )

# change to the working test_exe_dir
os.chdir( test_exe_dir )

i = 0

# loop over parameters
for core_num in core_num_list:
    for matrix_type in matrix_type_list:
        routine_id = 0
        for test_routine in test_routine_list:
            test_label = test_label_list[routine_id]
            routine_id += 1
            jobname_suffix_string = ""

            jobname = test_label + '_' + str(matrix_type) + '_' + str(core_num) + 'core';
            inputs = [num_nodes, procspernode, matrix_type, test_routine, test_label, core_num,
                      jobname, test_exe_dir]
            input_dict = dict(zip(arg_list, inputs))
            tacc_script_filename = run_job(input_dict)

            i = i + 1

            if test_executable_flag:
                res = subprocess.Popen('/usr/bin/sbatch ' + tacc_script_filename, shell=True)
                print "res:" + str(res)

# return to the original directory
os.chdir(orig_directory)


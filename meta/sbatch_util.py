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

# This function prints a batch script to the command line suitable for sending to sbatch
import shutil
import os, sys, stat
import datetime

# takes a dictionary of argument value pairs
def run_job(args):
    # number of compute cores per node
    try: 
        num_cores_per_node = args['num_cores_per_node']
    except:
        num_cores_per_node = 20
        #num_cores_per_node = 16

    # allowed time
    try: 
        walltime = args['walltime']
    except:
        walltime = '04:00:00'
        # walltime = '12:00:00'
        
    # the job queue
    try: 
        queue = args['queue']
    except:
        queue = 'vis'
        #queue = 'gpu'
        # queue = 'normal'

    # flag for email notifications when jobs complete
    #send_email = True
    send_email = False

    # email address for job completion notifications
    useremail='hjyahead@gmail.com'

    # first, get the number of processors, etc.
    # if these aren't specified, we default to one MPI rank on one node and
    # use the shared executable
    try: 
        nodes = args['nodes']
    except:
        nodes = 1
        
    try:
        procpernode = args['procpernode']
    except:
        procpernode = 1 
        
    num_threads = num_cores_per_node / procpernode    


    # add a timestamp to the cout file
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(' ', '.')
    timestamp = timestamp.replace(':', '.')

    # number of MPI processes to run
    npRequested = nodes * procpernode
    
    try:
        my_jobname=args['jobname']
    except:
        my_jobname='strassen'

    # there is a limit on the size of filenames, which these might exceed
    my_outfilename = 'output_' + my_jobname + '-%j' + '-' + timestamp
    # replace '/' in the string
    my_outfilename = my_outfilename.replace('/', '.')

    my_test_routine = args['test_routine']
    my_test_label   = args['test_label']
    my_matrix_type  = args['matrix_type']
    my_core_num     = args['core_num']

    test_exe_dir      = args['test_exe_dir']

    tacc_bash_filename   = test_exe_dir + "run_"  + my_jobname + ".sh"
    tacc_script_filename = test_exe_dir + "tacc_" + my_jobname + ".sh"

    if my_core_num == 1:
        my_test_routine_path = test_exe_dir + my_test_routine + '_st.x' 
    else:
        my_test_routine_path = test_exe_dir + my_test_routine + '_mt.x' 

    #shutil.copy2( my_test_routine_path, test_exe_dir )

    insert_step_string = "";

    if my_matrix_type == 'square':

        test_range_string  = 'k_start=240\n'
        test_range_string += 'k_end=12200\n'
        test_range_string += 'k_blocksize=240\n'

        my_m = "$k"
        my_k = "$k"
        my_n = "$k"
       
    elif my_matrix_type == 'rankk': #7400, 400, 17000; 200, 200, 16000

        test_range_string  = 'k_start=240\n'
        test_range_string += 'k_end=14400\n'
        test_range_string += 'k_blocksize=240\n'
        my_m = "14400"
        my_k = "$k"
        my_n = "14400"

    elif my_matrix_type == 'fixk':

        test_range_string  = 'k_start=240\n'
        test_range_string += 'k_end=20000\n'
        test_range_string += 'k_blocksize=240\n'

        my_m = "$k"
        my_k = "1024"
        my_n = "$k"

    elif my_matrix_type == 'fixk1200':
        test_range_string  = 'k_start=240\n'
        test_range_string += 'k_end=20000\n'
        test_range_string += 'k_blocksize=240\n'
        my_m = "$k"
        my_k = "1200"
        my_n = "$k"

    else:
        print "wrong branch! my_matrix_type is wrong!"

    if not os.path.exists( test_exe_dir + 'matlab_result' ):
        os.makedirs( test_exe_dir + 'matlab_result' )

    test_exp_string   = \
"""
echo \"sb_{4}=[\" >> {7}/{6}.m
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    {5}
    {0} {1} {2} {3} >> {7}/{6}.m
done
echo \"];\" >> {7}/{6}.m
""".format( my_test_routine_path, my_m, my_n, my_k, my_test_label, insert_step_string, my_jobname, test_exe_dir + 'matlab_result' )

    if ( my_core_num == 20 ):
        jc_size = 2
        ic_size = my_core_num / 2
    else:
        jc_size = 1
        ic_size = my_core_num / 1

    bash_string   = '#!/bin/bash' + '\n'
    bash_string  += 'export KMP_AFFINITY=compact' + '\n'
    bash_string  += 'export OMP_NUM_THREADS=' + str(my_core_num) +'\n'
    bash_string  += 'export BLISLAB_JC_NT=' + str(jc_size) +'\n'
    bash_string  += 'export BLISLAB_IC_NT=' + str(ic_size) +'\n'
    bash_string  += 'export BLISLAB_JR_NT=' + str(1) +'\n\n'
    bash_string  += test_range_string + '\n' + test_exp_string +'\n'
    
    # now, create the bash script
    script_string = '#!/bin/bash' + '\n'
    script_string = script_string + '#SBATCH -J ' + my_jobname + '\n'
    script_string = script_string + '#SBATCH -o ' + my_outfilename + '\n'
    script_string = script_string + '#SBATCH -p ' + queue + '\n'
    script_string = script_string + '#SBATCH -t ' + walltime + '\n'
    script_string = script_string + '#SBATCH -n ' + str(npRequested) + '\n'
    script_string = script_string + '#SBATCH -N ' + str(nodes) + '\n'
    script_string = script_string + '#SBATCH -A ' + 'CompEdu' + '\n'
    
    if send_email:
        script_string = script_string + '#SBATCH --mail-user=' + useremail + '\n'
        script_string = script_string + '#SBATCH --mail-type=end' + '\n'

    script_string = script_string + 'export OMP_NUM_THREADS=' + str(my_core_num) +'\n'
    script_string = script_string + 'ibrun tacc_affinity ' + tacc_bash_filename + '\n'

    tacc_bash_file   = open(tacc_bash_filename, 'w')
    tacc_bash_file.write(bash_string)
    tacc_bash_file.close()

    #os.chmod(tacc_bash_filename, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH )
    os.chmod(tacc_bash_filename, 0770)
    #os.chmod(tacc_bash_filename, stat.S_IRWXG | stat.S_IWGRP | stat.S_IXGRP )

    tacc_script_file = open(tacc_script_filename, 'w')
    tacc_script_file.write(script_string)
    tacc_script_file.close()

    os.chmod(tacc_script_filename, 0770)

    return tacc_script_filename


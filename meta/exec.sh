# Test routines for debugging purposes
python abc_gen_mix.py 2 algorithms/strassen  2,2,2 1 algorithms/smirnov333-23-139 3,3,3 1 my_dgemm_strassen_abc.c bl_dgemm_micro_kernel_stra.c bl_dgemm_kernel.h

python ab_gen_mix.py 2 algorithms/strassen 2,2,2 1 algorithms/grey323-15-103 3,2,3 1 my_dgemm_strassen_ab.c 

python naive_gen_mix.py 2 algorithms/strassen 2,2,2 1 algorithms/grey323-15-103 3,2,3 1 my_dgemm_strassen_naive.c

python model_gen.py algorithms/strassen 2,2,2 1 


#python control.py 1 323 1 abc ${GEN_PATH}
python control.py 1 222 1 abc ${GEN_PATH}
python control.py 1 222 1 abc ../
python control.py 2 strassen 1 323 1 abc ..
python control.py 1 222 2 ab ..

python run_sbatch_script.py

python model_coefficient.py 2 algorithms/strassen 1 algorithms/grey323-15-103 1
python model_coefficient.py 1 algorithms/strassen 1
python model_coefficient.py 1 algorithms/strassen 2

python model_gen.py


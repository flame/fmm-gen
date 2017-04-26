Generating Families of Practical Fast Matrix Multiplication Algorithms

# Install

* Go to `meta` directory:

```
$ cd fmm-gen/meta
```

* Set up environment variables:
Replace `$core_num` with the number of cores the user wants to run.

```
$ export OMP_NUM_THREADS=$core_num
$ export KMP_AFFINITY=compact
```

Note: if hyper-threading is enabled, the following alternative must be used:

```
$ export KMP_AFFINITY=compact,1
```

* Code generators:

  * If you want to generate the different implementations for a specific algorithm:

```
$ python control.py ${N} \
             $m1n1p1 $L1 \
             $m2n2p2 $L2 ...... \
             $m{N}n{N}p{N} $L{N} \
             ${pack_type} ${gen_path}
```
e.g.
```
$ python control.py 2 222 1 323 1 abc \
               ${HOME}/fmm-gen
$ python control.py 1 222 2 abc ../
```

This script will generate the code and compile it.

To further execute the code, go to the generated code directory (e.g. `${HOME}/fmm-gen/222-1_333-1_abc}`, or `../222-2_abc`). 

When `$core_num` is equal to 1,
run 
```
./test/test_xxx-x_st.x $m $n $k
```
When `$core_num` is greater than 1,
run
```
./test/test_xxx-x_mt.x $m $n $k
```

  * If you have access of a job submission system on a cluster, change the path_prefix variable in config.py, then:
$ python run_sbatch_script.py

This script will generate the code for all implementations, compile them, and submit the jobs to SLURM submission queue for execution.

* Hybrid partitions:

```
$ python control.py 1 222 1 abc
$ python control.py 1 222 2 abc
$ python control.py 1 232 1 abc
$ python control.py 1 232 2 abc
$ python control.py 1 333 1 abc
$ python control.py 1 333 2 abc
$ python control.py 2 222 1 232 1 abc
$ python control.py 2 222 1 333 1 abc
```

* Model:

```
$ python model_gen.py
```
This script will generate csv files for plotting the modeled performance curves.

* Evaluation and expected result

The output will include the following components:
  * Input problem size.
  * Running time (in seconds).
  * Effective GFLOPS (circle{1} in Figure 5).

The user can compare the relative Effective GFLOPS for different implementations.
The trend should match the performance curves shown in this paper.
Since the machines may be different from ours, the absolute GFLOPS could be different.


Bugs can be reported to jianyu@cs.utexas.edu


# Citation
For those of you looking for the appropriate article to cite regarding fmm-gen, we
recommend citing our
[IPDPS17 paper](http://www.cs.utexas.edu/~jianyu/papers/ipdps17.pdf): 

```
@inproceedings{FMM:IPDPS17,
    author    = {Jianyu Huang and Leslie Rice and Devin A. Matthews and Robert A. {v}an~{d}e~{G}eijn},
    title     = {Generating Families of Practical Fast Matrix Multiplication Algorithms},
    booktitle = {31st IEEE International Parallel and Distributed Processing Symposium (IPDPS 2017)},
    year      = 2017,
}
``` 

# Acknowledgement
This material was partially sponsored by grants from the National Science Foundation (Awards ACI-1550493), by Intel Corporation through an Intel Parallel Computing Center grant, and by a gift from Qualcomm Foundation.

_Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF)._

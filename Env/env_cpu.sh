export PATH=/vols/build/cms/mkomm/BPH/BMX/Env/env/miniconda/bin:$PATH
source activate tf_cpu
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8 #reduce further if out-of-memory
ulimit -s unlimited
ulimit -v 8380000 #Kib; about 8.6GB

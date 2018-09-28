export PATH=/vols/build/cms/mkomm/BPH/BMX/Env/env/miniconda/bin:$PATH
source activate tf_gpu
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=16 #reduce further if out-of-memory
ulimit -s unlimited

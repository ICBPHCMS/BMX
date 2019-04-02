#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh || return 1
    export OMP_NUM_THREADS=1
    
    python Utils/unpackForTraining.py \
    --mc="Tests/samples/*sig*.root" \
    --data="Tests/samples/*bkg*.root" \
    --repeatSignal 2 \
    -n 2 -b 0 -o $PWD || return 1
    
    python Utils/unpackForTraining.py \
    --mc="Tests/samples/*sig*.root" \
    --data="Tests/samples/*bkg*.root" \
    --repeatSignal 2 \
    -n 2 -b 1 -o $PWD || return 1
    
    python DNN/train.py \
    -i $PWD \
    -o output \
    -b 10 \
    -e 2 || return 1
    
    source deactivate tf_cpu || return 1
}

execute

#!/bin/bash

function execute() 
{
    source Env/env_cpu.sh || return 1
    python Utils/unpackForTraining.py \
    --mc="Tests/samples/*sig*.root" \
    --data="Tests/samples/*bkg*.root" \
    -n 1 -b 0 -o . || return 1
    source deactivate tf_cpu || return 1
}

execute

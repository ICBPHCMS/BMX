#!/bin/bash

function execute() 
{
    python Utils/unpackForTraining.py \
    --mc="Tests/samples/*sig*.root" \
    --data="Tests/samples/*bkg*.root" \
    -n 1 -b 0 -o . || return 1
}

execute

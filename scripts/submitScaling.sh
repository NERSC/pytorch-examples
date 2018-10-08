#!/bin/bash

name=cifar10
config=configs/${name}.yaml
mkdir -p logs
set -ex

function submitOne {
    sbatch -J $name -d singleton -o "logs/$name-%j.out" $@ scripts/batchScript.sh $config
}

submitOne -N 1 -q debug -t 30
submitOne -N 2 -q debug -t 30
submitOne -N 4 -q debug -t 30
submitOne -N 8 -q debug -t 30
submitOne -N 16 -q debug -t 30

submitOne -N 32 -q premium -t 15
submitOne -N 64 -q premium -t 15

submitOne -N 128 -q premium -t 5
submitOne -N 256 -q premium -t 5
submitOne -N 512 -q premium -t 5

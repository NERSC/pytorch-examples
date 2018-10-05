#!/bin/bash

name=cifar10
config=configs/${name}.yaml
#nodes="1 2 4 8 16"
#nodes="32 64"
nodes="128 256 512"
qos="regular"
t="10"

mkdir -p logs

set -ex
for n in $nodes; do
    sbatch -J $name -N $n -q $qos -t $t -d singleton -o "logs/$name-%j.out" \
        scripts/batchScript.sh $config
done

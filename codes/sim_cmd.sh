#!/bin/bash

cd simulation_data

max=1499
for i in `seq 0 $max`
do
  if [ $(($i%5)) -eq 0 ]; then
    sleep 120s
  fi
  cd $i
  blockMesh
  sonicFoam
  foamToVTK -ascii
  cd ..
  echo "$i/1499 is done" > ls-output_0.txt
done

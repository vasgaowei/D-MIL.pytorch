#!/bin/bash
if [ ! -d "log" ]; then
  mkdir log
fi

prefix=$1
year=$2
log=log/${prefix}_${year}.log
nohup bash both_${year}.sh ${prefix} > ${log} 2>&1 &

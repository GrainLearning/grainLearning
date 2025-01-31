#!/bin/sh
#File information and yade version 
yadeVersion="yade"
table="$1"
fileName="$2"
# Hyperthreading information 
cpus="1"
numThreads=$cpus

command="$yadeVersion-batch --job-threads=$numThreads $table $fileName"
#echo $command
eval $command

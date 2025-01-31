#!/bin/bash
#yade version 
yadeVersion="/home/ph344/myYade/install/bin/yadeCloud"
#read table and file name 
table="$1"
fileName="$2"

# PBS
cpus="1"
numThreads=$cpus
arg="$yadeVersion-batch --job-threads=$numThreads"

# Get number of lines n from table file (data is stored in lines 2:n)
n=$(wc -l < "$table")


for i in $(seq 2 $n); do
    command="$arg --lines $i $table $fileName"
    echo $command
    echo "qsub -v command=\"${command}\" platform_shells/rcg/runSingleYade.sh"  > /tmp/jobscript.$$

    cat /tmp/jobscript.$$
    source /tmp/jobscript.$$
    /bin/rm -f /tmp/jobscript.$$
done

# number of simulations nS: table file goes from 2:nS+1      example: line 2- line 81 for simulatios 0 to 79. Thus 80 simulations in total 
nS=$((n-1))


simulationsRunning=true
while $simulationsRunning
do 
    echo "Simulations not done yet"
    sleep 120s
    # get number of .npy files 
    numNPY=`find $pwd  -maxdepth 1 -iname '*.npy'| wc -l`
    if [ "$nS" -eq "$numNPY" ];then
        simulationsRunning=false
    fi
done


echo "Simulations done"

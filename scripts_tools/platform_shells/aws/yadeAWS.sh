#File information and yade version 
yadeVersion="/apps/myYade/install/bin/yadeCloud"
#read table and file name 
table="$1"
fileName="$2"

# Get number of lines n (=number of samples) from table file (data is stored in lines 2:n)
n=$(wc -l < "$table")
lines=($(python tableReader.py $table | tr -d '[],'))

# SLURM information
nodes="1"
cpus="1"
numThreads=$cpus

slurm="sbatch --nodes=$nodes --cpus-per-task=$cpus"


for i in $(seq 2 $n); do
    command="$slurm $yadeVersion-batch --job-threads=$numThreads --lines $i $table $fileName"
    eval $command
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


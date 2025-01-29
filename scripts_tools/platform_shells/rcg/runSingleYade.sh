#!/bin/bash
#
#PBS -l select=1:mem=4gb:ncpus="1"
#PBS -l walltime=12:00:00
#PBS -k oe

module --silent load git/2.24.0

module --silent load cmake/3.16.4

module --silent load python/3.6.3

module --silent load gts/gcc8/0.7.6

module --silent load eigen/gcc8/3.3.7

module --silent load cgal/gcc8/4.8.2

module --silent load metis/gcc8/5.1.0 

module --silent load openblas/gcc8/0.3.7

module --silent load loki/gcc8/0.1.7

module --silent load gmp/gcc8/6.1.2

module --silent load suitesparse/gcc8/4.4.4

module --silent load mpfr/gcc8/3.1.5

module --silent load vtk/gcc8/6.3.0

module --silent load boost/gcc8/1.72-python3.6

module --silent load graphviz/gcc8/2.44.0

module --silent load openmpi/gcc8/2.1.6

module --silent load gl2ps/gcc8/1.3.9 

export PYTHONPATH=/cm/software/apps/yade/2020.01a/lib/python3.6/site-packages/

sleep 3

cd /home/ph344/grainLearning

echo Command is = "${command}"
eval $command


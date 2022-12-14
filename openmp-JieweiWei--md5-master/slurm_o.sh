#!/usr/bin/env zsh
#SBATCH --job-name=md5
#SBATCH --partition=wacc
#SBATCH --ntasks=1
#SBATCH --nodes=1 --cpus-per-task=20
#SBATCH --time=00-00:00:10
#SBATCH -o out.out -e error.err

module load nvidia/cuda gcc/9.4.0

g++ ./src/md5.cpp ./test/test.cpp -Wall -O3 -std=c++17 -o md5_test -fopenmp

./md5_test


#!/usr/bin/env bash
#SBATCH --job-name=MD5
#SBATCH --partition=wacc
#SBATCH --time=00-00:0:05
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o out.out -e error.err

module load nvidia/cuda

nvcc md5CrackerGPU.cu md5LibGPU.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o cracker

./cracker md5Hash.txt wordList.txt


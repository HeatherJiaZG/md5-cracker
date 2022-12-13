#!/usr/bin/env bash
#SBATCH --job-name=MD5
#SBATCH --partition=wacc
#SBATCH --time=00-00:0:02
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o out.out -e error.err

module load nvidia/cuda

make

./md5_gpu 5480205f98c5904343a0337cd17794b1
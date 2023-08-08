#!/bin/bash

module load git/2.10.2
module load leveldb lapack openblas llvm protobuf ffmpeg fftw scons

module load matlab/R2021a

module load gcc/10.2
module load mpi/openmpi_4.0.7_gcc_10.2_slurm22
module load hdf5/1.12.2_openmpi_4.0.7_gcc_10.2_slurm22
module load cuda/11.7.1

module load anaconda/2022.05
module unload python

. /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate
conda activate isi_analysis

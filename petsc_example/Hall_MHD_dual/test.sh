#BSUB -J HallMHD_spatial
#BSUB -n 36
#BSUB -x
#BSUB -o output/test/Hall_MHD.out
#BSUB -e output/test/Hall_MHD.err
#BSUB -W 10080
#BSUB -q batch
#BSUB -R "span[ptile=36]"

# module purge
# module load mpi/mvapich2-2.3.5-gcc-10.2.0

ORDER=1

OUTPUTDIR=output/test

GENERAL_OPTS="-Re 1 -Rm 1 -s 1 -RH 0 -dt 0.01 -tf 0.01 -tc 1 -gammaint 2000 -gammahalf 2000 -rtol 1e-11 -atol 1e-13 -maxit 500 -amr -vis"

nn=(5)

# cd $LS_SUBCWD

mkdir -p $OUTPUTDIR

for i in `seq 0 $((${#nn[@]} - 1))`
do
    mpirun -np 4 ./Hall_MHD_dual -nx ${nn[i]} -ny ${nn[i]} -nz ${nn[i]} -o ${ORDER} -od $OUTPUTDIR $GENERAL_OPTS
done

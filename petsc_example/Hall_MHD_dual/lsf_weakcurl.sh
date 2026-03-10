#BSUB -q batch
#BSUB -R "span[ptile=36]"

if [ "${Machine}" = "LSSC" ]; then
    module purge
    module load mpi/mvapich2-2.3.5-gcc-10.2.0
    cd $LS_SUBCWD
fi

export MV2_NDREG_ENTRIES_MAX=32768
export MV2_NDREG_ENTRIES=16384

mpirun ./test-weakcurl $RUN_OPTS 

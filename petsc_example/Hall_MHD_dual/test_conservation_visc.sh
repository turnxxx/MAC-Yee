export Machine=LSSC

TEST_NAME=conservation_visc

# Test case
TESTCASE=2 # conservation

ORDER=1
NPROC=32

# Mesh
N=10
Nx=${N}
Ny=${N}
Nz=${N}
Size=1
Sx=${Size}
Sy=${Size}
Sz=${Size}
Ax=0
Ay=0
Az=0
PERIODIC="-px -py -pz" # "-px -py -pz"

# Solverinfo
dt=0.1
final_time=1000
Hall="-noHall"
Visc="-visc"
Resist="-noresist"

# linear solver
GAMMAINT=2000
GAMMAHALF=1000
RTOL=1e-15
ATOL=1e-20
MAXIT=500
SOLVER_TYPE=0 # 0: mfem 1: petsc
PETSC_OPT_FILE="petsc-opts.dat"

# AMR
AMR=-noamr
MAX_AMR_ITER_INIT=100
MAX_ELEM_INIT=1000
REFINE_FRAC_INIT=0.6
COARSE_FRAC_INIT=0.3
MAX_AMR_ITER=1
REFINE_FRAC=0.7
COARSE_FRAC=0.3
MAX_ELEM=2000
ERROR_GOAL=1e-2

# Visualization
visual=-vis
plottime=mesh/plot_time_conservation_visc.dat

# Parameters
Re=1000
Rm=1
s=1.0
RH=0.1
Whistler_m=1 # only for Whistler wave test case

#debug
DEBUG="-nodebug"

# Parallel
TIMELIMIT=20000

for j in `seq 0 $((${#dt[@]} - 1))`
do
    
    # OUTPUT
    JOBNAME=Hall_MHD_${TEST_NAME}
    OUTPUTDIR=output/${TEST_NAME}/N${N}Order${ORDER}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    mkdir -p $OUTPUTDIR
    
    export RUN_OPTS="-nx ${Nx} -ny ${Ny} -nz ${Nz} \
                     -sx ${Sx} -sy ${Sy} -sz ${Sz} \
                     -ax ${Ax} -ay ${Ay} -az ${Az} ${PERIODIC} \
                     -o ${ORDER} -dt ${dt} -tf ${final_time} \
                      ${Hall} ${Visc} ${Resist} \
                     -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} \
                     -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} \
                     -stint ${SOLVER_TYPE} -sthalf ${SOLVER_TYPE} -petsc-opts ${PETSC_OPT_FILE} \
                     ${AMR} -maxamrinit ${MAX_AMR_ITER_INIT} -maxelinit ${MAX_ELEM_INIT} \
                      -refinefracinit ${REFINE_FRAC_INIT} -coarsefracinit ${COARSE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC} -coarsefrac ${COARSE_FRAC} \
                     -maxel ${MAX_ELEM} -errgoal ${ERROR_GOAL} \
                     -od ${OUTPUTDIR} ${visual} -pt ${plottime} \
                     -tc ${TESTCASE} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${RH} -WhistlerM ${Whistler_m} \
                     ${DEBUG}"
    
    if [ "$Machine" = "LSSC" ]; then
        bsub -J ${JOBNAME} -n ${NPROC} -o ${BSUB_OUT} -e ${BSUB_ERR} -W ${TIMELIMIT} < lsf_Hall.sh
    else
        mpirun -np 8 ./Hall_MHD_dual $RUN_OPTS
    fi
    unset RUN_OPTS
done


unset Machine

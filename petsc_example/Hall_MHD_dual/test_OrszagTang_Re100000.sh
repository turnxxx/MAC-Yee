export Machine=LSSC

TESTNAME=orszag_tang_Re100000

# Test case
TESTCASE=12 # OrszagTang

# key parameters
# Nz=(25)
# N=(200)
# ORDER=1
# NPROC=(360)
# dt=0.0025
Nz=(16)
N=(80)
ORDER=1
NPROC=(360)
dt=0.001

# Mesh
Sx=6.283185307179586
Sy=6.283185307179586
Sz=1
Ax=0
Ay=0
Az=0
PERIODIC="-px -py -pz" # "-px -py -pz"

# Solverinfo
final_time=10.0
Hall="-noHall"
Visc="-visc"
Resist="-resist"

# linear solver
GAMMAINT=2000
GAMMAHALF=200000
RTOL=1e-12
ATOL=1e-15
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
plottime=mesh/plot_time_orszag_tang.dat

# Parameters
Re=100000
Rm=100000
s=1.0
RH=0.1
Whistler_m=1 # only for Whistler wave test case

#debug
DEBUG="-nodebug"

# Parallel
TIMELIMIT=20000

for j in `seq 0 $((${#N[@]} - 1))`
do
    
    # OUTPUT
    JOBNAME=Hall_orszag_tang_ORDER${ORDER[i]}_N${N[j]}
    OUTPUTDIR=output/${TESTNAME}/order${ORDER[i]}/N${N[j]}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    mkdir -p $OUTPUTDIR
    
    export RUN_OPTS="-nx ${N[j]} -ny ${N[j]} -nz ${Nz[j]} \
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
        bsub -J ${JOBNAME} -n ${NPROC[j]} -o ${BSUB_OUT} -e ${BSUB_ERR} -W ${TIMELIMIT} < lsf_Hall.sh
    else
        mpirun -np 8 ./Hall_MHD_dual $RUN_OPTS
    fi
    unset RUN_OPTS
done

unset Machine
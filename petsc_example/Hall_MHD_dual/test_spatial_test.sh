export Machine=LSSC

# Test case
TESTCASE=3 # spatial:1 spatial2: 10 spatial_test: 3
TESTNAME=spatial_test

# key parameters
# N=(4 8 16 32)
# ORDER=3
# # NPROC=(4 8 16 32)
# NPROC=(8 16 32 64)
# GAMMAINT=(10 2.5 0.625 0.15625)

N=(32)
ORDER=3
NPROC=(96)
GAMMAINT=(0.15625)

# Mesh
Size=1
Sx=${Size}
Sy=${Size}
Sz=${Size}
Ax=0
Ay=0
Az=0
PERIODIC="" # "-px -py -pz"

# Solverinfo
dt=0.01
final_time=0.5
Hall="-noHall"
Visc="-visc"
Resist="-resist"

# linear solver
GAMMAHALF=20000
RTOL=1e-15
ATOL=1e-15
MAXIT=300
SOLVER_TYPE=0 # 0: mfem 1: petsc
PETSC_OPT_FILE="petsc-opts.dat"
ITERTION_MODE="-iterint -iterhalf"

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
visual=-novis
plottime=mesh/plot_time_spatial.dat

# Parameters
Re=1.0
Rm=1.0
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
    JOBNAME=Hall_MHD_Spatial_ORDER${ORDER[i]}_N${N[j]}
    OUTPUTDIR=output/${TESTNAME}/order${ORDER[i]}/N${N[j]}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    mkdir -p $OUTPUTDIR
    
    export RUN_OPTS="-nx ${N[j]} -ny ${N[j]} -nz ${N[j]} \
                     -sx ${Sx} -sy ${Sy} -sz ${Sz} \
                     -ax ${Ax} -ay ${Ay} -az ${Az} ${PERIODIC} \
                     -o ${ORDER} -dt ${dt} -tf ${final_time} \
                      ${Hall} ${Visc} ${Resist} \
                     -gammaint ${GAMMAINT[j]} -gammahalf ${GAMMAHALF} \
                     -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} \
                     -stint ${SOLVER_TYPE} -sthalf ${SOLVER_TYPE} -petsc-opts ${PETSC_OPT_FILE} ${ITERTION_MODE} \
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
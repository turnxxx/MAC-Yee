export Machine=LSSC

# Test case
TESTCASE=9 # MHD vortex
TESTNAME=MHDvortex

# key parameters
# Nz=(4 6 8 10 12)
# N=(40 60 80 100 120)
# ORDER=2
# NPROC=(72 72 108 144 180)
# timestep=(0.005 0.0033333333333333335 0.0025 0.002 0.0016666666666666668)

Nz=(15 18 20)
N=(150 180 200)
ORDER=2
NPROC=(216 252 360)
timestep=(0.0013333333333333333 0.0011111111111111111 0.001)

# Mesh
Sx=10
Sy=10
Sz=1
Ax=0
Ay=0
Az=0
PERIODIC="-px -py -pz" # "-px -py -pz"

# Solverinfo
final_time=0.1
Hall="-noHall"
Visc="-novisc"
Resist="-noresist"

# linear solver
GAMMAINT=200
GAMMAHALF=10000
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
visual=-novis
plottime=mesh/plot_time_vortex.dat

# Parameters
Re=1000000
Rm=1000000
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
    JOBNAME=Hall_MHD_Vortex_ORDER${ORDER[i]}_N${N[j]}
    OUTPUTDIR=output/${TESTNAME}/order${ORDER[i]}/N${N[j]}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    dt=${timestep[j]}
    
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
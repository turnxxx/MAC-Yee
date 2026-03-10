export Machine=LSSC

# Test case
TESTCASE=19 # island case in original helicity paper

# key parameters
Nz=(40)
Nx=(320)
Ny=(160)
ORDER=1
NPROC=(720)

# Mesh
Sx=2
Sy=1
Sz=0.25
Ax=0
Ay=-0.5
Az=0
PERIODIC="" # "-px -py -pz"

# Solverinfo
dt=0.005
final_time=5.0
Hall="-noHall"
Visc="-visc"
Resist="-resist"

# linear solver
GAMMAINT=200000
GAMMAHALF=200000
RTOL=1e-10
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
plottime=mesh/plot_time_island_origin.dat

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

# OUTPUT
JOBNAME=Hall_island_ORDER${ORDER[i]}_N${N[j]}
OUTPUTDIR=output/island_origin
BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err

mkdir -p $OUTPUTDIR

export RUN_OPTS="-nx ${Nx[j]} -ny ${Ny[j]} -nz ${Nz[j]} \
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

unset Machine
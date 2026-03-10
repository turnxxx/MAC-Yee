export Machine=LSSC

# Test case
TESTCASE=5 # Island

# Parameters
Re=1000
Rm=1000
s=1
RH=0.1

# Time step
dt=0.1
tf=0.1

# solver parameter
GAMMAINT=2000
GAMMAHALF=2000
RTOL=1e-10
ATOL=1e-15
MAXIT=500

# Grid size
Nx=40
Ny=20
Nz=5
Sx=2
Sy=1
Sz=0.25
Ax=0
Ay=-0.5
Az=0
ORDER=1

# OUTPUT
JOBNAME=Hall_MHD_Island
OUTPUTDIR=output/island/Re${Re}Rm${Rm}S${s}RH${RH}/small
BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err

#Visualization
visual=-novis
plottime=mesh/plot_time_island.dat

# AMR
AMR=-noamr
MAX_AMR_ITER_INIT=100
REFINE_FRAC_INIT=0.6
MAX_ELEM_INIT=20000
REFINE_FRAC=0.6
MAX_AMR_ITER=1

# Parallel
NPROC=36
TIMELIMIT=20000


export RUN_OPTS="-nx ${Nx} -ny ${Ny} -nz ${Nz} -sx ${Sx} -sy ${Sy} -sz ${Sz} -ax ${Ax} -ay ${Ay} -az ${Az} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${RH} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} -pt ${plottime} ${AMR} -maxelinit ${MAX_ELEM_INIT} -maxamrinit ${MAX_AMR_ITER_INIT} -refinefracinit ${REFINE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC}"

mkdir -p $OUTPUTDIR

bsub -J ${JOBNAME} -n $NPROC -o $BSUB_OUT -e $BSUB_ERR -W ${TIMELIMIT} < lsf_Hall.sh

unset RUN_OPTS
unset Machine

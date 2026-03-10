
# Machine=LSSC

# Test case
TESTCASE=6 # adaptive

# Parameters
Re=1
Rm=1
s=1
RH=0

# Time step
dt=0.01
tf=0.05

# solver parameter
GAMMAINT=2000
GAMMAHALF=2000
RTOL=1e-11
ATOL=1e-13
MAXIT=500

# Grid size
N=5
ORDER=1

# OUTPUT
JOBNAME=Hall_MHD_Adaptive
OUTPUTDIR=output/adaptive/withamr
BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err

#Visualization
visual=-vis

# AMR
AMR=-amr
MAX_AMR_ITER_INIT=100
MAX_ELEM_INIT=1000
REFINE_FRAC_INIT=0.6
COARSE_FRAC_INIT=0.3

MAX_AMR_ITER=1
REFINE_FRAC=0.7
COARSE_FRAC=0.3

# Parallel
NPROC=36

export RUN_OPTS="-nx ${N} -ny ${N} -nz ${N} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${RH} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} ${AMR} -maxelinit ${MAX_ELEM_INIT} -maxamrinit ${MAX_AMR_ITER_INIT} -refinefracinit ${REFINE_FRAC_INIT} -coarsefracinit ${COARSE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC} -coarsefrac ${COARSE_FRAC} -nodebug"

mkdir -p $OUTPUTDIR

if [ "${Machine}" = "LSSC" ]; then
    bsub -J ${JOBNAME} -n $NPROC -o $BSUB_OUT -e $BSUB_ERR -W 10080 < lsf_Hall.sh
else
    mpirun -np 1 ./Hall_MHD_dual $RUN_OPTS
fi

unset RUN_OPTS

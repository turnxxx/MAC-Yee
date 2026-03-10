export Machine=LSSC

# Test case
TESTCASE=7 # Island

# Parameters
Re=1000
Rm=1000
s=1
RH=(0)

# Time step
dt=0.002
tf=2.0

# solver parameter
GAMMAINT=200000
GAMMAHALF=200000
RTOL=1e-10
ATOL=1e-15
MAXIT=500

# Grid size
Nx=80
Ny=80
Nz=80
Sx=2
Sy=2
Sz=2
Ax=-1.0
Ay=-1.0
Az=-1.0
ORDER=1

#Visualization
visual=-vis
plottime=mesh/plot_time_island_guo.dat

# AMR
AMR=-noamr
MAX_AMR_ITER_INIT=100
REFINE_FRAC_INIT=0.6
MAX_ELEM_INIT=20000
REFINE_FRAC=0.6
MAX_AMR_ITER=1

# Parallel
NPROC=180
TIMELIMIT=20000

for rh in ${RH[@]}
do
    # OUTPUT
    JOBNAME=Hall_MHD_Island
    OUTPUTDIR=output/island_guo/Re${Re}Rm${Rm}S${s}RH${rh}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    export RUN_OPTS="-nx ${Nx} -ny ${Ny} -nz ${Nz} -sx ${Sx} -sy ${Sy} -sz ${Sz} -ax ${Ax} -ay ${Ay} -az ${Az} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${rh} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} -pt ${plottime} ${AMR} -maxelinit ${MAX_ELEM_INIT} -maxamrinit ${MAX_AMR_ITER_INIT} -refinefracinit ${REFINE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC}"
    mkdir -p $OUTPUTDIR
    bsub -J ${JOBNAME} -n $NPROC -o $BSUB_OUT -e $BSUB_ERR -W ${TIMELIMIT} < lsf_Hall.sh
    
    unset RUN_OPTS
done

unset Machine

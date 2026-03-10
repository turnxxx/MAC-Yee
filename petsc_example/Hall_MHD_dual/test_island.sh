export Machine=LSSC

# Test case
TESTCASE=5 # Island

# Parameters
Re=1000
Rm=1000
s=1
# RH=(0.01 0.025 0.05 0.1 0.15)
RH=(0 0.025 0.05 0.1

# Time step
dt=0.02
tf=12.0

# solver parameter
GAMMAINT=20000
GAMMAHALF=20000
RTOL=1e-10
ATOL=1e-15
MAXIT=500

# Grid size
Nx=160
Ny=160
Nz=20
Sx=2
Sy=2
Sz=0.25
Ax=0
Ay=-1
Az=0
ORDER=1

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

# Periodic boundary conditions
PBC="-pz"

# Parallel
NPROC=180
TIMELIMIT=20000

for rh in ${RH[@]}
do
    # OUTPUT
    JOBNAME=Hall_MHD_Island
    OUTPUTDIR=output/island_hu/Re${Re}Rm${Rm}S${s}RH${rh}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    export RUN_OPTS="-nx ${Nx} -ny ${Ny} -nz ${Nz} -sx ${Sx} -sy ${Sy} -sz ${Sz} -ax ${Ax} -ay ${Ay} -az ${Az} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${rh} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} -pt ${plottime} ${AMR} -maxelinit ${MAX_ELEM_INIT} -maxamrinit ${MAX_AMR_ITER_INIT} -refinefracinit ${REFINE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC} ${PBC}"
    mkdir -p $OUTPUTDIR
    bsub -J ${JOBNAME} -n $NPROC -o $BSUB_OUT -e $BSUB_ERR -W ${TIMELIMIT} < lsf_Hall.sh
    
    unset RUN_OPTS
done

unset Machine

export Machine=LSSC

# Test case
TESTCASE=8 # Whistler

# Parameters
Re=0
Rm=0
s=1
RH=(0.02)

WHISTLER_M=(1 2)

# Time step
dt=0.001
tf=25

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
visual=-novis
plottime=mesh/plot_time_Whistler.dat

# AMR
AMR=-noamr

# Parallel
NPROC=360
TIMELIMIT=20000

for rh in ${RH[@]}
do

for m in ${WHISTLER_M[@]}
do
    # OUTPUT
    JOBNAME=Hall_MHD_Whistler
    OUTPUTDIR=output/Whistler/Re${Re}Rm${Rm}S${s}RH${rh}M${m}
    BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
    BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
    
    export RUN_OPTS="-nx ${Nx} -ny ${Ny} -nz ${Nz} -sx ${Sx} -sy ${Sy} -sz ${Sz} -ax ${Ax} -ay ${Ay} -az ${Az} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${rh} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} -pt ${plottime} ${AMR} -WhistlerM ${m} -px -py -pz"
    mkdir -p $OUTPUTDIR
    bsub -J ${JOBNAME} -n $NPROC -o $BSUB_OUT -e $BSUB_ERR -W ${TIMELIMIT} < lsf_Hall.sh
    
    unset RUN_OPTS
    
done
done

unset Machine

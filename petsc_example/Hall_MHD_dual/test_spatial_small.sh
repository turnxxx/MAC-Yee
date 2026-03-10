# export Machine=LSSC

# Test case
TESTCASE=1 # spatial

# Parameters
Re=0
Rm=0
s=1
RH=1

# Time step
dt=0.01
tf=0.1

# solver parameter
GAMMAINT=20
GAMMAHALF=2000
RTOL=1e-10
ATOL=1e-15
MAXIT=500

#Visualization
visual=-novis

# AMR
AMR=-noamr

ORDER=(2)
N=(4 8 16)


# Parallel
NPROC=(8 32 128 360)
TIMELIMIT=20000

for i in `seq 0 $((${#ORDER[@]} - 1))`
do
    for j in `seq 0 $((${#N[@]} - 1))`
    do
        
        # OUTPUT
        JOBNAME=Hall_MHD_Spatial_ORDER${ORDER[i]}_N${N[j]}
        OUTPUTDIR=output/spatial/order${ORDER[i]}/N${N[j]}
        BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
        BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err
        
        mkdir -p $OUTPUTDIR
        
        export RUN_OPTS="-nx ${N[j]} -ny ${N[j]} -nz ${N[j]} -o ${ORDER[i]} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${RH} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} ${AMR}"
        
        if [ "$Machine" == "LSSC" ]; then
            bsub -J ${JOBNAME} -n ${NPROC[j]} -o ${BSUB_OUT} -e ${BSUB_ERR} -W ${TIMELIMIT} < lsf_Hall.sh
        else 
            mpirun ./Hall_MHD_dual ${RUN_OPTS}
        fi
        unset RUN_OPTS
    done
done

unset Machine

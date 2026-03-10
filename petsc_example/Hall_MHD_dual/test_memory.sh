#!/bin/bash

# Test case
TESTCASE=6 # adaptive

# Parameters
Re=1
Rm=1
s=1
RH=0

# Time step
dt=0.01
tf=0.02

# solver parameter
GAMMAINT=2000
GAMMAHALF=2000
RTOL=1e-11
ATOL=1e-13
MAXIT=500

# Grid size
N=4
ORDER=1

# OUTPUT
JOBNAME=Hall_MHD_Adaptive
OUTPUTDIR=output/adaptive/withamr
BSUB_OUT=${OUTPUTDIR}/Hall_MHD.out
BSUB_ERR=${OUTPUTDIR}/Hall_MHD.err

#Visualization
visual=-vis

# AMR
AMR=-noamr
MAX_AMR_ITER_INIT=100
REFINE_FRAC_INIT=0.6
MAX_ELEM_INIT=200
REFINE_FRAC=0.6
MAX_AMR_ITER=1

export RUN_OPTS="-nx ${N} -ny ${N} -nz ${N} -o ${ORDER} -od ${OUTPUTDIR} -Re ${Re} -Rm ${Rm} -s ${s} -RH ${RH} -dt ${dt} -tf ${tf} -tc ${TESTCASE} -gammaint ${GAMMAINT} -gammahalf ${GAMMAHALF} -rtol ${RTOL} -atol ${ATOL} -maxit ${MAXIT} ${visual} ${AMR} -maxelinit ${MAX_ELEM_INIT} -maxamrinit ${MAX_AMR_ITER_INIT} -refinefracinit ${REFINE_FRAC_INIT} -maxamr ${MAX_AMR_ITER} -refinefrac ${REFINE_FRAC}"

mkdir -p $OUTPUTDIR

valgrind --tool=memcheck --leak-check=full ./Hall_MHD_dual $RUN_OPTS &> test_memory.txt
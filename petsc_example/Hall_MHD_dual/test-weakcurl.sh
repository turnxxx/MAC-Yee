order=2
refine=(0 1 2)
NPROC=(1 1 4)

JOBNAME=test-weakcurl


for i in "${refine[@]}"; do
    outputdir=output/test-weakcurl/order${order}/refine${refine[i]}
    
    BSUB_OUT=${outputdir}/test-weakcurl.out
    BSUB_ERR=${outputdir}/test-weakcurl.err
    TIMELIMIT=20000
    
    mkdir -p $outputdir
    
    export RUN_OPTS="-o $order -rs $i"
    
    bsub -J ${JOBNAME} -n ${NPROC[$i]} -o ${BSUB_OUT} -e ${BSUB_ERR} -W ${TIMELIMIT} < lsf_weakcurl.sh
done

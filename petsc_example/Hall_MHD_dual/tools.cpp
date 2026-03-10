
#include "tools.hpp"

// Evaluate a Vector-valued GridFunction at a point
void EvaluateVectorGFAtPoint(const ParGridFunction &gf, const Vector &point, Vector &val)
{
    ParMesh *pmesh = gf.ParFESpace()->GetParMesh();
    
    Vector val_loc;
    int num = 0;
    IntegrationPoint ip;
    int elem_idx=-1;
    ElementTransformation *tran;
    int found = 0;
    int sdim = pmesh->SpaceDimension();
        
    for(int i=0; i<pmesh->GetNE(); i++){
        tran = pmesh->GetElementTransformation(i);
        InverseElementTransformation invtran(tran);
        int ret = invtran.Transform(point, ip);
        if(ret == 0){
            elem_idx = i;
            found = 1;
            break;
        }   
    }
    
    val_loc.SetSize(sdim);
    val_loc = 0.0;
    if(found){
        gf.GetVectorValue(elem_idx, ip, val_loc);
    }
    
    val.SetSize(sdim);
    val = 0.0;
    
    MPI_Allreduce(&found, &num, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
        
    real_t *val_loc_ptr = val_loc.GetData();
    real_t *val_ptr = val.GetData();
    
    MPI_Allreduce(val_loc_ptr, val_ptr, sdim, MPITypeMap<real_t>::mpi_type, MPI_SUM, pmesh->GetComm());
    
    if(num)
    {
        val /= num;
    }
    else{
        val = 9999.0;
        mfem_warning("EvaluateGFAtPoint: Point not found in any element.");
    }
    
    mfemPrintf("EvaluateGFAtPoint [" );
    for(int i=0; i<point.Size(); i++){
        mfemPrintf(" %f, ", point[i]);
    }
    mfemPrintf("] : Found in %d elements, value = [", num);
    for(int i=0; i<val.Size(); i++){
        mfemPrintf(" %f, ", val[i]);
    }
    mfemPrintf("]\n");
}

void Zeromean(ParGridFunction &p)
{

    ParFiniteElementSpace *p_space = p.ParFESpace();

    ParLinearForm int_form(p_space);
    ConstantCoefficient one_coeff(1.0);
    int_form.AddDomainIntegrator(new DomainLFIntegrator(one_coeff));
    int_form.Assemble();
    real_t int_p = 0.0;
    real_t int_p_loc = InnerProduct(int_form, p);
    MPI_Allreduce(&int_p_loc, &int_p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    mfemPrintf("Integral of p: %lg, ", int_p);

    ParGridFunction one_gf(p_space);
    one_gf.ProjectCoefficient(one_coeff);
    real_t vol_loc = InnerProduct(int_form, one_gf);
    real_t vol = 0.0;
    MPI_Allreduce(&vol_loc, &vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    real_t mean_p = int_p / vol;

    p.Add(-mean_p, one_gf);

    int_p_loc = InnerProduct(int_form, p);
    MPI_Allreduce(&int_p_loc, &int_p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mfemPrintf("after modification: %lg\n", int_p);
}

real_t CheckDivergenceFree(ParGridFunction *u_gf)
{
    ConstantCoefficient zerocoeff(0.0);
    real_t Diverror = u_gf->ComputeDivError(&zerocoeff);

    mfemPrintf("Divergence error: %lg\n", Diverror);
    
    return Diverror;
}

void weak_curl(const ParGridFunction &u, ParGridFunction &w, Array<int> bdr_attr)
{
    StopWatch timer;
    
    ParFiniteElementSpace *fes_u = u.ParFESpace();
    ParFiniteElementSpace *fes_w = w.ParFESpace();

    ParBilinearForm *M1_varf;
    M1_varf = new ParBilinearForm(fes_w);
    M1_varf->AddDomainIntegrator(new VectorFEMassIntegrator());
    M1_varf->Assemble();
    M1_varf->Finalize();

    ParMixedBilinearForm *K12_varf;
    K12_varf = new ParMixedBilinearForm(fes_u, fes_w);
    K12_varf->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator());
    K12_varf->Assemble();
    K12_varf->Finalize();
    
    ParLinearForm *f_form = new ParLinearForm(fes_w);
    VectorGridFunctionCoefficient ubdry_coeff(&u);
    f_form->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(ubdry_coeff), bdr_attr);
    f_form->Assemble();
    K12_varf->AddMult(u, *f_form, 1.0);

    HypreParMatrix *M1mat;
    Vector *f_vec;

    M1mat = M1_varf->ParallelAssemble();
    f_vec = f_form->ParallelAssemble();

    HypreDiagScale *M1prec = new HypreDiagScale;

    FGMRESSolver *M1solver = new FGMRESSolver(MPI_COMM_WORLD);
    M1solver->SetAbsTol(1e-15);
    M1solver->SetRelTol(1e-15);
    M1solver->SetMaxIter(500);
    M1solver->SetPreconditioner(*M1prec);
    M1solver->SetOperator(*M1mat);
    M1solver->SetPrintLevel(0);

    Vector w_vec(fes_w->GetTrueVSize());
    w_vec = 0.0;

    timer.Restart();
    M1solver->Mult(*f_vec, w_vec);
    timer.Stop();

    mfemPrintf("weak_curl: %d its, res: %lg . time: %lg \n", M1solver->GetNumIterations(), M1solver->GetFinalNorm(), timer.RealTime());

    w.Distribute(&w_vec);

    delete M1_varf;
    delete K12_varf;
    delete f_form;
    delete M1mat;
    delete f_vec;
    delete M1prec;
    delete M1solver;
}


// w2 = curl u1
void strong_curl(const ParGridFunction &u1, ParGridFunction &w2)
{

    StopWatch timer;
    
    ParFiniteElementSpace *fes_u = u1.ParFESpace();
    ParFiniteElementSpace *fes_w = w2.ParFESpace();

    ParBilinearForm *M2_varf;
    M2_varf = new ParBilinearForm(fes_w);
    M2_varf->AddDomainIntegrator(new VectorFEMassIntegrator());
    M2_varf->Assemble();
    M2_varf->Finalize();

    ParMixedBilinearForm *curl_varf;
    curl_varf = new ParMixedBilinearForm(fes_u, fes_w);
    curl_varf->AddDomainIntegrator(new MixedVectorCurlIntegrator);
    curl_varf->Assemble();
    curl_varf->Finalize();

    ParLinearForm *f_form = new ParLinearForm(fes_w);
    f_form->Assemble();
    curl_varf->AddMult(u1, *f_form, 1.0);

    HypreParMatrix *M2mat;
    Vector *f_vec;

    M2mat = M2_varf->ParallelAssemble();
    f_vec = f_form->ParallelAssemble();

    HypreDiagScale M2prec(*M2mat);
    FGMRESSolver *M2solver = new FGMRESSolver(MPI_COMM_WORLD);
    M2solver->SetAbsTol(1e-15);
    M2solver->SetRelTol(1e-15);
    M2solver->SetMaxIter(500);
    M2solver->SetOperator(*M2mat);
    M2solver->SetPreconditioner(M2prec);
    M2solver->SetPrintLevel(0);

    Vector w_vec(fes_w->GetTrueVSize());
    w_vec = 0.0;

    timer.Restart();
    M2solver->Mult(*f_vec, w_vec);
    timer.Stop();

    mfemPrintf("strong_curl: %d its, res: %lg . time: %lg \n", M2solver->GetNumIterations(), M2solver->GetFinalNorm(), timer.RealTime());

    w2.Distribute(&w_vec);

    delete f_form;
    delete f_vec;

    delete M2_varf;
    delete curl_varf;

    delete M2mat;
    delete M2solver;
}
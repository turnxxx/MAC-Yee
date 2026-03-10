#include "MHD_solver.hpp"
#include "preconditioners.hpp"
#include "ErrorEstimator.hpp"
#include "refiner.hpp"
#include "tools.hpp"

IntegerEvolutionOperator::IntegerEvolutionOperator(ProblemData *pd_,
                                                   ParMesh *pmesh_,
                                                   MHDSolverInfo solver_info_,
                                                   LinearSolverInfo lin_solver_info_,
                                                   AMRInfo amr_info_,
                                                   GridFunctions gfs_,
                                                   GridFunctions old_gfs_)  
    : EvolutionOperator(pd_, pmesh_, solver_info_, lin_solver_info_, amr_info_, gfs_, old_gfs_)
{
    oneondt = new ConstantCoefficient(1.0 / sol_info.dt);
    m_one = new ConstantCoefficient(-1.0);
    gammacoeff = new ConstantCoefficient(lin_info.gamma);
    oneon2Re = new ConstantCoefficient(0.5 / pd->param.Re);
    if(sol_info.resistivity)
    {
        oneon2Rm = new ConstantCoefficient(0.5 / pd->param.Rm);
    }
    else
    {
        oneon2Rm = nullptr;
    }
    ubdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->ubdry_fun);
    Bbdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->Bbdry_fun);
    wbdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->wbdry_fun);
    pbdrycoeff = new FunctionCoefficient(pd->pbdry_fun);

    w2_gf_coeff = new VectorGridFunctionCoefficient(gfs.w2_gf);
    halfw2_coeff = new ScalarVectorProductCoefficient(0.5, *w2_gf_coeff);
    mhalfw2_coeff = new ScalarVectorProductCoefficient(-0.5, *w2_gf_coeff);

    B1_gf_coeff = new VectorGridFunctionCoefficient(gfs.B1_gf);
    halfB1_coeff = new ScalarVectorProductCoefficient(0.5, *B1_gf_coeff);
    halfsB1_coeff = new ScalarVectorProductCoefficient(0.5 * pd->param.s, *B1_gf_coeff);
    mhalfRHB1_coeff = new ScalarVectorProductCoefficient(-0.5 * pd->param.RH, *B1_gf_coeff);

    // bilinear forms
    UU_varf = new ParBilinearForm(sol_info.RTspace);
    UU_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));
    MixedCrossProductIntegrator *convection_integ = new MixedCrossProductIntegrator(*halfw2_coeff);
    convection_integ->SetIntRule(&IntRules.Get(sol_info.RTspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UU_varf->AddDomainIntegrator(convection_integ);
    UU_varf->AddDomainIntegrator(new DivDivIntegrator(*gammacoeff));

    UU_old_varf = new ParBilinearForm(sol_info.RTspace);
    UU_old_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));
    MixedCrossProductIntegrator *convection_old_integ = new MixedCrossProductIntegrator(*mhalfw2_coeff);
    convection_old_integ->SetIntRule(&IntRules.Get(sol_info.RTspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UU_old_varf->AddDomainIntegrator(convection_old_integ);

    UJ_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.RTspace);
    MixedCrossProductIntegrator *lorentz_integ = new MixedCrossProductIntegrator(*halfsB1_coeff);
    lorentz_integ->SetIntRule(&IntRules.Get(sol_info.RTspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UJ_varf->AddDomainIntegrator(lorentz_integ);

    if (sol_info.viscosity)
    {
        UW_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.RTspace);
        UW_varf->AddDomainIntegrator(new VectorFECurlIntegrator(*oneon2Re));

        WW_varf = new ParBilinearForm(sol_info.NDspace);
        WW_varf->AddDomainIntegrator(new VectorFEMassIntegrator);

        WU_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
        WU_varf->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*m_one));
    }
    else
    {
        UW_varf = nullptr;
        WW_varf = nullptr;
        WU_varf = nullptr;
    }

    PU_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.L2space);
    PU_varf->AddDomainIntegrator(new MixedScalarDivergenceIntegrator(*m_one));

    UP_varf = new ParMixedBilinearForm(sol_info.L2space, sol_info.RTspace);
    UP_varf->AddDomainIntegrator(new MixedScalarWeakGradientIntegrator);

    AA_varf = new ParBilinearForm(sol_info.NDspace);
    AA_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));

    AJ_varf = new ParBilinearForm(sol_info.NDspace);
    if (sol_info.resistivity)
        AJ_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneon2Rm));
    if (sol_info.Hall)
    {
        MixedCrossProductIntegrator *hall_integ = new MixedCrossProductIntegrator(*mhalfRHB1_coeff);
        hall_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
        AJ_varf->AddDomainIntegrator(hall_integ);
    }

    AU_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
    MixedCrossProductIntegrator *faraday_integ = new MixedCrossProductIntegrator(*halfB1_coeff);
    faraday_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    AU_varf->AddDomainIntegrator(faraday_integ);

    BB_varf = new ParBilinearForm(sol_info.RTspace);
    BB_varf->AddDomainIntegrator(new VectorFEMassIntegrator);

    BA_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.RTspace);
    BA_varf->AddDomainIntegrator(new MixedVectorCurlIntegrator(*m_one));

    JJ_varf = new ParBilinearForm(sol_info.NDspace);
    JJ_varf->AddDomainIntegrator(new VectorFEMassIntegrator);

    JB_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
    JB_varf->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*m_one));

    /* Matrices */
    UUmat = nullptr;
    UJmat = nullptr;
    UWmat = nullptr;
    WWmat = nullptr;
    WUmat = nullptr;
    PUmat = nullptr;
    UPmat = nullptr;
    AAmat = nullptr;
    AJmat = nullptr;
    AUmat = nullptr;
    BBmat = nullptr;
    BAmat = nullptr;
    JJmat = nullptr;
    JBmat = nullptr;

    WUmat_e = nullptr;
    PUmat_e = nullptr;
    AUmat_e = nullptr;
    UWmat_e = nullptr;
    UUmat_e = nullptr;
    WWmat_e = nullptr;

    pc = nullptr;
    switch (lin_info.type)
    {
    case MFEM:
        fgmres_solver = new FGMRESSolver(MPI_COMM_WORLD);
        fgmres_solver->SetAbsTol(lin_info.atol);
        fgmres_solver->SetRelTol(lin_info.rtol);
        fgmres_solver->SetMaxIter(lin_info.maxit);
        fgmres_solver->SetPrintLevel(3);
        fgmres_solver->iterative_mode = lin_info.iterative_mode;
        solver = fgmres_solver;

        break;

    case PETSC:
        petsc_solver = new PetscLinearSolver(MPI_COMM_WORLD, "integer_", false, true);
        solver = petsc_solver;
        break;

    default:
        mfem_error("Unknown solver type! ");
        break;
    }
}

IntegerEvolutionOperator::~IntegerEvolutionOperator()
{
    if (oneondt)
        delete oneondt;
    if (m_one)
        delete m_one;
    if (gammacoeff)
        delete gammacoeff;
    if (oneon2Re)
        delete oneon2Re;
    if (oneon2Rm)
        delete oneon2Rm;
    if (ubdrycoeff)
        delete ubdrycoeff;
    if (Bbdrycoeff)
        delete Bbdrycoeff;
    if (wbdrycoeff)
        delete wbdrycoeff;
    if (pbdrycoeff)
        delete pbdrycoeff;

    if (w2_gf_coeff)
        delete w2_gf_coeff;
    if (halfw2_coeff)
        delete halfw2_coeff;
    if (mhalfw2_coeff)
        delete mhalfw2_coeff;

    if (B1_gf_coeff)
        delete B1_gf_coeff;
    if (halfB1_coeff)
        delete halfB1_coeff;
    if (halfsB1_coeff)
        delete halfsB1_coeff;
    if (mhalfRHB1_coeff)
        delete mhalfRHB1_coeff;

    if (UU_varf)
        delete UU_varf;
    if (UU_old_varf)
        delete UU_old_varf;
    if (UJ_varf)
        delete UJ_varf;
    if (UW_varf)
        delete UW_varf;
    if (WW_varf)
        delete WW_varf;
    if (WU_varf)
        delete WU_varf;
    if (PU_varf)
        delete PU_varf;
    if (UP_varf)
        delete UP_varf;
    if (AA_varf)
        delete AA_varf;
    if (AJ_varf)
        delete AJ_varf;
    if (AU_varf)
        delete AU_varf;
    if (BB_varf)
        delete BB_varf;
    if (BA_varf)
        delete BA_varf;
    if (JJ_varf)
        delete JJ_varf;
    if (JB_varf)
        delete JB_varf;

    if (UUmat)
        delete UUmat;
    if (UJmat)
        delete UJmat;
    if (UWmat)
        delete UWmat;
    if (WWmat)
        delete WWmat;
    if (WUmat)
        delete WUmat;
    if (PUmat)
        delete PUmat;
    if (UPmat)
        delete UPmat;
    if (AAmat)
        delete AAmat;
    if (AJmat)
        delete AJmat;
    if (AUmat)
        delete AUmat;
    if (BBmat)
        delete BBmat;
    if (BAmat)
        delete BAmat;
    if (JJmat)
        delete JJmat;
    if (JBmat)
        delete JBmat;

    if (WUmat_e)
        delete WUmat_e;
    if (PUmat_e)
        delete PUmat_e;
    if (AUmat_e)
        delete AUmat_e;
    if (UWmat_e)
        delete UWmat_e;
    if (UUmat_e)
        delete UUmat_e;
    if (WWmat_e)
        delete WWmat_e;

    if (pc)
        delete pc;
    if (solver)
        delete solver;
}

void IntegerEvolutionOperator::AssembleOperators()
{
    StopWatch timer;
    timer.Start();

    UU_varf->Update();
    UU_varf->Assemble(0);
    UU_varf->Finalize(0);
    if (UUmat)
        delete UUmat;
    UUmat = UU_varf->ParallelAssemble();

    UU_old_varf->Update();
    UU_old_varf->Assemble(0);
    UU_old_varf->Finalize(0);

    UJ_varf->Update();
    UJ_varf->Assemble(0);
    UJ_varf->Finalize(0);
    if (UJmat)
        delete UJmat;
    UJmat = UJ_varf->ParallelAssemble();

    if (sol_info.viscosity)
    {
        UW_varf->Update();
        UW_varf->Assemble(0);
        UW_varf->Finalize(0);
        if (UWmat)
            delete UWmat;
        UWmat = UW_varf->ParallelAssemble();

        WW_varf->Update();
        WW_varf->Assemble(0);
        WW_varf->Finalize(0);
        if (WWmat)
            delete WWmat;
        WWmat = WW_varf->ParallelAssemble();

        WU_varf->Update();
        WU_varf->Assemble(0);
        WU_varf->Finalize(0);
        if (WUmat)
            delete WUmat;
        WUmat = WU_varf->ParallelAssemble();
    }

    PU_varf->Update();
    PU_varf->Assemble(0);
    PU_varf->Finalize(0);
    if (PUmat)
        delete PUmat;
    PUmat = PU_varf->ParallelAssemble();

    UP_varf->Update();
    UP_varf->Assemble(0);
    UP_varf->Finalize(0);
    if (UPmat)
        delete UPmat;
    UPmat = UP_varf->ParallelAssemble();

    AA_varf->Update();
    AA_varf->Assemble(0);
    AA_varf->Finalize(0);
    if (AAmat)
        delete AAmat;
    AAmat = AA_varf->ParallelAssemble();

    AJ_varf->Update();
    AJ_varf->Assemble(0);
    AJ_varf->Finalize(0);
    if (AJmat)
        delete AJmat;
    AJmat = AJ_varf->ParallelAssemble();

    AU_varf->Update();
    AU_varf->Assemble(0);
    AU_varf->Finalize(0);
    if (AUmat)
        delete AUmat;
    AUmat = AU_varf->ParallelAssemble();

    BB_varf->Update();
    BB_varf->Assemble(0);
    BB_varf->Finalize(0);
    if (BBmat)
        delete BBmat;
    BBmat = BB_varf->ParallelAssemble();

    BA_varf->Update();
    BA_varf->Assemble(0);
    BA_varf->Finalize(0);
    if (BAmat)
        delete BAmat;
    BAmat = BA_varf->ParallelAssemble();

    JJ_varf->Update();
    JJ_varf->Assemble(0);
    JJ_varf->Finalize(0);
    if (JJmat)
        delete JJmat;
    JJmat = JJ_varf->ParallelAssemble();

    JB_varf->Update();
    JB_varf->Assemble(0);
    JB_varf->Finalize(0);
    if (JBmat)
        delete JBmat;
    JBmat = JB_varf->ParallelAssemble();

    // todo: boundary treatment
    Array<int> ess_tdofs_normal_u;
    sol_info.RTspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_u);
    Array<int> ess_tdofs_normal_w;
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_w);
    if (sol_info.viscosity)
        UWmat->EliminateRows(ess_tdofs_normal_u);
    UPmat->EliminateRows(ess_tdofs_normal_u);
    UJmat->EliminateRows(ess_tdofs_normal_u);
    UUmat->EliminateRows(ess_tdofs_normal_u);
    if (sol_info.viscosity)
    {
        WUmat->EliminateRows(ess_tdofs_normal_w);
        WWmat->EliminateRows(ess_tdofs_normal_w);

        if (WUmat_e)
            delete WUmat_e;
        WUmat_e = WUmat->EliminateCols(ess_tdofs_normal_u);
    }

    if (PUmat_e)
        delete PUmat_e;
    PUmat_e = PUmat->EliminateCols(ess_tdofs_normal_u);
    if (AUmat_e)
        delete AUmat_e;
    AUmat_e = AUmat->EliminateCols(ess_tdofs_normal_u);
    if (UUmat_e)
        delete UUmat_e;
    UUmat_e = UUmat->EliminateCols(ess_tdofs_normal_u);
    UUmat->EliminateBC(ess_tdofs_normal_u, Operator::DIAG_ONE);

    if (sol_info.viscosity)
    {
        if (UWmat_e)
            delete UWmat_e;
        UWmat_e = UWmat->EliminateCols(ess_tdofs_normal_w);
        if (WWmat_e)
            delete WWmat_e;
        WWmat_e = WWmat->EliminateCols(ess_tdofs_normal_w);
        WWmat->EliminateBC(ess_tdofs_normal_w, Operator::DIAG_ONE);
    }

    // get trueoffsets
    if (sol_info.viscosity)
    {
        offsets.SetSize(7);
        offsets[0] = 0;
        offsets[1] = sol_info.RTspace->GetTrueVSize();
        offsets[2] = sol_info.NDspace->GetTrueVSize();
        offsets[3] = sol_info.L2space->GetTrueVSize();
        offsets[4] = sol_info.NDspace->GetTrueVSize();
        offsets[5] = sol_info.RTspace->GetTrueVSize();
        offsets[6] = sol_info.NDspace->GetTrueVSize();
        offsets.PartialSum();

        if (BigOp)
            delete BigOp;
        BigOp = new BlockOperator(offsets);
        BigOp->SetBlock(0, 0, UUmat);
        BigOp->SetBlock(0, 1, UWmat);
        BigOp->SetBlock(0, 2, UPmat);
        BigOp->SetBlock(0, 5, UJmat);
        BigOp->SetBlock(1, 0, WUmat);
        BigOp->SetBlock(1, 1, WWmat);
        BigOp->SetBlock(2, 0, PUmat);
        BigOp->SetBlock(3, 0, AUmat);
        BigOp->SetBlock(3, 3, AAmat);
        BigOp->SetBlock(3, 5, AJmat);
        BigOp->SetBlock(4, 4, BBmat);
        BigOp->SetBlock(4, 3, BAmat);
        BigOp->SetBlock(5, 5, JJmat);
        BigOp->SetBlock(5, 4, JBmat);
    }
    else // without w
    {
        offsets.SetSize(6);
        offsets[0] = 0;
        offsets[1] = sol_info.RTspace->GetTrueVSize();
        offsets[2] = sol_info.L2space->GetTrueVSize();
        offsets[3] = sol_info.NDspace->GetTrueVSize();
        offsets[4] = sol_info.RTspace->GetTrueVSize();
        offsets[5] = sol_info.NDspace->GetTrueVSize();
        offsets.PartialSum();

        if (BigOp)
            delete BigOp;
        BigOp = new BlockOperator(offsets);
        BigOp->SetBlock(0, 0, UUmat);
        BigOp->SetBlock(0, 1, UPmat);
        BigOp->SetBlock(0, 4, UJmat);
        BigOp->SetBlock(1, 0, PUmat);
        BigOp->SetBlock(2, 0, AUmat);
        BigOp->SetBlock(2, 2, AAmat);
        BigOp->SetBlock(2, 4, AJmat);
        BigOp->SetBlock(3, 3, BBmat);
        BigOp->SetBlock(3, 2, BAmat);
        BigOp->SetBlock(4, 4, JJmat);
        BigOp->SetBlock(4, 3, JBmat);
    }

    // set operator for solver
    if (fgmres_solver)
    {
        if (pc)
            delete pc;
        pc = new pc_integer(pd, sol_info, lin_info, BigOp, offsets);
        fgmres_solver->SetPreconditioner(*pc);
    }
    solver->SetOperator(*BigOp);
    mfemPrintf("AssembleOperators time: %f\n", timer.RealTime());
}

void IntegerEvolutionOperator::AssembleTDOperators()
{
    StopWatch timer;
    timer.Start();

    UU_varf->Update();
    UU_varf->Assemble(0);
    UU_varf->Finalize(0);
    if (UUmat)
        delete UUmat;
    UUmat = UU_varf->ParallelAssemble();

    UU_old_varf->Update();
    UU_old_varf->Assemble(0);
    UU_old_varf->Finalize(0);

    UJ_varf->Update();
    UJ_varf->Assemble(0);
    UJ_varf->Finalize(0);
    if (UJmat)
        delete UJmat;
    UJmat = UJ_varf->ParallelAssemble();

    // todo: Hall?
    if (sol_info.Hall)
    {
        AJ_varf->Update();
        AJ_varf->Assemble(0);
        AJ_varf->Finalize(0);
        if (AJmat)
            delete AJmat;
        AJmat = AJ_varf->ParallelAssemble();
    }

    AU_varf->Update();
    AU_varf->Assemble(0);
    AU_varf->Finalize(0);
    if (AUmat)
        delete AUmat;
    AUmat = AU_varf->ParallelAssemble();

    // todo: boundary treatment
    Array<int> ess_tdofs_normal_u;
    sol_info.RTspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_u);
    Array<int> ess_tdofs_normal_w;
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_w);

    UJmat->EliminateRows(ess_tdofs_normal_u);
    UUmat->EliminateRows(ess_tdofs_normal_u);
    if (AUmat_e)
        delete AUmat_e;
    AUmat_e = AUmat->EliminateCols(ess_tdofs_normal_u);
    if (UUmat_e)
        delete UUmat_e;
    UUmat_e = UUmat->EliminateCols(ess_tdofs_normal_u);
    UUmat->EliminateBC(ess_tdofs_normal_u, Operator::DIAG_ONE);

    if (sol_info.viscosity)
    {
        BigOp->SetBlock(0, 0, UUmat);
        BigOp->SetBlock(0, 5, UJmat);
        BigOp->SetBlock(3, 0, AUmat);
        if (sol_info.Hall)
            BigOp->SetBlock(3, 5, AJmat);
    }
    else
    {
        BigOp->SetBlock(0, 0, UUmat);
        BigOp->SetBlock(0, 4, UJmat);
        BigOp->SetBlock(2, 0, AUmat);
        if (sol_info.Hall)
            BigOp->SetBlock(2, 4, AJmat);
    }

    // reset preconditioner
    if (fgmres_solver)
    {
        pc->SetTDoperators(UUmat, UJmat, AUmat, sol_info.Hall ? AJmat : nullptr);
    }
    solver->SetOperator(*BigOp);

    mfemPrintf("AssembleTDOperators time: %f\n", timer.RealTime());
}

// Step: Assemble right hand side and solve equation
void IntegerEvolutionOperator::Step(real_t &t_int, real_t dt)
{

    StopWatch timer;
    timer.Start();

    /* right hand side */
    // linear forms
    // linear form for u
    VectorFunctionCoefficient fucoeff(sol_info.dim, pd->fu_fun);
    fucoeff.SetTime(t_int + 0.5 * dt);
    ParLinearForm fu_form(sol_info.RTspace);
    fu_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fucoeff));
    fu_form.Assemble();
    UU_old_varf->AddMult(*old_gfs.u2_gf, fu_form, 1.0);
    if (sol_info.viscosity)
        UW_varf->AddMult(*old_gfs.w1_gf, fu_form, -1.0);
    UJ_varf->AddMult(*old_gfs.j1_gf, fu_form, -1.0);

    ParLinearForm fu_p_form(sol_info.RTspace);
    pbdrycoeff->SetTime(t_int + 0.5 * dt);
    fu_p_form.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(*pbdrycoeff), pd->ess_bdr_tangent);
    fu_p_form.Assemble();
    fu_form -= fu_p_form;

    // linear form for w
    ubdrycoeff->SetTime(t_int + dt);
    ParLinearForm fw_form(sol_info.NDspace);
    fw_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(*ubdrycoeff), pd->ess_bdr_tangent);
    fw_form.Assemble();

    // linear form for p
    ParLinearForm fp_form(sol_info.L2space);
    fp_form.Assemble();

    // linear form for A
    ParLinearForm fA_form(sol_info.NDspace);
    VectorFunctionCoefficient fAcoeff(sol_info.dim, pd->fA_fun);
    fAcoeff.SetTime(t_int + 0.5 * dt);
    fA_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fAcoeff));
    fA_form.Assemble();
    AA_varf->AddMult(*old_gfs.A1_gf, fA_form, 1.0);
    if(sol_info.Hall || sol_info.resistivity)
    {
        AJ_varf->AddMult(*old_gfs.j1_gf, fA_form, -1.0);
    }
    AU_varf->AddMult(*old_gfs.u2_gf, fA_form, -1.0);

    // linear form for B
    ParLinearForm fB_form(sol_info.RTspace);
    VectorFunctionCoefficient Bstabcoeff(sol_info.dim, pd->Bstab_fun);
    if (pd->periodic)
        fB_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Bstabcoeff));
    fB_form.Assemble();

    // linear form for j
    ParLinearForm fJ_form(sol_info.NDspace);
    Bbdrycoeff->SetTime(t_int + dt);
    fJ_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(*Bbdrycoeff), pd->ess_bdr_magnetic);
    fJ_form.Assemble();

    /* apply boundary conditions and form linear system */
    BlockVector Bigvec(offsets), solvec(offsets);
    Bigvec = 0.0;
    solvec = 0.0;

    if (sol_info.viscosity)
    {
        fu_form.ParallelAssemble(Bigvec.GetBlock(0));
        fw_form.ParallelAssemble(Bigvec.GetBlock(1));
        fp_form.ParallelAssemble(Bigvec.GetBlock(2));
        fA_form.ParallelAssemble(Bigvec.GetBlock(3));
        fB_form.ParallelAssemble(Bigvec.GetBlock(4));
        fJ_form.ParallelAssemble(Bigvec.GetBlock(5));
    }
    else
    {
        fu_form.ParallelAssemble(Bigvec.GetBlock(0));
        fp_form.ParallelAssemble(Bigvec.GetBlock(1));
        fA_form.ParallelAssemble(Bigvec.GetBlock(2));
        fB_form.ParallelAssemble(Bigvec.GetBlock(3));
        fJ_form.ParallelAssemble(Bigvec.GetBlock(4));
    }
    
    {
        Array<int> ess_tdofs_normal_u;
        sol_info.RTspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_u);
        Array<int> ess_tdofs_normal_w;
        sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_normal, ess_tdofs_normal_w);

        *gfs.u2_gf = 0.0;
        ubdrycoeff->SetTime(t_int + dt);
        gfs.u2_gf->ProjectBdrCoefficientNormal(*ubdrycoeff, pd->ess_bdr_normal);
        gfs.u2_gf->GetTrueDofs(solvec.GetBlock(0));

        if (sol_info.viscosity)
        {
            wbdrycoeff->SetTime(t_int + dt);
            *gfs.w1_gf = 0.0;
            gfs.w1_gf->ProjectBdrCoefficientTangent(*wbdrycoeff, pd->ess_bdr_normal);
            gfs.w1_gf->GetTrueDofs(solvec.GetBlock(1));

            UUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(0), -1.0);
            WUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(1), -1.0);
            PUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(2), -1.0);
            AUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(3), -1.0);

            WWmat_e->AddMult(solvec.GetBlock(1), Bigvec.GetBlock(1), -1.0);
            UWmat_e->AddMult(solvec.GetBlock(1), Bigvec.GetBlock(0), -1.0);

            Bigvec.GetBlock(0).SetSubVector(ess_tdofs_normal_u, 0.0);
            Bigvec.GetBlock(0).Add(1.0, solvec.GetBlock(0));
            Bigvec.GetBlock(1).SetSubVector(ess_tdofs_normal_w, 0.0);
            Bigvec.GetBlock(1).Add(1.0, solvec.GetBlock(1));
        }
        else
        {
            UUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(0), -1.0);
            PUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(1), -1.0);
            AUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(2), -1.0);

            Bigvec.GetBlock(0).SetSubVector(ess_tdofs_normal_u, 0.0);
            Bigvec.GetBlock(0).Add(1.0, solvec.GetBlock(0));
        }
    }

    timer.Stop();
    mfemPrintf("Assemble rhs and treat bc time: %lg \n", timer.RealTime());
    
    // use old solutions as initial guess
    if(sol_info.viscosity)
    {
        old_gfs.u2_gf->GetTrueDofs(solvec.GetBlock(0));
        old_gfs.w1_gf->GetTrueDofs(solvec.GetBlock(1));
        old_gfs.p3_gf->GetTrueDofs(solvec.GetBlock(2));
        old_gfs.A1_gf->GetTrueDofs(solvec.GetBlock(3));
        old_gfs.B2_gf->GetTrueDofs(solvec.GetBlock(4));
        old_gfs.j1_gf->GetTrueDofs(solvec.GetBlock(5));
    }
    else
    {
        old_gfs.u2_gf->GetTrueDofs(solvec.GetBlock(0));
        old_gfs.p3_gf->GetTrueDofs(solvec.GetBlock(1));
        old_gfs.A1_gf->GetTrueDofs(solvec.GetBlock(2));
        old_gfs.B2_gf->GetTrueDofs(solvec.GetBlock(3));
        old_gfs.j1_gf->GetTrueDofs(solvec.GetBlock(4));
    }

    /* solve */
    timer.Restart();
    solver->Mult(Bigvec, solvec);
    mfemPrintf("Integer step solve time: %lg \n", timer.RealTime());

    /* update grid functions */
    if (sol_info.viscosity)
    {
        gfs.u2_gf->Distribute(solvec.GetBlock(0));
        gfs.w1_gf->Distribute(solvec.GetBlock(1));
        gfs.p3_gf->Distribute(solvec.GetBlock(2));
        gfs.A1_gf->Distribute(solvec.GetBlock(3));
        gfs.B2_gf->Distribute(solvec.GetBlock(4));
        gfs.j1_gf->Distribute(solvec.GetBlock(5));
    }
    else
    {
        gfs.u2_gf->Distribute(solvec.GetBlock(0));
        gfs.p3_gf->Distribute(solvec.GetBlock(1));
        gfs.A1_gf->Distribute(solvec.GetBlock(2));
        gfs.B2_gf->Distribute(solvec.GetBlock(3));
        gfs.j1_gf->Distribute(solvec.GetBlock(4));
        Array<int> bdr_not_periodic(pd->ess_bdr_normal);
        for(int i = 0; i < pd->ess_bdr_tangent.Size(); i++)
        {
            bdr_not_periodic[i] = pd->ess_bdr_tangent[i] || bdr_not_periodic[i];
        }
        weak_curl(*gfs.u2_gf, *gfs.w1_gf, bdr_not_periodic);
    }

    Zeromean(*gfs.p3_gf);
    
}

HalfEvolutionOperator::HalfEvolutionOperator(ProblemData *pd_,
                                             ParMesh *pmesh_,
                                             MHDSolverInfo solver_info_,
                                             LinearSolverInfo lin_solver_info_,
                                             AMRInfo amr_info_,
                                             GridFunctions gfs_,
                                             GridFunctions old_gfs_)
    : EvolutionOperator(pd_, pmesh_, solver_info_, lin_solver_info_, amr_info_, gfs_, old_gfs_)
{
    oneondt = new ConstantCoefficient(1.0 / sol_info.dt);
    m_one = new ConstantCoefficient(-1.0);
    ubdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->ubdry_fun);
    Bbdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->Bbdry_fun);
    wbdrycoeff = new VectorFunctionCoefficient(sol_info.dim, pd->wbdry_fun);
    pbdrycoeff = new FunctionCoefficient(pd->pbdry_fun);

    theta_coeff = new ConstantCoefficient(0.5);
    thetam1_coeff = new SumCoefficient(*theta_coeff, *m_one);
    mtheta_coeff = new ProductCoefficient(*m_one, *theta_coeff);
    mthetam1_coeff = new ProductCoefficient(*m_one, *thetam1_coeff);
    w1_gf_coeff = new VectorGridFunctionCoefficient(gfs.w1_gf);
    thetaw1_coeff = new ScalarVectorProductCoefficient(*theta_coeff, *w1_gf_coeff);
    thetam1_w1_coeff = new ScalarVectorProductCoefficient(*thetam1_coeff, *w1_gf_coeff);

    one_on_Re = new ConstantCoefficient(1.0 / pd->param.Re);
    theta_on_Re = new ProductCoefficient(*theta_coeff, *one_on_Re);
    thetam1_on_Re = new ProductCoefficient(*thetam1_coeff, *one_on_Re);

    if(sol_info.resistivity)
    {
        one_on_Rm = new ConstantCoefficient(1.0 / pd->param.Rm);
        theta_on_Rm = new ProductCoefficient(*theta_coeff, *one_on_Rm);
        thetam1_on_Rm = new ProductCoefficient(*thetam1_coeff, *one_on_Rm);
    }
    else
    {
        one_on_Rm = nullptr;
        theta_on_Rm = nullptr;
        thetam1_on_Rm = nullptr;
    }
    

    B2_gf_coeff = new VectorGridFunctionCoefficient(gfs.B2_gf);
    s_coeff = new ConstantCoefficient(pd->param.s);
    thetaB2_coeff = new ScalarVectorProductCoefficient(*theta_coeff, *B2_gf_coeff);
    sB2_coeff = new ScalarVectorProductCoefficient(*s_coeff, *B2_gf_coeff);
    thetasB2_coeff = new ScalarVectorProductCoefficient(*theta_coeff, *sB2_coeff);
    thetam1sB2_coeff = new ScalarVectorProductCoefficient(*thetam1_coeff, *sB2_coeff);

    RH_coeff = new ConstantCoefficient(pd->param.RH);
    RH_B2_coeff = new ScalarVectorProductCoefficient(*RH_coeff, *B2_gf_coeff);
    mthetaRH_B2_coeff = new ScalarVectorProductCoefficient(*mtheta_coeff, *RH_B2_coeff);
    mthetam1RH_B2_coeff = new ScalarVectorProductCoefficient(*mthetam1_coeff, *RH_B2_coeff);

    // bilinear forms
    UU_varf = new ParBilinearForm(sol_info.NDspace);
    UU_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));
    MixedCrossProductIntegrator *convection_integ = new MixedCrossProductIntegrator(*thetaw1_coeff);
    convection_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UU_varf->AddDomainIntegrator(convection_integ);

    UU_old_varf = new ParBilinearForm(sol_info.NDspace);
    UU_old_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));
    MixedCrossProductIntegrator *convection_old_integ = new MixedCrossProductIntegrator(*thetam1_w1_coeff);
    convection_old_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UU_old_varf->AddDomainIntegrator(convection_old_integ);

    if (sol_info.viscosity)
    {
        UU_varf->AddDomainIntegrator(new CurlCurlIntegrator(*theta_on_Re));
        UU_old_varf->AddDomainIntegrator(new CurlCurlIntegrator(*thetam1_on_Re));
    }

    PU_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.H1space);
    PU_varf->AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(*m_one));

    UP_varf = new ParMixedBilinearForm(sol_info.H1space, sol_info.NDspace);
    UP_varf->AddDomainIntegrator(new MixedVectorGradientIntegrator);

    PP_varf = new ParBilinearForm(sol_info.H1space);
    // PP_varf->AddDomainIntegrator(new MassIntegrator(*epsilon_coeff));

    UJ_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
    MixedCrossProductIntegrator *lorentz_integ = new MixedCrossProductIntegrator(*thetasB2_coeff);
    lorentz_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UJ_varf->AddDomainIntegrator(lorentz_integ);

    UJ_old_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
    MixedCrossProductIntegrator *lorentz_old_integ = new MixedCrossProductIntegrator(*thetam1sB2_coeff);
    lorentz_old_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    UJ_old_varf->AddDomainIntegrator(lorentz_old_integ);

    AA_varf = new ParBilinearForm(sol_info.RTspace);
    AA_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*oneondt));

    AJ_varf = new ParBilinearForm(sol_info.RTspace);
    if (sol_info.resistivity)
    {
        AJ_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*theta_on_Rm));
    }
    if (sol_info.Hall)
    {
        MixedCrossProductIntegrator *Hall_integ = new MixedCrossProductIntegrator(*mthetaRH_B2_coeff);
        Hall_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
        AJ_varf->AddDomainIntegrator(Hall_integ);
    }

    AJ_old_varf = new ParBilinearForm(sol_info.RTspace);
    if (sol_info.resistivity)
    {
        AJ_old_varf->AddDomainIntegrator(new VectorFEMassIntegrator(*thetam1_on_Rm));
    }
    if (sol_info.Hall)
    {
        MixedCrossProductIntegrator *Hall_old_integ = new MixedCrossProductIntegrator(*mthetam1RH_B2_coeff);
        Hall_old_integ->SetIntRule(&IntRules.Get(sol_info.NDspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
        AJ_old_varf->AddDomainIntegrator(Hall_old_integ);
    }

    AU_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.RTspace);
    MixedCrossProductIntegrator *faraday_integ = new MixedCrossProductIntegrator(*thetaB2_coeff);
    faraday_integ->SetIntRule(&IntRules.Get(sol_info.RTspace->GetFE(0)->GetGeomType(), 3*sol_info.order));
    AU_varf->AddDomainIntegrator(faraday_integ);

    BB_varf = new ParBilinearForm(sol_info.NDspace);
    BB_varf->AddDomainIntegrator(new VectorFEMassIntegrator);

    BA_varf = new ParMixedBilinearForm(sol_info.RTspace, sol_info.NDspace);
    BA_varf->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(*m_one));

    JJ_varf = new ParBilinearForm(sol_info.RTspace);
    JJ_varf->AddDomainIntegrator(new VectorFEMassIntegrator);

    JB_varf = new ParMixedBilinearForm(sol_info.NDspace, sol_info.RTspace);
    JB_varf->AddDomainIntegrator(new MixedVectorCurlIntegrator(*m_one));

    /* Matrices */
    UUmat = nullptr;
    PUmat = nullptr;
    UPmat = nullptr;
    PPmat = nullptr;
    UJmat = nullptr;
    AAmat = nullptr;
    AJmat = nullptr;
    AUmat = nullptr;
    BBmat = nullptr;
    BAmat = nullptr;
    JJmat = nullptr;
    JBmat = nullptr;

    PUmat_e = nullptr;
    AUmat_e = nullptr;
    UUmat_e = nullptr;
    UPmat_e = nullptr;
    PPmat_e = nullptr;
    JBmat_e = nullptr;
    BBmat_e = nullptr;

    pc = nullptr;
    switch (lin_info.type)
    {
    case MFEM:
        fgmres_solver = new FGMRESSolver(MPI_COMM_WORLD);
        fgmres_solver->SetAbsTol(lin_info.atol);
        fgmres_solver->SetRelTol(lin_info.rtol);
        fgmres_solver->SetMaxIter(lin_info.maxit);
        fgmres_solver->SetPrintLevel(lin_info.print_level);
        fgmres_solver->iterative_mode = lin_info.iterative_mode;
        solver = fgmres_solver;
        break;

    case PETSC:
        petsc_solver = new PetscLinearSolver(MPI_COMM_WORLD, "half_", false, true);
        solver = petsc_solver;
        break;

    default:
        mfem_error("Unknown solver type! ");
        break;
    }
}

HalfEvolutionOperator::~HalfEvolutionOperator()
{
    delete oneondt;
    delete m_one;
    delete ubdrycoeff;
    delete Bbdrycoeff;
    delete wbdrycoeff;
    delete pbdrycoeff;
    delete theta_coeff;
    delete thetam1_coeff;
    delete mtheta_coeff;
    delete mthetam1_coeff;
    delete w1_gf_coeff;
    delete thetaw1_coeff;
    delete thetam1_w1_coeff;
    delete one_on_Re;
    delete theta_on_Re;
    delete thetam1_on_Re;
    if(sol_info.resistivity)
    {
        delete one_on_Rm;
        delete theta_on_Rm;
        delete thetam1_on_Rm;
    }
    delete B2_gf_coeff;
    delete s_coeff;
    delete thetaB2_coeff;
    delete sB2_coeff;
    delete thetasB2_coeff;
    delete thetam1sB2_coeff;
    delete RH_coeff;
    delete RH_B2_coeff;
    delete mthetaRH_B2_coeff;
    delete mthetam1RH_B2_coeff;

    delete UU_varf;
    delete UU_old_varf;
    delete PU_varf;
    delete UP_varf;
    delete PP_varf;
    delete UJ_varf;
    delete UJ_old_varf;
    delete AA_varf;
    delete AJ_varf;
    delete AJ_old_varf;
    delete AU_varf;
    delete BB_varf;
    delete BA_varf;
    delete JJ_varf;
    delete JB_varf;

    if (UUmat)
        delete UUmat;
    if (PUmat)
        delete PUmat;
    if (UPmat)
        delete UPmat;
    if (PPmat)
        delete PPmat;
    if (UJmat)
        delete UJmat;
    if (AAmat)
        delete AAmat;
    if (AJmat)
        delete AJmat;
    if (AUmat)
        delete AUmat;
    if (BBmat)
        delete BBmat;
    if (BAmat)
        delete BAmat;
    if (JJmat)
        delete JJmat;
    if (JBmat)
        delete JBmat;

    if (PUmat_e)
        delete PUmat_e;
    if (AUmat_e)
        delete AUmat_e;
    if (UUmat_e)
        delete UUmat_e;
    if (UPmat_e)
        delete UPmat_e;
    if (PPmat_e)
        delete PPmat_e;
    if (JBmat_e)
        delete JBmat_e;
    if (BBmat_e)
        delete BBmat_e;

    if (pc)
        delete pc;
}

void HalfEvolutionOperator::AssembleOperators(real_t theta, real_t dt)
{
    StopWatch timer;
    timer.Start();

    theta_coeff->constant = theta;
    oneondt->constant = 1.0/dt;

    UU_varf->Update();
    UU_varf->Assemble(0);
    UU_varf->Finalize(0);
    if (UUmat)
        delete UUmat;
    UUmat = UU_varf->ParallelAssemble();

    UU_old_varf->Update();
    UU_old_varf->Assemble(0);
    UU_old_varf->Finalize(0);

    PU_varf->Update();
    PU_varf->Assemble();
    PU_varf->Finalize();
    if (PUmat)
        delete PUmat;
    PUmat = PU_varf->ParallelAssemble();

    UP_varf->Update();
    UP_varf->Assemble();
    UP_varf->Finalize();
    if (UPmat)
        delete UPmat;
    UPmat = UP_varf->ParallelAssemble();

    PP_varf->Update();
    PP_varf->Assemble(0);
    PP_varf->Finalize(0);
    if (PPmat)
        delete PPmat;
    PPmat = PP_varf->ParallelAssemble();

    UJ_varf->Update();
    UJ_varf->Assemble(0);
    UJ_varf->Finalize(0);
    if (UJmat)
        delete UJmat;
    UJmat = UJ_varf->ParallelAssemble();

    UJ_old_varf->Update();
    UJ_old_varf->Assemble(0);
    UJ_old_varf->Finalize(0);

    AA_varf->Update();
    AA_varf->Assemble(0);
    AA_varf->Finalize(0);
    if (AAmat)
        delete AAmat;
    AAmat = AA_varf->ParallelAssemble();

    AJ_varf->Update();
    AJ_varf->Assemble(0);
    AJ_varf->Finalize(0);
    if (AJmat)
        delete AJmat;
    AJmat = AJ_varf->ParallelAssemble();

    AJ_old_varf->Update();
    AJ_old_varf->Assemble(0);
    AJ_old_varf->Finalize(0);

    AU_varf->Update();
    AU_varf->Assemble(0);
    AU_varf->Finalize(0);
    if (AUmat)
        delete AUmat;
    AUmat = AU_varf->ParallelAssemble();

    BB_varf->Update();
    BB_varf->Assemble(0);
    BB_varf->Finalize(0);
    if (BBmat)
        delete BBmat;
    BBmat = BB_varf->ParallelAssemble();

    BA_varf->Update();
    BA_varf->Assemble(0);
    BA_varf->Finalize(0);
    if (BAmat)
        delete BAmat;
    BAmat = BA_varf->ParallelAssemble();

    JJ_varf->Update();
    JJ_varf->Assemble(0);
    JJ_varf->Finalize(0);
    if (JJmat)
        delete JJmat;
    JJmat = JJ_varf->ParallelAssemble();

    JB_varf->Update();
    JB_varf->Assemble(0);
    JB_varf->Finalize(0);
    if (JBmat)
        delete JBmat;
    JBmat = JB_varf->ParallelAssemble();

    Array<int> ess_tdofs_tangent_ND;
    Array<int> ess_tdofs_tangent_H1;
    Array<int> ess_tdofs_magnetic_ND;
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_ND);
    sol_info.H1space->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_H1);
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_magnetic, ess_tdofs_magnetic_ND);

    UPmat->EliminateRows(ess_tdofs_tangent_ND);
    UJmat->EliminateRows(ess_tdofs_tangent_ND);
    UUmat->EliminateRows(ess_tdofs_tangent_ND);
    PPmat->EliminateRows(ess_tdofs_tangent_H1);
    PUmat->EliminateRows(ess_tdofs_tangent_H1);
    BAmat->EliminateRows(ess_tdofs_magnetic_ND);
    BBmat->EliminateRows(ess_tdofs_magnetic_ND);

    if (PUmat_e)
        delete PUmat_e;
    PUmat_e = PUmat->EliminateCols(ess_tdofs_tangent_ND);
    if (AUmat_e)
        delete AUmat_e;
    AUmat_e = AUmat->EliminateCols(ess_tdofs_tangent_ND);
    if (UUmat_e)
        delete UUmat_e;
    UUmat_e = UUmat->EliminateCols(ess_tdofs_tangent_ND);
    UUmat->EliminateBC(ess_tdofs_tangent_ND, Operator::DIAG_ONE);

    if (UPmat_e)
        delete UPmat_e;
    UPmat_e = UPmat->EliminateCols(ess_tdofs_tangent_H1);
    if (PPmat_e)
        delete PPmat_e;
    PPmat_e = PPmat->EliminateCols(ess_tdofs_tangent_H1);
    PPmat->EliminateBC(ess_tdofs_tangent_H1, Operator::DIAG_ONE);

    if (JBmat_e)
        delete JBmat_e;
    JBmat_e = JBmat->EliminateCols(ess_tdofs_magnetic_ND);
    if (BBmat_e)
        delete BBmat_e;
    BBmat_e = BBmat->EliminateCols(ess_tdofs_magnetic_ND);
    BBmat->EliminateBC(ess_tdofs_magnetic_ND, Operator::DIAG_ONE);

    offsets.SetSize(6);
    offsets[0] = 0;
    offsets[1] = sol_info.NDspace->GetTrueVSize();
    offsets[2] = sol_info.H1space->GetTrueVSize();
    offsets[3] = sol_info.RTspace->GetTrueVSize();
    offsets[4] = sol_info.RTspace->GetTrueVSize();
    offsets[5] = sol_info.NDspace->GetTrueVSize();
    offsets.PartialSum();
    
    if (BigOp)
        delete BigOp;
    BigOp = new BlockOperator(offsets);
    BigOp->SetBlock(0, 0, UUmat);
    BigOp->SetBlock(0, 1, UPmat);
    BigOp->SetBlock(0, 3, UJmat);
    BigOp->SetBlock(1, 0, PUmat);
    BigOp->SetBlock(1, 1, PPmat);
    BigOp->SetBlock(2, 2, AAmat);
    BigOp->SetBlock(2, 3, AJmat);
    BigOp->SetBlock(2, 0, AUmat);
    BigOp->SetBlock(4, 4, BBmat);
    BigOp->SetBlock(4, 2, BAmat);
    BigOp->SetBlock(3, 3, JJmat);
    BigOp->SetBlock(3, 4, JBmat);
    
    // set operator for solver
    if (fgmres_solver)
    {
        if (pc)
            delete pc;
        pc = new pc_half(pd, sol_info, lin_info, BigOp, offsets, theta);
        fgmres_solver->SetPreconditioner(*pc);
    }
    solver->SetOperator(*BigOp);
    mfemPrintf("AssembleOperators time: %f\n", timer.RealTime());
}

void HalfEvolutionOperator::AssembleTDOperators(real_t theta, real_t dt)
{
    StopWatch timer;
    timer.Start();
    
    theta_coeff->constant = theta;
    oneondt->constant = 1.0/dt;

    UU_varf->Update();
    UU_varf->Assemble(0);
    UU_varf->Finalize(0);
    if (UUmat)
        delete UUmat;
    UUmat = UU_varf->ParallelAssemble();

    UU_old_varf->Update();
    UU_old_varf->Assemble(0);
    UU_old_varf->Finalize(0);

    UJ_varf->Update();
    UJ_varf->Assemble(0);
    UJ_varf->Finalize(0);
    if (UJmat)
        delete UJmat;
    UJmat = UJ_varf->ParallelAssemble();
    
    UJ_old_varf->Update();
    UJ_old_varf->Assemble(0);
    UJ_old_varf->Finalize(0);

    if (sol_info.Hall)
    {
        AJ_varf->Update();
        AJ_varf->Assemble(0);
        AJ_varf->Finalize(0);
        if (AJmat)
            delete AJmat;
        AJmat = AJ_varf->ParallelAssemble();
        
        AJ_old_varf->Update();
        AJ_old_varf->Assemble(0);
        AJ_old_varf->Finalize(0);
    }

    AU_varf->Update();
    AU_varf->Assemble(0);
    AU_varf->Finalize(0);
    if (AUmat)
        delete AUmat;
    AUmat = AU_varf->ParallelAssemble();

    Array<int> ess_tdofs_tangent_ND;
    Array<int> ess_tdofs_tangent_H1;
    Array<int> ess_tdofs_magnetic_ND;
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_ND);
    sol_info.H1space->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_H1);
    sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_magnetic, ess_tdofs_magnetic_ND);

    UJmat->EliminateRows(ess_tdofs_tangent_ND);
    UUmat->EliminateRows(ess_tdofs_tangent_ND);
    if (AUmat_e)
        delete AUmat_e;
    AUmat_e = AUmat->EliminateCols(ess_tdofs_tangent_ND);
    if (UUmat_e)
        delete UUmat_e;
    UUmat_e = UUmat->EliminateCols(ess_tdofs_tangent_ND);
    UUmat->EliminateBC(ess_tdofs_tangent_ND, Operator::DIAG_ONE);

    BigOp->SetBlock(0, 0, UUmat);
    BigOp->SetBlock(0, 3, UJmat);
    BigOp->SetBlock(2, 0, AUmat);
    if (sol_info.Hall)
        BigOp->SetBlock(2, 3, AJmat);

    // reset preconditioner
    if (fgmres_solver)
    {
        pc->SetTDOperators(UUmat, UJmat, AUmat, sol_info.Hall ? AJmat : nullptr);
    }
    solver->SetOperator(*BigOp);
    mfemPrintf("AssembleTDOperators time: %f\n", timer.RealTime());
}

void HalfEvolutionOperator::Step(real_t &t_half, real_t dt, real_t theta)
{

    StopWatch timer;
    timer.Start();
    
    theta_coeff->constant = theta;

    /* right hand side */
    // linear forms
    VectorFunctionCoefficient fucoeff(sol_info.dim, pd->fu_fun);
    fucoeff.SetTime(t_half + theta * dt);
    ParLinearForm fu_form(sol_info.NDspace);
    fu_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fucoeff));
    fu_form.Assemble();

    ParLinearForm fu_wbdry_form(sol_info.NDspace);
    wbdrycoeff->SetTime(t_half + theta * dt);
    fu_wbdry_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(*wbdrycoeff), pd->ess_bdr_normal);
    if (sol_info.viscosity)
    {
        fu_wbdry_form.Assemble();
        fu_form.Add(-1.0 / pd->param.Re, fu_wbdry_form);
    }

    UU_old_varf->AddMult(*old_gfs.u1_gf, fu_form, 1.0);
    UJ_old_varf->AddMult(*old_gfs.j2_gf, fu_form, 1.0);

    ParLinearForm fp_form(sol_info.H1space);
    fp_form.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*ubdrycoeff), pd->ess_bdr_normal);
    ubdrycoeff->SetTime(t_half + dt);
    fp_form.Assemble();

    ParLinearForm fA_form(sol_info.RTspace);
    VectorFunctionCoefficient fAcoeff(sol_info.dim, pd->fA_fun);
    fA_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fAcoeff));
    fAcoeff.SetTime(t_half + theta * dt);
    fA_form.Assemble();
    AA_varf->AddMult(*old_gfs.A2_gf, fA_form, 1.0);
    if(sol_info.Hall || sol_info.resistivity)
    {
        AJ_old_varf->AddMult(*old_gfs.j2_gf, fA_form, 1.0);
    }
    AU_varf->AddMult(*old_gfs.u1_gf, fA_form, -(1.0 - theta) / theta);

    ParLinearForm fB_form(sol_info.NDspace);
    VectorFunctionCoefficient Bstabcoeff(sol_info.dim, pd->Bstab_fun);
    if (pd->periodic)
    {
        fB_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Bstabcoeff));
    }
    fB_form.Assemble();

    ParLinearForm fJ_form(sol_info.RTspace);
    fJ_form.Assemble();

    /* apply boundary conditions and form linear system */
    BlockVector Bigvec(offsets), solvec(offsets);
    Bigvec = 0.0;
    solvec = 0.0;

    fu_form.ParallelAssemble(Bigvec.GetBlock(0));
    fp_form.ParallelAssemble(Bigvec.GetBlock(1));
    fA_form.ParallelAssemble(Bigvec.GetBlock(2));
    fJ_form.ParallelAssemble(Bigvec.GetBlock(3));
    fB_form.ParallelAssemble(Bigvec.GetBlock(4));

    // todo: initial guess
    {
        Array<int> ess_tdofs_tangent_ND;
        Array<int> ess_tdofs_tangent_H1;
        Array<int> ess_tdofs_magnetic_ND;
        sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_ND);
        sol_info.H1space->GetEssentialTrueDofs(pd->ess_bdr_tangent, ess_tdofs_tangent_H1);
        sol_info.NDspace->GetEssentialTrueDofs(pd->ess_bdr_magnetic, ess_tdofs_magnetic_ND);

        *gfs.u1_gf = 0.0;
        ubdrycoeff->SetTime(t_half + dt);
        gfs.u1_gf->ProjectBdrCoefficientTangent(*ubdrycoeff, pd->ess_bdr_tangent);
        gfs.u1_gf->GetTrueDofs(solvec.GetBlock(0));

        *gfs.p0_gf = 0.0;
        pbdrycoeff->SetTime(t_half + theta*dt);
        gfs.p0_gf->ProjectBdrCoefficient(*pbdrycoeff, pd->ess_bdr_tangent);
        gfs.p0_gf->GetTrueDofs(solvec.GetBlock(1));

        *gfs.B1_gf = 0.0;
        Bbdrycoeff->SetTime(t_half + dt);
        gfs.B1_gf->ProjectBdrCoefficientTangent(*Bbdrycoeff, pd->ess_bdr_magnetic);
        gfs.B1_gf->GetTrueDofs(solvec.GetBlock(4));

        PUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(1), -1.0);
        AUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(2), -1.0);
        UUmat_e->AddMult(solvec.GetBlock(0), Bigvec.GetBlock(0), -1.0);
        UPmat_e->AddMult(solvec.GetBlock(1), Bigvec.GetBlock(0), -1.0);
        PPmat_e->AddMult(solvec.GetBlock(1), Bigvec.GetBlock(1), -1.0);
        JBmat_e->AddMult(solvec.GetBlock(4), Bigvec.GetBlock(3), -1.0);
        BBmat_e->AddMult(solvec.GetBlock(4), Bigvec.GetBlock(4), -1.0);

        Bigvec.GetBlock(0).SetSubVector(ess_tdofs_tangent_ND, 0.0);
        Bigvec.GetBlock(0).Add(1.0, solvec.GetBlock(0));
        Bigvec.GetBlock(1).SetSubVector(ess_tdofs_tangent_H1, 0.0);
        Bigvec.GetBlock(1).Add(1.0, solvec.GetBlock(1));
        Bigvec.GetBlock(4).SetSubVector(ess_tdofs_magnetic_ND, 0.0);
        Bigvec.GetBlock(4).Add(1.0, solvec.GetBlock(4));
    }

    timer.Stop();
    mfemPrintf("Assemble rhs and treat bc time: %lg \n", timer.RealTime());
    
    /* set old values as initial guess */
    old_gfs.u1_gf->GetTrueDofs(solvec.GetBlock(0));
    old_gfs.p0_gf->GetTrueDofs(solvec.GetBlock(1));
    old_gfs.A2_gf->GetTrueDofs(solvec.GetBlock(2));
    old_gfs.j2_gf->GetTrueDofs(solvec.GetBlock(3));
    old_gfs.B1_gf->GetTrueDofs(solvec.GetBlock(4));

    /* solve */
    timer.Restart();
    solver->Mult(Bigvec, solvec);
    mfemPrintf("Half step solve time: %lg \n", timer.RealTime());

    /* update grid functions */
    gfs.u1_gf->Distribute(solvec.GetBlock(0));
    gfs.p0_gf->Distribute(solvec.GetBlock(1));
    gfs.A2_gf->Distribute(solvec.GetBlock(2));
    gfs.j2_gf->Distribute(solvec.GetBlock(3));
    gfs.B1_gf->Distribute(solvec.GetBlock(4));

    strong_curl(*gfs.u1_gf, *gfs.w2_gf);

    if (pd->ess_bdr_tangent.Max() == 0)
        Zeromean(*gfs.p0_gf);
}

MHD_solver::MHD_solver(ProblemData *pd_,
                       ParMesh *pmesh_,
                       MHDSolverInfo sol_info_,
                       LinearSolverInfo lin_sol_int_,
                       LinearSolverInfo lin_sol_half_,
                       AMRInfo amr_info_,
                       bool visualization_) : pd(pd_),
                                              pmesh(pmesh_),
                                              sol_info(sol_info_),
                                              lin_sol_int(lin_sol_int_),
                                              lin_sol_half(lin_sol_half_),
                                              amr_info(amr_info_),
                                              visualization(visualization_) // changed to use the parameter

{

    // finite element spaces:
    // P(k+1) -> ND(k+1) -> RT(k) -> L2(k)
    H1_coll = new H1_FECollection(sol_info.order, sol_info.dim);
    ND_coll = new ND_FECollection(sol_info.order, sol_info.dim);
    RT_coll = new RT_FECollection(sol_info.order - 1, sol_info.dim);
    L2_coll = new L2_FECollection(sol_info.order - 1, sol_info.dim);

    sol_info.H1space = new ParFiniteElementSpace(pmesh, H1_coll);
    sol_info.NDspace = new ParFiniteElementSpace(pmesh, ND_coll);
    sol_info.RTspace = new ParFiniteElementSpace(pmesh, RT_coll);
    sol_info.L2space = new ParFiniteElementSpace(pmesh, L2_coll);

    HYPRE_BigInt H1_v_size = sol_info.H1space->GlobalTrueVSize();
    HYPRE_BigInt ND_v_size = sol_info.NDspace->GlobalTrueVSize();
    HYPRE_BigInt RT_v_size = sol_info.RTspace->GlobalTrueVSize();
    HYPRE_BigInt L2_v_size = sol_info.L2space->GlobalTrueVSize();

    mfemPrintf("Number of finite element unknowns: H1: %d, ND: %d, RT: %d, L2: %d\n",
               H1_v_size, ND_v_size, RT_v_size, L2_v_size);

    mfemPrintf("Integer: %d, Half: %d\n", RT_v_size + ND_v_size + L2_v_size + ND_v_size + RT_v_size + ND_v_size, ND_v_size + RT_v_size + H1_v_size + RT_v_size + ND_v_size + RT_v_size);

    // grid functions
    gfs.u2_gf = new ParGridFunction(sol_info.RTspace);
    gfs.w1_gf = new ParGridFunction(sol_info.NDspace);
    gfs.p3_gf = new ParGridFunction(sol_info.L2space);
    gfs.A1_gf = new ParGridFunction(sol_info.NDspace);
    gfs.B2_gf = new ParGridFunction(sol_info.RTspace);
    gfs.j1_gf = new ParGridFunction(sol_info.NDspace);

    gfs.u1_gf = new ParGridFunction(sol_info.NDspace);
    gfs.w2_gf = new ParGridFunction(sol_info.RTspace);
    gfs.p0_gf = new ParGridFunction(sol_info.H1space);
    gfs.A2_gf = new ParGridFunction(sol_info.RTspace);
    gfs.B1_gf = new ParGridFunction(sol_info.NDspace);
    gfs.j2_gf = new ParGridFunction(sol_info.RTspace);
    
    old_gfs.u2_gf = new ParGridFunction(sol_info.RTspace);
    old_gfs.w1_gf = new ParGridFunction(sol_info.NDspace);
    old_gfs.p3_gf = new ParGridFunction(sol_info.L2space);
    old_gfs.A1_gf = new ParGridFunction(sol_info.NDspace);
    old_gfs.B2_gf = new ParGridFunction(sol_info.RTspace);
    old_gfs.j1_gf = new ParGridFunction(sol_info.NDspace);
    
    old_gfs.u1_gf = new ParGridFunction(sol_info.NDspace);
    old_gfs.w2_gf = new ParGridFunction(sol_info.RTspace);
    old_gfs.p0_gf = new ParGridFunction(sol_info.H1space);
    old_gfs.A2_gf = new ParGridFunction(sol_info.RTspace);
    old_gfs.B1_gf = new ParGridFunction(sol_info.NDspace);
    old_gfs.j2_gf = new ParGridFunction(sol_info.RTspace);
    

    // visualization
    paraview_dc = nullptr;

    integer_evo = nullptr;
    half_evo = nullptr;
}

MHD_solver::~MHD_solver()
{
    delete integer_evo;
    delete half_evo;

    // delete grid functions
    delete gfs.u2_gf;
    delete gfs.w1_gf;
    delete gfs.p3_gf;
    delete gfs.u1_gf;
    delete gfs.w2_gf;
    delete gfs.p0_gf;

    delete gfs.A1_gf;
    delete gfs.B2_gf;
    delete gfs.j1_gf;
    delete gfs.A2_gf;
    delete gfs.B1_gf;
    delete gfs.j2_gf;

    delete sol_info.H1space;
    delete sol_info.NDspace;
    delete sol_info.RTspace;
    delete sol_info.L2space;

    delete H1_coll;
    delete ND_coll;
    delete RT_coll;
    delete L2_coll;

    if (paraview_dc)
        delete paraview_dc;
}

void MHD_solver::Init()
{
    InitializeByProjection();

    integer_evo = new IntegerEvolutionOperator(pd, pmesh, sol_info, lin_sol_int, amr_info, gfs, old_gfs);
    integer_evo->AssembleOperators();

    half_evo = new HalfEvolutionOperator(pd, pmesh, sol_info, lin_sol_half, amr_info, gfs, old_gfs);
}

void MHD_solver::InitializeByProjection()
{
    mfemPrintf("Initializing by Projection\n");

    StopWatch timer;

    timer.Restart();

    // Coefficients
    VectorFunctionCoefficient u0coeff(sol_info.dim, pd->u0_fun);
    VectorFunctionCoefficient w0coeff(sol_info.dim, pd->w0_fun);
    FunctionCoefficient p0coeff(pd->p0_fun);
    VectorFunctionCoefficient A0coeff(sol_info.dim, pd->A0_fun);
    VectorFunctionCoefficient B0coeff(sol_info.dim, pd->B0_fun);
    VectorFunctionCoefficient j0coeff(sol_info.dim, pd->j0_fun);

    MultipleLpEstimator *estimator = nullptr;
    CustomRefiner *refiner = nullptr;
    ThresholdDerefiner *derefiner = nullptr;

    if (amr_info.amr)
    {
        estimator = new MultipleLpEstimator(2);
        estimator->AppendEstimator(u0coeff, *gfs.u2_gf);
        estimator->AppendEstimator(w0coeff, *gfs.w1_gf);
        estimator->AppendEstimator(p0coeff, *gfs.p3_gf);
        estimator->AppendEstimator(A0coeff, *gfs.A1_gf);
        estimator->AppendEstimator(B0coeff, *gfs.B2_gf);
        estimator->AppendEstimator(j0coeff, *gfs.j1_gf);
        estimator->AppendEstimator(u0coeff, *gfs.u1_gf);
        estimator->AppendEstimator(w0coeff, *gfs.w2_gf);
        estimator->AppendEstimator(p0coeff, *gfs.p0_gf);
        estimator->AppendEstimator(A0coeff, *gfs.A2_gf);
        estimator->AppendEstimator(B0coeff, *gfs.B1_gf);
        estimator->AppendEstimator(j0coeff, *gfs.j2_gf);

        refiner = new CustomRefiner(*estimator);
        refiner->SetTotalErrorFraction(amr_info.refine_frac_init);
        refiner->SetMaxElements(amr_info.max_elements_init);
        refiner->PreferConformingRefinement();
        derefiner = new ThresholdDerefiner(*estimator);
    }

    for (int ref_it = 0; ref_it < (amr_info.amr ? amr_info.max_amr_iter_init : 1); ref_it++)
    {
        if (amr_info.amr)
            mfemPrintf("AMR iteration: %d\n", ref_it);

        // initialize integer solutions
        gfs.u2_gf->ProjectCoefficient(u0coeff);
        gfs.w1_gf->ProjectCoefficient(w0coeff);
        gfs.p3_gf->ProjectCoefficient(p0coeff);
        gfs.A1_gf->ProjectCoefficient(A0coeff);
        gfs.B2_gf->ProjectCoefficient(B0coeff);
        gfs.j1_gf->ProjectCoefficient(j0coeff);

        // initialize half solutions
        gfs.u1_gf->ProjectCoefficient(u0coeff);
        gfs.w2_gf->ProjectCoefficient(w0coeff);
        gfs.p0_gf->ProjectCoefficient(p0coeff);
        gfs.A2_gf->ProjectCoefficient(A0coeff);
        gfs.B1_gf->ProjectCoefficient(B0coeff);
        gfs.j2_gf->ProjectCoefficient(j0coeff);

        if (amr_info.amr)
        {
            refiner->Apply(*pmesh);

            HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
            mfemPrintf("Refinement: threshold = %e, marked element: %lld, total element: %d, max element number: %d \n", refiner->GetThreshold(), refiner->GetNumMarkedElements(), glob_el, amr_info.max_elements_init);

            if (refiner->Stop())
            {
                mfemPrintf("Refinement stopped\n");
                break;
            }

            // update and rebalance
            // update the fe spaces
            UpdateAllFEspaces();
            // update all useful grid functions
            UpdateAllSolutions({});
            UpdateFinishedFEspaces();

            derefiner->SetThreshold(amr_info.coarse_frac_init * refiner->GetTotalError());

            if (derefiner->Apply(*pmesh))
            {
                HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
                mfemPrintf("Derefinement applied. Total Element: %d .\n", glob_el);
                UpdateAllFEspaces();
                UpdateAllSolutions({});
                UpdateFinishedFEspaces();
            }
        }
    }

    if (amr_info.amr)
    {
        delete estimator;
        delete refiner;
        delete derefiner;
    }

    timer.Stop();
    mfemPrintf("Initialization time: %lg\n", timer.RealTime());

    // print initial error
    real_t L2error_u1 = gfs.u1_gf->ComputeL2Error(u0coeff);
    real_t L2error_u2 = gfs.u2_gf->ComputeL2Error(u0coeff);
    real_t L2error_w1 = gfs.w1_gf->ComputeL2Error(w0coeff);
    real_t L2error_w2 = gfs.w2_gf->ComputeL2Error(w0coeff);
    real_t L2error_p0 = gfs.p0_gf->ComputeL2Error(p0coeff);
    real_t L2error_p3 = gfs.p3_gf->ComputeL2Error(p0coeff);
    real_t L2error_A1 = gfs.A1_gf->ComputeL2Error(A0coeff);
    real_t L2error_A2 = gfs.A2_gf->ComputeL2Error(A0coeff);
    real_t L2error_B1 = gfs.B1_gf->ComputeL2Error(B0coeff);
    real_t L2error_B2 = gfs.B2_gf->ComputeL2Error(B0coeff);
    real_t L2error_j1 = gfs.j1_gf->ComputeL2Error(j0coeff);
    real_t L2error_j2 = gfs.j2_gf->ComputeL2Error(j0coeff);

    mfemPrintf("----------Initial L2 error----------\n");
    mfemPrintf("Initial L2 error of u1: %lg\n", L2error_u1);
    mfemPrintf("Initial L2 error of u2: %lg\n", L2error_u2);
    mfemPrintf("Initial L2 error of w1: %lg\n", L2error_w1);
    mfemPrintf("Initial L2 error of w2: %lg\n", L2error_w2);
    mfemPrintf("Initial L2 error of p0: %lg\n", L2error_p0);
    mfemPrintf("Initial L2 error of p3: %lg\n", L2error_p3);
    mfemPrintf("Initial L2 error of A1: %lg\n", L2error_A1);
    mfemPrintf("Initial L2 error of A2: %lg\n", L2error_A2);
    mfemPrintf("Initial L2 error of B1: %lg\n", L2error_B1);
    mfemPrintf("Initial L2 error of B2: %lg\n", L2error_B2);
    mfemPrintf("Initial L2 error of j1: %lg\n", L2error_j1);
    mfemPrintf("Initial L2 error of j2: %lg\n", L2error_j2);
    mfemPrintf("-----------------------------------\n");
    
    // if(pd->has_exact_solution)
    // {
    //     ComputeError(0.0, 0.0, 0.0);
    // }
}

void MHD_solver::ComputeError(real_t t_int, real_t t_half, real_t dt)
{

    MFEM_VERIFY(pd->has_exact_solution, "Exact solution is not provided");

    VectorFunctionCoefficient uexcoeff(sol_info.dim, pd->u_fun);
    VectorFunctionCoefficient wexcoeff(sol_info.dim, pd->w_fun);
    VectorFunctionCoefficient curlwexcoeff(sol_info.dim, pd->curlw_fun);
    FunctionCoefficient pexcoeff(pd->p_fun);
    VectorFunctionCoefficient gradpexcoeff(sol_info.dim, pd->gradp_fun);
    VectorFunctionCoefficient Aexcoeff(sol_info.dim, pd->A_fun);
    FunctionCoefficient divAexcoeff(pd->divA_fun);
    VectorFunctionCoefficient Bexcoeff(sol_info.dim, pd->B_fun);
    VectorFunctionCoefficient jexcoeff(sol_info.dim, pd->j_fun);
    VectorFunctionCoefficient curljexcoeff(sol_info.dim, pd->curlj_fun);
    ConstantCoefficient zerocoeff(0.0);

    uexcoeff.SetTime(t_int);
    real_t L2error_u2 = gfs.u2_gf->ComputeL2Error(uexcoeff);
    real_t Diverror_u2 = gfs.u2_gf->ComputeDivError(&zerocoeff);
    real_t Hdiverror_u2 = gfs.u2_gf->ComputeHDivError(&uexcoeff, &zerocoeff);

    uexcoeff.SetTime(t_half);
    wexcoeff.SetTime(t_half);
    real_t L2error_u1 = gfs.u1_gf->ComputeL2Error(uexcoeff);
    real_t Hcurlerror_u1 = gfs.u1_gf->ComputeHCurlError(&uexcoeff, &wexcoeff);

    wexcoeff.SetTime(t_int);
    curlwexcoeff.SetTime(t_int);
    real_t L2error_w1 = gfs.w1_gf->ComputeL2Error(wexcoeff);
    real_t Hcurlerror_w1 = gfs.w1_gf->ComputeHCurlError(&wexcoeff, &curlwexcoeff);

    wexcoeff.SetTime(t_half);
    real_t L2error_w2 = gfs.w2_gf->ComputeL2Error(wexcoeff);
    real_t Hdiverror_w2 = gfs.w2_gf->ComputeHDivError(&wexcoeff, &zerocoeff);

    pexcoeff.SetTime(max(t_int - 0.5 * dt, 0.0));
    real_t L2error_p3 = gfs.p3_gf->ComputeL2Error(pexcoeff);

    pexcoeff.SetTime(max(t_half - 0.5 * dt, 0.0));
    gradpexcoeff.SetTime(max(t_half - 0.5 * dt, 0.0));
    real_t L2error_p0 = gfs.p0_gf->ComputeL2Error(pexcoeff);
    real_t H1error_p0 = gfs.p0_gf->ComputeH1Error(&pexcoeff, &gradpexcoeff);

    Aexcoeff.SetTime(t_int);
    Bexcoeff.SetTime(t_int);
    real_t L2error_A1 = gfs.A1_gf->ComputeL2Error(Aexcoeff);
    real_t Hcurlerror_A1 = gfs.A1_gf->ComputeHCurlError(&Aexcoeff, &Bexcoeff);

    Aexcoeff.SetTime(t_half);
    divAexcoeff.SetTime(t_half);
    real_t L2error_A2 = gfs.A2_gf->ComputeL2Error(Aexcoeff);
    real_t Hdiverror_A2 = gfs.A2_gf->ComputeHDivError(&Aexcoeff, &divAexcoeff);

    Bexcoeff.SetTime(t_int);
    real_t L2error_B2 = gfs.B2_gf->ComputeL2Error(Bexcoeff);
    real_t Diverror_B2 = gfs.B2_gf->ComputeDivError(&zerocoeff);
    real_t Hdiverror_B2 = gfs.B2_gf->ComputeHDivError(&Bexcoeff, &zerocoeff);

    Bexcoeff.SetTime(t_half);
    jexcoeff.SetTime(t_half);
    real_t L2error_B1 = gfs.B1_gf->ComputeL2Error(Bexcoeff);
    real_t Hcurlerror_B1 = gfs.B1_gf->ComputeHCurlError(&Bexcoeff, &jexcoeff);

    jexcoeff.SetTime(t_int);
    curljexcoeff.SetTime(t_int);
    real_t L2error_j1 = gfs.j1_gf->ComputeL2Error(jexcoeff);
    real_t Hcurlerror_j1 = gfs.j1_gf->ComputeHCurlError(&jexcoeff, &curljexcoeff);

    jexcoeff.SetTime(t_half);
    real_t L2error_j2 = gfs.j2_gf->ComputeL2Error(jexcoeff);
    real_t Diverror_j2 = gfs.j2_gf->ComputeDivError(&zerocoeff);
    real_t Hdiverror_j2 = gfs.j2_gf->ComputeHDivError(&jexcoeff, &zerocoeff);

    mfemPrintf("-------------integer time step errors--------------\n");
    mfemPrintf("t = %f, L2 error of u2: %lg\n", t_int, L2error_u2);
    mfemPrintf("t = %f, Div error of u2: %lg\n", t_int, Diverror_u2);
    mfemPrintf("t = %f, Hdiv error of u2: %lg\n", t_int, Hdiverror_u2);
    mfemPrintf("t = %f, L2 error of w1: %lg\n", t_int, L2error_w1);
    mfemPrintf("t = %f, Hcurl error of w1: %lg\n", t_int, Hcurlerror_w1);
    mfemPrintf("t = %f, L2 error of p3: %lg\n", t_int, L2error_p3);
    mfemPrintf("t = %f, L2 error of A1: %lg\n", t_int, L2error_A1);
    mfemPrintf("t = %f, Hcurl error of A1: %lg\n", t_int, Hcurlerror_A1);
    mfemPrintf("t = %f, L2 error of B2: %lg\n", t_int, L2error_B2);
    mfemPrintf("t = %f, Hdiv error of B2: %lg\n", t_int, Hdiverror_B2);
    mfemPrintf("t = %f, Div error of B2: %lg\n", t_int, Diverror_B2);
    mfemPrintf("t = %f, L2 error of j1: %lg\n", t_int, L2error_j1);
    mfemPrintf("t = %f, Hcurl error of j1: %lg\n", t_int, Hcurlerror_j1);

    mfemPrintf("---------------half time step errors---------------\n");
    mfemPrintf("t = %f, L2 error of u1: %lg\n", t_half, L2error_u1);
    mfemPrintf("t = %f, Hcurl error of u1: %lg\n", t_half, Hcurlerror_u1);
    mfemPrintf("t = %f, L2 error of w2: %lg\n", t_half, L2error_w2);
    mfemPrintf("t = %f, Hdiv error of w2: %lg\n", t_half, Hdiverror_w2);
    mfemPrintf("t = %f, L2 error of p0: %lg\n", t_half, L2error_p0);
    mfemPrintf("t = %f, H1 error of p0: %lg\n", t_half, H1error_p0);
    mfemPrintf("t = %f, L2 error of A2: %lg\n", t_half, L2error_A2);
    mfemPrintf("t = %f, Hdiv error of A2: %lg\n", t_half, Hdiverror_A2);
    mfemPrintf("t = %f, L2 error of B1: %lg\n", t_half, L2error_B1);
    mfemPrintf("t = %f, Hcurl error of B1: %lg\n", t_half, Hcurlerror_B1);
    mfemPrintf("t = %f, L2 error of j2: %lg\n", t_half, L2error_j2);
    mfemPrintf("t = %f, Div error of j2: %lg\n", t_half, Diverror_j2);
    mfemPrintf("t = %f, HDiv error of j2: %lg\n", t_half, Hdiverror_j2);
}


void MHD_solver::ComputeErrorPrimalDual(real_t t, bool print, ofstream *error_out)
{
    
    ParGridFunction u2_mid(*gfs.u2_gf);
    ParGridFunction w1_mid(*gfs.w1_gf);
    ParGridFunction p3_mid(*gfs.p3_gf);
    ParGridFunction A1_mid(*gfs.A1_gf);
    ParGridFunction B2_mid(*gfs.B2_gf);
    ParGridFunction j1_mid(*gfs.j1_gf);
    
    u2_mid += *old_gfs.u2_gf;
    w1_mid += *old_gfs.w1_gf;
    p3_mid += *old_gfs.p3_gf;
    A1_mid += *old_gfs.A1_gf;
    B2_mid += *old_gfs.B2_gf;
    j1_mid += *old_gfs.j1_gf;
    
    u2_mid *= 0.5;
    w1_mid *= 0.5;
    p3_mid *= 0.5;
    A1_mid *= 0.5;
    B2_mid *= 0.5;
    j1_mid *= 0.5;
    
    VectorGridFunctionCoefficient u2_mid_coeff(&u2_mid);
    VectorGridFunctionCoefficient w1_mid_coeff(&w1_mid);
    GridFunctionCoefficient p3_mid_coeff(&p3_mid);
    VectorGridFunctionCoefficient A1_mid_coeff(&A1_mid);
    VectorGridFunctionCoefficient B2_mid_coeff(&B2_mid);
    VectorGridFunctionCoefficient j1_mid_coeff(&j1_mid);
    
    real_t L2error_u = gfs.u1_gf->ComputeL2Error(u2_mid_coeff);
    real_t L2error_w = gfs.w2_gf->ComputeL2Error(w1_mid_coeff);
    real_t L2error_p = gfs.p0_gf->ComputeL2Error(p3_mid_coeff);
    real_t L2error_A = gfs.A2_gf->ComputeL2Error(A1_mid_coeff);
    real_t L2error_B = gfs.B1_gf->ComputeL2Error(B2_mid_coeff);
    real_t L2error_j = gfs.j2_gf->ComputeL2Error(j1_mid_coeff);
    
    if(print)
    {
        mfemPrintf("-------------Primal dual errors--------------\n");
        mfemPrintf("L2 error of u: %lg\n", L2error_u);
        mfemPrintf("L2 error of w: %lg\n", L2error_w);
        mfemPrintf("L2 error of p: %lg\n", L2error_p);
        mfemPrintf("L2 error of A: %lg\n", L2error_A);
        mfemPrintf("L2 error of B: %lg\n", L2error_B);
        mfemPrintf("L2 error of j: %lg\n", L2error_j);
        mfemPrintf("---------------------------------------------\n");
    }
    
    if(error_out)
    {
        *error_out << t <<" " << L2error_u << " " << L2error_w << " " << L2error_p << " " << L2error_A << " " << L2error_B << " " << L2error_j << endl;
    }
    
}

void MHD_solver::SetupParaview(const char *paraview_dir)
{
    if (!visualization)
    {
        if (Mpi::Root())
            std::cerr << "Visualization is not enabled" << std::endl;
        return;
    }

    paraview_dc = new ParaViewDataCollection("MHD_helicity", pmesh);
    paraview_dc->SetPrefixPath(paraview_dir);
    paraview_dc->SetLevelsOfDetail(sol_info.order);
    paraview_dc->SetDataFormat(VTKFormat::BINARY);
    paraview_dc->SetHighOrderOutput(true);

    paraview_dc->RegisterField("u2", gfs.u2_gf);
    paraview_dc->RegisterField("u1", gfs.u1_gf);
    paraview_dc->RegisterField("w1", gfs.w1_gf);
    paraview_dc->RegisterField("A1", gfs.A1_gf);
    paraview_dc->RegisterField("B2", gfs.B2_gf);
    paraview_dc->RegisterField("j1", gfs.j1_gf);

    paraview_dc->RegisterField("w2", gfs.w2_gf);
    paraview_dc->RegisterField("p3", gfs.p3_gf);
    paraview_dc->RegisterField("p0", gfs.p0_gf);
    paraview_dc->RegisterField("A2", gfs.A2_gf);
    paraview_dc->RegisterField("B1", gfs.B1_gf);
    paraview_dc->RegisterField("j2", gfs.j2_gf);
}

void MHD_solver::OutputParaview(const char *paraview_dir, int cycle, real_t t)
{
    if (!visualization)
    {
        if (Mpi::Root())
            std::cerr << "Visualization is not enabled" << std::endl;
        return;
    }

    paraview_dc->SetMesh(pmesh);

    mfemPrintf("Outputting Paraview data\n");
    paraview_dc->SetCycle(cycle);
    paraview_dc->SetTime(t);
    paraview_dc->Save();
    mfemPrintf("Outputted Paraview data, cycle : %d, t = %lg\n", cycle, t);
}

// integer time steps:
int MHD_solver::IntegerStep(real_t &t_int, const real_t dt, bool update)
{
    mfemPrintf("-----------------------------------------------------------\n");
    mfemPrintf("(Integer time step) Advancing from t = %lg to t = %lg \n",
               t_int, t_int + dt);
    mfemPrintf("-----------------------------------------------------------\n");

    StopWatch timer;
    timer.Restart();
    
    sol_info.dt = dt;

    // save old data
    *old_gfs.u2_gf = *gfs.u2_gf;
    *old_gfs.w1_gf = *gfs.w1_gf;
    *old_gfs.p3_gf = *gfs.p3_gf;
    *old_gfs.A1_gf = *gfs.A1_gf;
    *old_gfs.B2_gf = *gfs.B2_gf;
    *old_gfs.j1_gf = *gfs.j1_gf;

    ParGridFunction u2mid_gf(sol_info.RTspace);
    ParGridFunction B2mid_gf(sol_info.RTspace);

    // adaptive mesh refinement
    DifferenceLpEstimator *estimator = nullptr;
    CustomRefiner *refiner = nullptr;
    ThresholdDerefiner *derefiner = nullptr;
    if (amr_info.amr)
    {
        BilinearFormIntegrator *CurlCurlInteg = new CurlCurlIntegrator();
        // estimator = new L2ZienkiewiczZhuEstimator(*CurlCurlInteg, *A1_gf, *RTspace, *NDspace);
        estimator = new DifferenceLpEstimator(2, pmesh);
        estimator->RegisterVariable(u2mid_gf, *gfs.u1_gf);
        estimator->RegisterVariable(B2mid_gf, *gfs.B1_gf);
        refiner = new CustomRefiner(*estimator);
        refiner->SetTotalErrorFraction(amr_info.refine_frac);
        refiner->SetMaxElements(amr_info.max_elements);
        refiner->SetTotalErrorGoal(amr_info.total_err_goal);
        refiner->PreferConformingRefinement();

        derefiner = new ThresholdDerefiner(*estimator);
    }

    for (int ref_it = 0; ref_it < (amr_info.amr ? amr_info.max_amr_iter : 1); ref_it++)
    {

        if (amr_info.amr)
            mfemPrintf("AMR iteration: %d\n", ref_it);

        /* ASSEMBLE */
        timer.Restart();

        if(update)
        {
            integer_evo->AssembleOperators();
        }
        else
        {
            integer_evo->AssembleTDOperators();
        }

        integer_evo->Step(t_int, dt);

        // update mid point values
        u2mid_gf = *gfs.u2_gf;
        u2mid_gf += *old_gfs.u2_gf;
        u2mid_gf *= 0.5;

        B2mid_gf = *gfs.B2_gf;
        B2mid_gf += *old_gfs.B2_gf;
        B2mid_gf *= 0.5;

        if (amr_info.amr)
        {
            // refine mesh
            refiner->Apply(*pmesh);
            HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
            mfemPrintf("Refinement: total error = %e, marked element: %lld, total element: %d \n", refiner->GetTotalError(), refiner->GetNumMarkedElements(), glob_el);

            if (refiner->Stop())
            {
                mfemPrintf("Refinement stopped\n");
                break;
            }

            // update and rebalance
            // update the fe spaces
            UpdateAllFEspaces();
            // update all useful grid functions
            // UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});

            mfemPrintf("Rebalancing mesh\n");
            // rebalance mesh
            if (pmesh->Nonconforming())
            {
                pmesh->Rebalance();
                // update the fe spaces and grid functions again
                UpdateAllFEspaces();
                // UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});
            }

            // todo: tell the evo to update bilinear forms

            // update the bilinear and linear forms
            UpdateFinishedFEspaces();

            derefiner->SetThreshold(amr_info.coarse_frac * refiner->GetTotalError());

            // derefine mesh
            if (derefiner->Apply(*pmesh))
            {
                HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
                mfemPrintf("Derefined elements. Total Elements: %d .\n", glob_el);

                // update the fe spaces
                UpdateAllFEspaces();
                // update all useful grid functions
                // UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});

                // rebalance mesh
                mfemPrintf("Rebalancing mesh\n");
                if (pmesh->Nonconforming())
                {
                    pmesh->Rebalance();
                    // update the fe spaces and grid functions again
                    UpdateAllFEspaces();
                    // UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});
                }

                // todo: tell the evo to update bilinear forms

                UpdateFinishedFEspaces();
            }
        }
    }

    t_int += dt;

    if (amr_info.amr)
    {
        pmesh->PrintInfo();
        delete estimator;
        delete refiner;
        delete derefiner;
    }

    return integer_evo->GetIterCount();
}

// // integer time steps:
// int MHD_solver::IntegerStep(real_t &t_int, const real_t dt)
// {
//     mfemPrintf("-----------------------------------------------------------\n");
//     mfemPrintf("(Integer time step) Advancing from t = %lg to t = %lg \n",
//                t_int, t_int + dt);
//     mfemPrintf("-----------------------------------------------------------\n");

//     StopWatch timer;
//     timer.Restart();

//     // save old data
//     ParGridFunction u2_old(*u2_gf);
//     ParGridFunction w1_old(*w1_gf);
//     ParGridFunction A1_old(*A1_gf);
//     ParGridFunction B2_old(*B2_gf);
//     ParGridFunction j1_old(*j1_gf);

//     ParGridFunction u2mid_gf(RTspace);
//     ParGridFunction B2mid_gf(RTspace);

//     // Coefficients
//     ConstantCoefficient oneondt(1.0 / dt);
//     ConstantCoefficient m_one(-1.0);
//     ConstantCoefficient gammacoeff(gamma_int);
//     VectorGridFunctionCoefficient w2_gf_coeff(w2_gf);
//     VectorFunctionCoefficient ubdrycoeff(dim, pd->ubdry_fun);
//     VectorFunctionCoefficient Bbdrycoeff(dim, pd->Bbdry_fun);
//     VectorFunctionCoefficient wbdrycoeff(dim, pd->wbdry_fun);
//     FunctionCoefficient pbdrycoeff(pd->pbdry_fun);

//     // Bilinear forms
//     ParBilinearForm UU_varf(RTspace);
//     ScalarVectorProductCoefficient halfw2_coeff(0.5, w2_gf_coeff);
//     UU_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));
//     UU_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(halfw2_coeff));
//     UU_varf.AddDomainIntegrator(new DivDivIntegrator(gammacoeff));

//     ParBilinearForm UU_old_varf(RTspace);
//     ScalarVectorProductCoefficient mhalfw2_coeff(-0.5, w2_gf_coeff);
//     UU_old_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));
//     UU_old_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(mhalfw2_coeff));

//     ParMixedBilinearForm UJ_varf(NDspace, RTspace);
//     VectorGridFunctionCoefficient B1_gf_coeff(B1_gf);
//     ScalarVectorProductCoefficient halfsB1_coeff(0.5 * pd->param.s, B1_gf_coeff);
//     UJ_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(halfsB1_coeff));

//     ParMixedBilinearForm UW_varf(NDspace, RTspace);
//     ConstantCoefficient oneon2Re;
//     if (viscosity)
//     {
//         oneon2Re.constant = 1.0 / (2.0 * pd->param.Re);
//         UW_varf.AddDomainIntegrator(new VectorFECurlIntegrator(oneon2Re));
//     }

//     ParBilinearForm WW_varf(NDspace);
//     WW_varf.AddDomainIntegrator(new VectorFEMassIntegrator());

//     ParMixedBilinearForm WU_varf(RTspace, NDspace);
//     WU_varf.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(m_one));

//     ParMixedBilinearForm WU_bdry_varf(NDspace, RTspace);
//     if (!viscosity)
//     {
//         WU_bdry_varf.AddBdrTraceFaceIntegrator(new VectorBdryNormalDotUxVIntegrator(1.0));
//     }

//     ParMixedBilinearForm PU_varf(RTspace, L2space);
//     PU_varf.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(m_one));

//     ParMixedBilinearForm UP_varf(L2space, RTspace);
//     UP_varf.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator);

//     ParBilinearForm AA_varf(NDspace);
//     AA_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));

//     ParBilinearForm AJ_varf(NDspace);
//     ConstantCoefficient oneon2Rm;
//     if (resistivity)
//     {
//         oneon2Rm.constant = 1.0 / (2.0 * pd->param.Rm);
//         AJ_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneon2Rm));
//     }

//     ScalarVectorProductCoefficient mhalfRHB1_coeff(-0.5 * pd->param.RH, B1_gf_coeff);
//     if (Hall)
//     {
//         AJ_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(mhalfRHB1_coeff));
//     }

//     ParMixedBilinearForm AU_varf(RTspace, NDspace);
//     ScalarVectorProductCoefficient halfB1_coeff(0.5, B1_gf_coeff);
//     AU_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(halfB1_coeff));

//     ParBilinearForm BB_varf(RTspace);
//     BB_varf.AddDomainIntegrator(new VectorFEMassIntegrator);

//     ParMixedBilinearForm BA_varf(NDspace, RTspace);
//     BA_varf.AddDomainIntegrator(new MixedVectorCurlIntegrator(m_one));

//     ParBilinearForm JJ_varf(NDspace);
//     JJ_varf.AddDomainIntegrator(new VectorFEMassIntegrator);

//     ParMixedBilinearForm JB_varf(RTspace, NDspace);
//     JB_varf.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(m_one));

//     // linear forms
//     // linear form for u
//     VectorFunctionCoefficient fucoeff(dim, pd->fu_fun);
//     fucoeff.SetTime(t_int + 0.5 * dt);
//     ParLinearForm fu_form(RTspace);
//     fu_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fucoeff));

//     ParLinearForm fu_p_form(RTspace);
//     if (ess_bdr_tangent.Size())
//         fu_p_form.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(pbdrycoeff), ess_bdr_tangent);

//     // linear form for w
//     ubdrycoeff.SetTime(t_int + dt);
//     ParLinearForm fw_form(NDspace);
//     if (ess_bdr_tangent.Size())
//         fw_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(ubdrycoeff), ess_bdr_tangent);

//     // linear form for p
//     ParLinearForm fp_form(L2space);

//     // linear form for A
//     ParLinearForm fA_form(NDspace);
//     VectorFunctionCoefficient fAcoeff(dim, pd->fA_fun);
//     fAcoeff.SetTime(t_int + 0.5 * dt);
//     fA_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fAcoeff));

//     // linear form for B
//     ParLinearForm fB_form(RTspace);
//     VectorFunctionCoefficient Bstabcoeff(dim, pd->Bstab_fun);
//     if (pd->periodic)
//     {
//         fB_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Bstabcoeff));
//     }

//     // linear form for j
//     ParLinearForm fJ_form(NDspace);
//     Bbdrycoeff.SetTime(t_int + dt);
//     if (ess_bdr_magnetic.Size())
//         fJ_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(Bbdrycoeff), ess_bdr_magnetic);

//     // adaptive mesh refinement
//     DifferenceLpEstimator *estimator = nullptr;
//     CustomRefiner *refiner = nullptr;
//     ThresholdDerefiner *derefiner = nullptr;
//     if (amr)
//     {
//         BilinearFormIntegrator *CurlCurlInteg = new CurlCurlIntegrator();
//         // estimator = new L2ZienkiewiczZhuEstimator(*CurlCurlInteg, *A1_gf, *RTspace, *NDspace);
//         estimator = new DifferenceLpEstimator(2, pmesh);
//         estimator->RegisterVariable(u2mid_gf, *u1_gf);
//         estimator->RegisterVariable(B2mid_gf, *B1_gf);
//         refiner = new CustomRefiner(*estimator);
//         refiner->SetTotalErrorFraction(refine_frac);
//         refiner->SetMaxElements(max_elements);
//         refiner->SetTotalErrorGoal(total_err_goal);
//         refiner->PreferConformingRefinement();

//         derefiner = new ThresholdDerefiner(*estimator);
//     }

//     for (int ref_it = 0; ref_it < (amr ? max_amr_iter : 1); ref_it++)
//     {

//         if (amr)
//             mfemPrintf("AMR iteration: %d\n", ref_it);

//         /* ASSEMBLE */
//         timer.Restart();

//         // assemble bilinear forms
//         UU_varf.Assemble();
//         UU_old_varf.Assemble();
//         UJ_varf.Assemble();
//         UW_varf.Assemble();
//         WW_varf.Assemble();
//         WU_varf.Assemble();
//         if (!viscosity)
//             WU_bdry_varf.Assemble();
//         PU_varf.Assemble();
//         UP_varf.Assemble();
//         AA_varf.Assemble();
//         AJ_varf.Assemble();
//         AU_varf.Assemble();
//         BB_varf.Assemble();
//         BA_varf.Assemble();
//         JJ_varf.Assemble();
//         JB_varf.Assemble();

//         // assemble linear forms
//         fu_form.Assemble();
//         UU_old_varf.AddMult(u2_old, fu_form, 1.0);
//         UW_varf.AddMult(w1_old, fu_form, -1.0);
//         UJ_varf.AddMult(j1_old, fu_form, -1.0);
//         fu_p_form.Assemble();
//         fu_form -= fu_p_form;

//         fw_form.Assemble();
//         fp_form.Assemble();

//         fA_form.Assemble();
//         AA_varf.AddMult(A1_old, fA_form, 1.0);
//         AJ_varf.AddMult(j1_old, fA_form, -1.0);
//         AU_varf.AddMult(u2_old, fA_form, -1.0);

//         fB_form.Assemble();
//         fJ_form.Assemble();

//         // apply boundary conditions
//         if (ess_bdr_normal.Size())
//         {
//             ubdrycoeff.SetTime(t_int + dt);
//             wbdrycoeff.SetTime(t_int + dt);
//             *u2_gf = 0.0;
//             u2_gf->ProjectBdrCoefficientNormal(ubdrycoeff, ess_bdr_normal);
//             UW_varf.EliminateTestDofs(ess_bdr_normal);
//             WU_varf.EliminateTrialDofs(ess_bdr_normal, *u2_gf, fw_form);
//             if (!viscosity)
//             {
//                 WU_bdry_varf.AddMultTranspose(*u2_gf, fw_form, -1.0);
//                 WU_bdry_varf.EliminateTestDofs(ess_bdr_normal);
//             }
//             UP_varf.EliminateTestDofs(ess_bdr_normal);
//             PU_varf.EliminateTrialDofs(ess_bdr_normal, *u2_gf, fp_form);
//             UJ_varf.EliminateTestDofs(ess_bdr_normal);
//             AU_varf.EliminateTrialDofs(ess_bdr_normal, *u2_gf, fA_form);
//             UU_varf.EliminateEssentialBC(ess_bdr_normal, *u2_gf, fu_form);

//             if (viscosity)
//             {
//                 w1_gf->ProjectBdrCoefficientTangent(wbdrycoeff, ess_bdr_normal);
//                 WU_varf.EliminateTestDofs(ess_bdr_normal);
//                 UW_varf.EliminateTrialDofs(ess_bdr_normal, *w1_gf, fu_form);
//                 WW_varf.EliminateEssentialBC(ess_bdr_normal, *w1_gf, fw_form);
//             }
//         }

//         // Assemble Matrices
//         UU_varf.Finalize();
//         HypreParMatrix *UUmat = UU_varf.ParallelAssemble();

//         UJ_varf.Finalize();
//         HypreParMatrix *UJmat = UJ_varf.ParallelAssemble();

//         WU_varf.Finalize();
//         HypreParMatrix *WUmat0 = WU_varf.ParallelAssemble();

//         HypreParMatrix *WUmat;
//         if (!viscosity)
//         {
//             WU_bdry_varf.Finalize();
//             HypreParMatrix *WU_bdry_mat = WU_bdry_varf.ParallelAssemble();
//             HypreParMatrix *WU_bdry_mat_T = WU_bdry_mat->Transpose();
//             WUmat = ParAdd(WUmat0, WU_bdry_mat_T);
//             delete WU_bdry_mat;
//             delete WU_bdry_mat_T;
//             delete WUmat0;
//         }
//         else
//         {
//             WUmat = WUmat0;
//         }

//         UW_varf.Finalize();
//         HypreParMatrix *UWmat = UW_varf.ParallelAssemble();

//         PU_varf.Finalize();
//         HypreParMatrix *PUmat = PU_varf.ParallelAssemble();

//         UP_varf.Finalize();
//         HypreParMatrix *UPmat = UP_varf.ParallelAssemble();

//         WW_varf.Finalize();
//         HypreParMatrix *WWmat = WW_varf.ParallelAssemble();

//         AA_varf.Finalize();
//         HypreParMatrix *AAmat = AA_varf.ParallelAssemble();

//         AJ_varf.Finalize();
//         HypreParMatrix *AJmat = AJ_varf.ParallelAssemble();

//         AU_varf.Finalize();
//         HypreParMatrix *AUmat = AU_varf.ParallelAssemble();

//         BB_varf.Finalize();
//         HypreParMatrix *BBmat = BB_varf.ParallelAssemble();

//         BA_varf.Finalize();
//         HypreParMatrix *BAmat = BA_varf.ParallelAssemble();

//         JJ_varf.Finalize();
//         HypreParMatrix *JJmat = JJ_varf.ParallelAssemble();

//         JB_varf.Finalize();
//         HypreParMatrix *JBmat = JB_varf.ParallelAssemble();

//         // get trueoffsets
//         Array<int> block_trueOffsets(7);
//         block_trueOffsets[0] = 0;
//         block_trueOffsets[1] = RTspace->GetTrueVSize();
//         block_trueOffsets[2] = NDspace->GetTrueVSize();
//         block_trueOffsets[3] = L2space->GetTrueVSize();
//         block_trueOffsets[4] = NDspace->GetTrueVSize();
//         block_trueOffsets[5] = RTspace->GetTrueVSize();
//         block_trueOffsets[6] = NDspace->GetTrueVSize();
//         block_trueOffsets.PartialSum();

//         BlockOperator BigOp(block_trueOffsets);
//         BigOp.SetBlock(0, 0, UUmat);
//         BigOp.SetBlock(0, 1, UWmat);
//         BigOp.SetBlock(0, 2, UPmat);
//         BigOp.SetBlock(0, 5, UJmat);
//         BigOp.SetBlock(1, 0, WUmat);
//         BigOp.SetBlock(1, 1, WWmat);
//         BigOp.SetBlock(2, 0, PUmat);
//         BigOp.SetBlock(3, 0, AUmat);
//         BigOp.SetBlock(3, 3, AAmat);
//         BigOp.SetBlock(3, 5, AJmat);
//         BigOp.SetBlock(4, 4, BBmat);
//         BigOp.SetBlock(4, 3, BAmat);
//         BigOp.SetBlock(5, 5, JJmat);
//         BigOp.SetBlock(5, 4, JBmat);

//         BlockVector Bigvec(block_trueOffsets), solvec(block_trueOffsets);
//         Bigvec = 0.0;
//         solvec = 0.0;

//         fu_form.ParallelAssemble(Bigvec.GetBlock(0));
//         fw_form.ParallelAssemble(Bigvec.GetBlock(1));
//         fp_form.ParallelAssemble(Bigvec.GetBlock(2));
//         fA_form.ParallelAssemble(Bigvec.GetBlock(3));
//         fB_form.ParallelAssemble(Bigvec.GetBlock(4));
//         fJ_form.ParallelAssemble(Bigvec.GetBlock(5));

//         timer.Stop();
//         mfemPrintf("Assemble time: %lg \n", timer.RealTime());

//         /* SOLVE */
//         timer.Restart();
//         pc_integer prec(NDspace, RTspace, L2space, &BigOp, block_trueOffsets, gamma_int, dt, pd->param.Re, pd->param.Rm, pd->param.RH, 1e-5, 300, use_petsc_integer, Hall, viscosity, resistivity, false);
//         prec.PrintInfo(1);

//         FGMRESSolver solver(MPI_COMM_WORLD);
//         solver.SetAbsTol(atol);
//         solver.SetRelTol(rtol);
//         solver.SetMaxIter(maxit);
//         solver.SetOperator(BigOp);
//         solver.SetPrintLevel(3);
//         solver.SetPreconditioner(prec);
//         solver.iterative_mode = true;

//         // initial guess
//         u2_old.GetTrueDofs(solvec.GetBlock(0));
//         w1_old.GetTrueDofs(solvec.GetBlock(1));
//         A1_old.GetTrueDofs(solvec.GetBlock(3));
//         B2_old.GetTrueDofs(solvec.GetBlock(4));
//         j1_old.GetTrueDofs(solvec.GetBlock(5));

//         timer.Stop();
//         mfemPrintf("Setup time: %lg \n", timer.RealTime());

//         timer.Restart();
//         solver.Mult(Bigvec, solvec);
//         timer.Stop();

//         mfemPrintf("solver: %d its, res: %lg, time: %lg \n", solver.GetNumIterations(), solver.GetFinalRelNorm(), timer.RealTime());

//         u2_gf->Distribute(solvec.GetBlock(0));
//         w1_gf->Distribute(solvec.GetBlock(1));
//         p3_gf->Distribute(solvec.GetBlock(2));
//         A1_gf->Distribute(solvec.GetBlock(3));
//         B2_gf->Distribute(solvec.GetBlock(4));
//         j1_gf->Distribute(solvec.GetBlock(5));

//         Zeromean(*p3_gf);
//         CheckDivergenceFree(u2_gf);

//         // update mid point values
//         u2mid_gf = *u2_gf;
//         u2mid_gf += u2_old;
//         u2mid_gf *= 0.5;

//         B2mid_gf = *B2_gf;
//         B2mid_gf += B2_old;
//         B2mid_gf *= 0.5;

//         // free memory
//         delete UUmat;
//         delete UJmat;
//         delete WWmat;
//         delete WUmat;
//         delete UWmat;
//         delete PUmat;
//         delete UPmat;
//         delete AAmat;
//         delete AUmat;
//         delete AJmat;
//         delete BBmat;
//         delete BAmat;
//         delete JJmat;
//         delete JBmat;

//         if (amr)
//         {
//             // refine mesh
//             refiner->Apply(*pmesh);
//             HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
//             mfemPrintf("Refinement: total error = %e, marked element: %lld, total element: %d \n", refiner->GetTotalError(), refiner->GetNumMarkedElements(), glob_el);

//             if (refiner->Stop())
//             {
//                 mfemPrintf("Refinement stopped\n");
//                 break;
//             }

//             // update and rebalance
//             // update the fe spaces
//             UpdateAllFEspaces();
//             // update all useful grid functions
//             UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});

//             mfemPrintf("Rebalancing mesh\n");
//             // rebalance mesh
//             if (pmesh->Nonconforming())
//             {
//                 pmesh->Rebalance();
//                 // update the fe spaces and grid functions again
//                 UpdateAllFEspaces();
//                 UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});
//             }

//             // update the bilinear and linear forms
//             UpdateForms({&UU_varf, &UU_old_varf, &WW_varf, &AA_varf, &AJ_varf, &BB_varf, &JJ_varf},
//                         {&UJ_varf, &UW_varf, &WU_varf, &PU_varf, &UP_varf, &AU_varf, &BA_varf, &JB_varf},
//                         {&fu_form, &fu_p_form, &fw_form, &fp_form, &fA_form, &fB_form, &fJ_form});
//             UpdateFinishedFEspaces();

//             derefiner->SetThreshold(coarse_frac * refiner->GetTotalError());

//             // derefine mesh
//             if (derefiner->Apply(*pmesh))
//             {
//                 HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
//                 mfemPrintf("Derefined elements. Total Elements: %d .\n", glob_el);

//                 // update the fe spaces
//                 UpdateAllFEspaces();
//                 // update all useful grid functions
//                 UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});

//                 // rebalance mesh
//                 mfemPrintf("Rebalancing mesh\n");
//                 if (pmesh->Nonconforming())
//                 {
//                     pmesh->Rebalance();
//                     // update the fe spaces and grid functions again
//                     UpdateAllFEspaces();
//                     UpdateAllSolutions({&u2_old, &w1_old, &A1_old, &B2_old, &j1_old, &u2mid_gf, &B2mid_gf});
//                 }

//                 UpdateForms({&UU_varf, &UU_old_varf, &WW_varf, &AA_varf, &AJ_varf, &BB_varf, &JJ_varf},
//                             {&UJ_varf, &UW_varf, &WU_varf, &PU_varf, &UP_varf, &AU_varf, &BA_varf, &JB_varf},
//                             {&fu_form, &fu_p_form, &fw_form, &fp_form, &fA_form, &fB_form, &fJ_form});
//                 UpdateFinishedFEspaces();
//             }
//         }
//     }

//     t_int += dt;

//     if (amr)
//     {
//         pmesh->PrintInfo();
//         delete estimator;
//         delete refiner;
//         delete derefiner;
//     }

//     return 0;
// }

// half integer time steps
// from t_half to t_half+dt, using theta scheme (theta = 1, 0.5)
// int MHD_solver::HalfStep(real_t &t_half, real_t dt, real_t theta)
// {
//     mfemPrintf("-----------------------------------------------------------\n");
//     mfemPrintf("(Half time step) Advancing from t = %lg to t = %lg, theta = %lg \n",
//                t_half, t_half + dt, theta);
//     mfemPrintf("-----------------------------------------------------------\n");

//     StopWatch timer;
//     timer.Restart();

//     /* save old data */
//     ParGridFunction u1_old(*gfs.u1_gf);
//     ParGridFunction w2_old(*gfs.w2_gf);
//     ParGridFunction p0_old(*gfs.p0_gf);
//     ParGridFunction A2_old(*gfs.A2_gf);
//     ParGridFunction B1_old(*gfs.B1_gf);
//     ParGridFunction j2_old(*gfs.j2_gf);

//     ParGridFunction u1mid_gf(sol_info.NDspace);
//     ParGridFunction B1mid_gf(sol_info.NDspace);

//     // Coefficients
//     VectorFunctionCoefficient ubdrycoeff(sol_info.dim, pd->ubdry_fun);
//     VectorFunctionCoefficient wbdrycoeff(sol_info.dim, pd->wbdry_fun);
//     FunctionCoefficient pbdrycoeff(pd->pbdry_fun);
//     VectorFunctionCoefficient Bbdrycoeff(sol_info.dim, pd->Bbdry_fun);
//     ConstantCoefficient oneondt(1.0 / dt);
//     ConstantCoefficient m_one(-1.0);

//     // Bilinear forms
//     ParBilinearForm UU_varf(sol_info.NDspace);
//     VectorGridFunctionCoefficient w1_gf_coeff(gfs.w1_gf);
//     ScalarVectorProductCoefficient thetaw1_coeff(theta, w1_gf_coeff);
//     UU_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));
//     UU_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(thetaw1_coeff));

//     ParBilinearForm UU_old_varf(sol_info.NDspace);
//     ScalarVectorProductCoefficient thetam1_w1_coeff(theta - 1.0, w1_gf_coeff);
//     UU_old_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));
//     UU_old_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(thetam1_w1_coeff));

//     ConstantCoefficient thetaOnRe;
//     ConstantCoefficient thetam1OnRe;
//     if (sol_info.viscosity)
//     {
//         thetaOnRe.constant = theta / pd->param.Re;
//         thetam1OnRe.constant = (theta - 1.0) / pd->param.Re;
//         UU_varf.AddDomainIntegrator(new CurlCurlIntegrator(thetaOnRe));
//         UU_old_varf.AddDomainIntegrator(new CurlCurlIntegrator(thetam1OnRe));
//     }

//     ParMixedBilinearForm PU_varf(sol_info.NDspace, sol_info.H1space);
//     PU_varf.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(m_one));

//     ParMixedBilinearForm UP_varf(sol_info.H1space, sol_info.NDspace);
//     UP_varf.AddDomainIntegrator(new MixedVectorGradientIntegrator);

//     ParBilinearForm PP_varf(sol_info.H1space);
//     ConstantCoefficient epsilon_coeff(0.001);
//     PP_varf.AddDomainIntegrator(new MassIntegrator(epsilon_coeff));

//     ParMixedBilinearForm UJ_varf(sol_info.RTspace, sol_info.NDspace);
//     VectorGridFunctionCoefficient B2_gf_coeff(gfs.B2_gf);
//     ScalarVectorProductCoefficient thetasB2_coeff(theta * pd->param.s, B2_gf_coeff);
//     UJ_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(thetasB2_coeff));

//     ParMixedBilinearForm UJ_old_varf(sol_info.RTspace, sol_info.NDspace);
//     ScalarVectorProductCoefficient thetam1sB2_coeff((theta - 1.0) * pd->param.s, B2_gf_coeff);
//     UJ_old_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(thetam1sB2_coeff));

//     ParBilinearForm AA_varf(sol_info.RTspace);
//     AA_varf.AddDomainIntegrator(new VectorFEMassIntegrator(oneondt));

//     ParBilinearForm AJ_varf(sol_info.RTspace);
//     ConstantCoefficient thetaOnRm;
//     if (sol_info.resistivity)
//     {
//         thetaOnRm.constant = theta / pd->param.Rm;
//         AJ_varf.AddDomainIntegrator(new VectorFEMassIntegrator(thetaOnRm));
//     }
//     ScalarVectorProductCoefficient Hall_coeff(-theta * pd->param.RH, B2_gf_coeff);
//     if (sol_info.Hall)
//         AJ_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(Hall_coeff));

//     ParBilinearForm AJ_old_varf(sol_info.RTspace);
//     ConstantCoefficient thetam1OnRm;
//     if (sol_info.resistivity)
//     {
//         thetam1OnRm.constant = (theta - 1.0) / pd->param.Rm;
//         AJ_old_varf.AddDomainIntegrator(new VectorFEMassIntegrator(thetam1OnRm));
//     }
//     ScalarVectorProductCoefficient Hall_old_coeff(-(theta - 1.0) * pd->param.RH, B2_gf_coeff);
//     if (sol_info.Hall)
//         AJ_old_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(Hall_old_coeff));

//     ParMixedBilinearForm AU_varf(sol_info.NDspace, sol_info.RTspace);
//     ScalarVectorProductCoefficient thetaB2_coeff(theta, B2_gf_coeff);
//     AU_varf.AddDomainIntegrator(new MixedCrossProductIntegrator(thetaB2_coeff));

//     ParBilinearForm BB_varf(sol_info.NDspace);
//     BB_varf.AddDomainIntegrator(new VectorFEMassIntegrator);

//     ParMixedBilinearForm BA_varf(sol_info.RTspace, sol_info.NDspace);
//     BA_varf.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(m_one));

//     ParBilinearForm JJ_varf(sol_info.RTspace);
//     JJ_varf.AddDomainIntegrator(new VectorFEMassIntegrator);

//     ParMixedBilinearForm JB_varf(sol_info.NDspace, sol_info.RTspace);
//     JB_varf.AddDomainIntegrator(new MixedVectorCurlIntegrator(m_one));

//     // linear forms
//     VectorFunctionCoefficient fucoeff(sol_info.dim, pd->fu_fun);
//     ParLinearForm fu_form(sol_info.NDspace);
//     fu_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fucoeff));

//     ParLinearForm fu_wbdry_form(sol_info.NDspace);
//     if (pd->ess_bdr_normal.Size())
//     {
//         fu_wbdry_form.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(wbdrycoeff), pd->ess_bdr_normal);
//     }

//     ParLinearForm fp_form(sol_info.H1space);
//     if (pd->ess_bdr_normal.Size())
//     {
//         fp_form.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(ubdrycoeff), pd->ess_bdr_normal);
//     }

//     ParLinearForm fA_form(sol_info.RTspace);
//     VectorFunctionCoefficient fAcoeff(sol_info.dim, pd->fA_fun);
//     fA_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fAcoeff));

//     ParLinearForm fB_form(sol_info.NDspace);
//     VectorFunctionCoefficient Bstabcoeff(sol_info.dim, pd->Bstab_fun);
//     if (pd->periodic)
//     {
//         fB_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Bstabcoeff));
//     }

//     ParLinearForm fJ_form(sol_info.RTspace);

//     // adaptive mesh refinement
//     BilinearFormIntegrator *CurlCurlInteg = nullptr;
//     ErrorEstimator *estimator = nullptr;
//     CustomRefiner *refiner = nullptr;
//     ThresholdDerefiner *derefiner = nullptr;
//     if (amr_info.amr)
//     {
//         CurlCurlInteg = new CurlCurlIntegrator();
//         if (theta > 1.0 - 1e-12)
//             estimator = new L2ZienkiewiczZhuEstimator(*CurlCurlInteg, *gfs.B1_gf, *sol_info.RTspace, *sol_info.NDspace);
//         else
//         {
//             estimator = new DifferenceLpEstimator(2, pmesh);
//             DifferenceLpEstimator *estimator_diff = dynamic_cast<DifferenceLpEstimator *>(estimator);
//             estimator_diff->RegisterVariable(u1mid_gf, *gfs.u2_gf);
//             estimator_diff->RegisterVariable(B1mid_gf, *gfs.B2_gf);
//         }

//         refiner = new CustomRefiner(*estimator);
//         refiner->SetTotalErrorFraction(amr_info.refine_frac);
//         refiner->SetMaxElements(amr_info.max_elements);
//         refiner->SetTotalErrorGoal(amr_info.total_err_goal);
//         refiner->PreferConformingRefinement();
//         derefiner = new ThresholdDerefiner(*estimator);
//     }

//     for (int ref_it = 0; ref_it < (amr_info.amr ? amr_info.max_amr_iter : 1); ref_it++)
//     {
//         if (amr_info.amr)
//             mfemPrintf("AMR iteration: %d\n", ref_it);

//         /* ASSEMBLE */
//         timer.Restart();

//         // assemble bilinear forms
//         UU_varf.Assemble();
//         UU_old_varf.Assemble();
//         PU_varf.Assemble();
//         UP_varf.Assemble();
//         PP_varf.Assemble();
//         UJ_varf.Assemble();
//         UJ_old_varf.Assemble();
//         AA_varf.Assemble();
//         AJ_varf.Assemble();
//         AJ_old_varf.Assemble();
//         AU_varf.Assemble();
//         BB_varf.Assemble();
//         BA_varf.Assemble();
//         JJ_varf.Assemble();
//         JB_varf.Assemble();

//         // assemble linear forms
//         fucoeff.SetTime(t_half + theta * dt);
//         fu_form.Assemble();

//         if (sol_info.viscosity)
//         {
//             wbdrycoeff.SetTime(t_half + theta * dt);
//             fu_wbdry_form.Assemble();
//             fu_wbdry_form *= -1.0 / pd->param.Re;
//             fu_form += fu_wbdry_form;
//         }

//         UU_old_varf.AddMult(u1_old, fu_form, 1.0);
//         UJ_old_varf.AddMult(j2_old, fu_form, 1.0);

//         ubdrycoeff.SetTime(t_half + dt);
//         fp_form.Assemble();

//         fAcoeff.SetTime(t_half + theta * dt);
//         fA_form.Assemble();
//         AA_varf.AddMult(A2_old, fA_form, 1.0);
//         AJ_old_varf.AddMult(j2_old, fA_form, 1.0);
//         AU_varf.AddMult(u1_old, fA_form, -(1.0 - theta) / theta);

//         fB_form.Assemble();
//         fJ_form.Assemble();

//         // boundary conditions
//         // Dirichlet BC
//         if (pd->ess_bdr_tangent.Size())
//         {
//             ubdrycoeff.SetTime(t_half + dt);
//             pbdrycoeff.SetTime(t_half + dt);
//             gfs.u1_gf->ProjectBdrCoefficientTangent(ubdrycoeff, pd->ess_bdr_tangent);
//             gfs.p0_gf->ProjectBdrCoefficient(pbdrycoeff, pd->ess_bdr_tangent);
//             PU_varf.EliminateTrialDofs(pd->ess_bdr_tangent, *gfs.u1_gf, fp_form);
//             AU_varf.EliminateTrialDofs(pd->ess_bdr_tangent, *gfs.u1_gf, fA_form);
//             UP_varf.EliminateTrialDofs(pd->ess_bdr_tangent, *gfs.p0_gf, fu_form);
//             UP_varf.EliminateTestDofs(pd->ess_bdr_tangent);
//             UJ_varf.EliminateTestDofs(pd->ess_bdr_tangent);
//             PU_varf.EliminateTestDofs(pd->ess_bdr_tangent);
//             UU_varf.EliminateEssentialBC(pd->ess_bdr_tangent, *gfs.u1_gf, fu_form);
//             PP_varf.EliminateEssentialBC(pd->ess_bdr_tangent, *gfs.p0_gf, fp_form);
//         }

//         if (pd->ess_bdr_magnetic.Size())
//         {
//             Bbdrycoeff.SetTime(t_half + dt);
//             gfs.B1_gf->ProjectBdrCoefficientTangent(Bbdrycoeff, pd->ess_bdr_magnetic);
//             BB_varf.EliminateEssentialBC(pd->ess_bdr_magnetic, *gfs.B1_gf, fB_form);
//             BA_varf.EliminateTestDofs(pd->ess_bdr_magnetic);
//             JB_varf.EliminateTrialDofs(pd->ess_bdr_magnetic, *gfs.B1_gf, fJ_form);
//         }

//         // Assemble matrices
//         PU_varf.Finalize();
//         UP_varf.Finalize();
//         UU_varf.Finalize();
//         UJ_varf.Finalize();
//         AA_varf.Finalize();
//         AJ_varf.Finalize();
//         AU_varf.Finalize();
//         BB_varf.Finalize();
//         BA_varf.Finalize();
//         JJ_varf.Finalize();
//         JB_varf.Finalize();

//         HypreParMatrix *UUmat = UU_varf.ParallelAssemble();
//         HypreParMatrix *PUmat = PU_varf.ParallelAssemble();
//         HypreParMatrix *UPmat = UP_varf.ParallelAssemble();
//         HypreParMatrix *UJmat = UJ_varf.ParallelAssemble();
//         HypreParMatrix *AAmat = AA_varf.ParallelAssemble();
//         HypreParMatrix *AJmat = AJ_varf.ParallelAssemble();
//         HypreParMatrix *AUmat = AU_varf.ParallelAssemble();
//         HypreParMatrix *BBmat = BB_varf.ParallelAssemble();
//         HypreParMatrix *BAmat = BA_varf.ParallelAssemble();
//         HypreParMatrix *JJmat = JJ_varf.ParallelAssemble();
//         HypreParMatrix *JBmat = JB_varf.ParallelAssemble();

//         // offsets
//         Array<int> block_trueOffsets(6);
//         block_trueOffsets[0] = 0;
//         block_trueOffsets[1] = sol_info.NDspace->GetTrueVSize();
//         block_trueOffsets[2] = sol_info.H1space->GetTrueVSize();
//         block_trueOffsets[3] = sol_info.RTspace->GetTrueVSize();
//         block_trueOffsets[4] = sol_info.RTspace->GetTrueVSize();
//         block_trueOffsets[5] = sol_info.NDspace->GetTrueVSize();
//         block_trueOffsets.PartialSum();

//         // set block matrix
//         BlockOperator BigOp(block_trueOffsets);
//         BigOp.SetBlock(0, 0, UUmat);
//         BigOp.SetBlock(0, 1, UPmat);
//         BigOp.SetBlock(0, 3, UJmat);
//         BigOp.SetBlock(1, 0, PUmat);
//         BigOp.SetBlock(2, 2, AAmat);
//         BigOp.SetBlock(2, 3, AJmat);
//         BigOp.SetBlock(2, 0, AUmat);
//         BigOp.SetBlock(4, 4, BBmat);
//         BigOp.SetBlock(4, 2, BAmat);
//         BigOp.SetBlock(3, 3, JJmat);
//         BigOp.SetBlock(3, 4, JBmat);

//         BlockVector Bigvec(block_trueOffsets), solvec(block_trueOffsets);
//         Bigvec = 0.0;
//         solvec = 0.0;

//         fu_form.ParallelAssemble(Bigvec.GetBlock(0));
//         fp_form.ParallelAssemble(Bigvec.GetBlock(1));
//         fA_form.ParallelAssemble(Bigvec.GetBlock(2));
//         fJ_form.ParallelAssemble(Bigvec.GetBlock(3));
//         fB_form.ParallelAssemble(Bigvec.GetBlock(4));

//         timer.Stop();
//         mfemPrintf("Assemble time: %lg\n", timer.RealTime());

//         /* SOLVE */
//         timer.Restart();

//         // preconditioner
//         pc_half prec(sol_info.NDspace, sol_info.H1space, &BigOp, block_trueOffsets, pd->ess_bdr_tangent, pd->ess_bdr_normal, pd->ess_bdr_magnetic, gfs.B2_gf, lin_sol_half.gamma, dt, pd->param.Re, pd->param.Rm, pd->param.RH, theta, 1e-5, 100, false, sol_info.Hall, sol_info.viscosity, sol_info.resistivity, false);
//         prec.PrintInfo(1);

//         FGMRESSolver solver(MPI_COMM_WORLD);
//         solver.SetAbsTol(lin_sol_half.atol);
//         solver.SetRelTol(lin_sol_half.rtol);
//         solver.SetMaxIter(lin_sol_half.maxit);
//         solver.SetPreconditioner(prec);
//         solver.SetOperator(BigOp);
//         solver.SetPrintLevel(3);

//         solvec = 0.0;
//         u1_old.GetTrueDofs(solvec.GetBlock(0));
//         p0_old.GetTrueDofs(solvec.GetBlock(1));
//         A2_old.GetTrueDofs(solvec.GetBlock(2));
//         j2_old.GetTrueDofs(solvec.GetBlock(3));
//         B1_old.GetTrueDofs(solvec.GetBlock(4));

//         timer.Stop();
//         mfemPrintf("Solver setup time: %lg\n", timer.RealTime());

//         timer.Restart();
//         solver.Mult(Bigvec, solvec);
//         timer.Stop();

//         mfemPrintf("solver: %d its, res: %lg, time: %lg \n", solver.GetNumIterations(), solver.GetFinalRelNorm(), timer.RealTime());

//         gfs.u1_gf->Distribute(solvec.GetBlock(0));
//         gfs.p0_gf->Distribute(solvec.GetBlock(1));
//         gfs.A2_gf->Distribute(solvec.GetBlock(2));
//         gfs.j2_gf->Distribute(solvec.GetBlock(3));
//         gfs.B1_gf->Distribute(solvec.GetBlock(4));

//         strong_curl(*gfs.u1_gf, *gfs.w2_gf);

//         // modify p to have zero mean
//         if (pd->ess_bdr_tangent.Max() == 0)
//             Zeromean(*gfs.p0_gf);

//         // update mid point values
//         u1mid_gf = *gfs.u1_gf;
//         u1mid_gf += u1_old;
//         u1mid_gf *= 0.5;

//         B1mid_gf = *gfs.B1_gf;
//         B1mid_gf += B1_old;
//         B1mid_gf *= 0.5;

//         // free memory
//         delete UUmat;
//         delete PUmat;
//         delete UPmat;
//         delete UJmat;
//         delete AAmat;
//         delete AJmat;
//         delete AUmat;
//         delete BBmat;
//         delete BAmat;
//         delete JJmat;
//         delete JBmat;

//         if (amr_info.amr)
//         {
//             // refine mesh
//             refiner->Apply(*pmesh);
//             HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
//             mfemPrintf("Refinement: total error = %e, marked element: %lld, total element: %d \n", refiner->GetTotalError(), refiner->GetNumMarkedElements(), glob_el);

//             if (refiner->Stop())
//             {
//                 mfemPrintf("Refinement stopped\n");
//                 break;
//             }

//             // update and rebalance
//             // update the fe spaces
//             UpdateAllFEspaces();
//             // update all useful grid functions
//             UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});

//             // TODO:rebalance here
//             if (pmesh->Nonconforming())
//             {
//                 pmesh->Rebalance();
//                 // update the fe spaces and grid functions again
//                 UpdateAllFEspaces();
//                 UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});
//             }

//             UpdateForms(
//                 {&UU_varf, &UU_old_varf, &PP_varf, &AA_varf, &AJ_varf, &AJ_old_varf, &BB_varf, &JJ_varf},
//                 {&PU_varf, &UP_varf, &UJ_varf, &UJ_old_varf, &AU_varf, &BA_varf, &JB_varf},
//                 {&fu_form, &fu_wbdry_form, &fp_form, &fA_form, &fB_form, &fJ_form});

//             UpdateFinishedFEspaces();

//             derefiner->SetThreshold(amr_info.coarse_frac * refiner->GetTotalError());

//             // derefine mesh
//             if (derefiner->Apply(*pmesh))
//             {
//                 HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
//                 mfemPrintf("Derefined elements. Total elements: %d .\n", glob_el);

//                 // update the fe spaces
//                 UpdateAllFEspaces();
//                 // update all useful grid functions
//                 UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});

//                 // TODO: rebalance here

//                 UpdateForms(
//                     {&UU_varf, &UU_old_varf, &PP_varf, &AA_varf, &AJ_varf, &AJ_old_varf, &BB_varf, &JJ_varf},
//                     {&PU_varf, &UP_varf, &UJ_varf, &UJ_old_varf, &AU_varf, &BA_varf, &JB_varf},
//                     {&fu_form, &fu_wbdry_form, &fp_form, &fA_form, &fB_form, &fJ_form});

//                 UpdateFinishedFEspaces();
//             }
//         }
//     }

//     t_half += dt;

//     if (amr_info.amr)
//     {
//         pmesh->PrintInfo();
//         delete CurlCurlInteg;
//         delete estimator;
//         delete refiner;
//         delete derefiner;
//     }

//     return 0;
// }

int MHD_solver::HalfStep(real_t &t_half, real_t dt, real_t theta, bool update, bool first_step)
{
    mfemPrintf("-----------------------------------------------------------\n");
    mfemPrintf("(Half time step) Advancing from t = %lg to t = %lg, theta = %lg \n",
               t_half, t_half + dt, theta);
    mfemPrintf("-----------------------------------------------------------\n");

    StopWatch timer;
    timer.Restart();
    
    sol_info.dt = dt;

    /* save old data */
    *old_gfs.u1_gf=*gfs.u1_gf;
    *old_gfs.w2_gf=*gfs.w2_gf;
    *old_gfs.p0_gf=*gfs.p0_gf;
    *old_gfs.A2_gf=*gfs.A2_gf;
    *old_gfs.B1_gf=*gfs.B1_gf;
    *old_gfs.j2_gf=*gfs.j2_gf;

    ParGridFunction u1mid_gf(sol_info.NDspace);
    ParGridFunction B1mid_gf(sol_info.NDspace);

    // adaptive mesh refinement
    BilinearFormIntegrator *CurlCurlInteg = nullptr;
    ErrorEstimator *estimator = nullptr;
    CustomRefiner *refiner = nullptr;
    ThresholdDerefiner *derefiner = nullptr;
    if (amr_info.amr)
    {
        CurlCurlInteg = new CurlCurlIntegrator();
        if (theta > 1.0 - 1e-12)
            estimator = new L2ZienkiewiczZhuEstimator(*CurlCurlInteg, *gfs.B1_gf, *sol_info.RTspace, *sol_info.NDspace);
        else
        {
            estimator = new DifferenceLpEstimator(2, pmesh);
            DifferenceLpEstimator *estimator_diff = dynamic_cast<DifferenceLpEstimator *>(estimator);
            estimator_diff->RegisterVariable(u1mid_gf, *gfs.u2_gf);
            estimator_diff->RegisterVariable(B1mid_gf, *gfs.B2_gf);
        }

        refiner = new CustomRefiner(*estimator);
        refiner->SetTotalErrorFraction(amr_info.refine_frac);
        refiner->SetMaxElements(amr_info.max_elements);
        refiner->SetTotalErrorGoal(amr_info.total_err_goal);
        refiner->PreferConformingRefinement();
        derefiner = new ThresholdDerefiner(*estimator);
    }

    for (int ref_it = 0; ref_it < (amr_info.amr ? amr_info.max_amr_iter : 1); ref_it++)
    {
        if (amr_info.amr)
            mfemPrintf("AMR iteration: %d\n", ref_it);

        /* ASSEMBLE */
        timer.Restart();

        if(update)
        {
            half_evo->AssembleOperators(theta, dt);
        }
        else
        {
            half_evo->AssembleTDOperators(theta, dt);
        }

        half_evo->Step(t_half, dt, theta);

        // update mid point values
        u1mid_gf = *gfs.u1_gf;
        u1mid_gf += *old_gfs.u1_gf;
        u1mid_gf *= 0.5;

        B1mid_gf = *gfs.B1_gf;
        B1mid_gf +=*old_gfs.B1_gf;
        B1mid_gf *= 0.5;

        if (amr_info.amr)
        {
            // refine mesh
            refiner->Apply(*pmesh);
            HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
            mfemPrintf("Refinement: total error = %e, marked element: %lld, total element: %d \n", refiner->GetTotalError(), refiner->GetNumMarkedElements(), glob_el);

            if (refiner->Stop())
            {
                mfemPrintf("Refinement stopped\n");
                break;
            }

            // update and rebalance
            // update the fe spaces
            UpdateAllFEspaces();
            // update all useful grid functions
            // UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});

            // TODO:rebalance here
            if (pmesh->Nonconforming())
            {
                pmesh->Rebalance();
                // update the fe spaces and grid functions again
                UpdateAllFEspaces();
                // UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});
            }

            // UpdateForms(
            //     {&UU_varf, &UU_old_varf, &PP_varf, &AA_varf, &AJ_varf, &AJ_old_varf, &BB_varf, &JJ_varf},
            //     {&PU_varf, &UP_varf, &UJ_varf, &UJ_old_varf, &AU_varf, &BA_varf, &JB_varf},
            //     {&fu_form, &fu_wbdry_form, &fp_form, &fA_form, &fB_form, &fJ_form});

            UpdateFinishedFEspaces();

            derefiner->SetThreshold(amr_info.coarse_frac * refiner->GetTotalError());

            // derefine mesh
            if (derefiner->Apply(*pmesh))
            {
                HYPRE_BigInt glob_el = pmesh->GetGlobalNE();
                mfemPrintf("Derefined elements. Total elements: %d .\n", glob_el);

                // update the fe spaces
                UpdateAllFEspaces();
                // update all useful grid functions
                // UpdateAllSolutions({&u1_old, &w2_old, &p0_old, &A2_old, &B1_old, &j2_old, &u1mid_gf, &B1mid_gf});

                // TODO: rebalance here

                // UpdateForms(
                //     {&UU_varf, &UU_old_varf, &PP_varf, &AA_varf, &AJ_varf, &AJ_old_varf, &BB_varf, &JJ_varf},
                //     {&PU_varf, &UP_varf, &UJ_varf, &UJ_old_varf, &AU_varf, &BA_varf, &JB_varf},
                //     {&fu_form, &fu_wbdry_form, &fp_form, &fA_form, &fB_form, &fJ_form});

                UpdateFinishedFEspaces();
            }
        }
    }

    t_half += dt;

    if (amr_info.amr)
    {
        pmesh->PrintInfo();
        delete CurlCurlInteg;
        delete estimator;
        delete refiner;
        delete derefiner;
    }

    return half_evo->GetIterCount();
}

// Calculate 1/2  (|u|^2 + s|B|^2)
void MHD_solver::CalcEnergy(const ParGridFunction &u_gf, const ParGridFunction &B_gf, real_t &Energy_kinetic, real_t &Energy_magnetic)
{

    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i = 0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, sol_info.order * 2));
    }

    VectorGridFunctionCoefficient u_gf_coeff(&u_gf);
    VectorGridFunctionCoefficient B_gf_coeff(&B_gf);
    real_t kinetic = ComputeGlobalLpNorm(2, u_gf_coeff, *pmesh, irs);
    real_t magnetic = ComputeGlobalLpNorm(2, B_gf_coeff, *pmesh, irs);

    Energy_kinetic = 0.5 * kinetic * kinetic;
    Energy_magnetic = 0.5 * pd->param.s * magnetic * magnetic;
}

// Calculate <u , v>
real_t MHD_solver::GFInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2)
{
    VectorGridFunctionCoefficient gf_2_coeff(&gf_2);
    ParLinearForm inner_form(gf_1.ParFESpace());
    inner_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(gf_2_coeff));
    inner_form.Assemble();

    real_t inner_loc = InnerProduct(inner_form, gf_1);
    real_t inner_glob;
    MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return inner_glob;
}

// Calculate <u , curl v>
real_t MHD_solver::GFDotCurlGF(const ParGridFunction &gf_1, const ParGridFunction &gf_2)
{
    CurlGridFunctionCoefficient curl_gf_2_coeff(&gf_2);
    ParLinearForm inner_form(gf_1.ParFESpace());
    inner_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(curl_gf_2_coeff));
    inner_form.Assemble();

    real_t inner_loc = InnerProduct(inner_form, gf_1);
    real_t inner_glob;
    MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return inner_glob;
}

real_t MHD_solver::GFComponentInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2, int component)
{
    MFEM_ASSERT(gf_1.VectorDim() == gf_2.VectorDim(), "Vector dimensions do not match");
    MFEM_ASSERT(component < gf_1.VectorDim(), "Invalid component");

    DenseMatrix Const;
    Const.SetSize(gf_1.VectorDim());
    Const = 0.0;
    Const(component, component) = 1.0;

    MatrixConstantCoefficient ConstCoeff(Const);
    VectorGridFunctionCoefficient gf_2_coeff(&gf_2);

    MatrixVectorProductCoefficient gf_2_comp_coeff(ConstCoeff, gf_2_coeff);

    ParLinearForm inner_form(gf_1.ParFESpace());
    inner_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(gf_2_comp_coeff));
    inner_form.Assemble();

    real_t inner_loc = InnerProduct(inner_form, gf_1);
    real_t inner_glob;
    MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return inner_glob;
}

void MHD_solver::UpdateAllFEspaces()
{
    sol_info.H1space->Update();
    sol_info.NDspace->Update();
    sol_info.RTspace->Update();
    sol_info.L2space->Update();
}

void MHD_solver::UpdateAllSolutions(const initializer_list<ParGridFunction *> &gf_list)
{
    if (Mpi::Root())
    {
        printf("----------------------------------------\n");
        printf("u1_gf: norm = %e\n", gfs.u1_gf->Norml2());
        printf("w2_gf: norm = %e\n", gfs.w2_gf->Norml2());
        printf("p0_gf: norm = %e\n", gfs.p0_gf->Norml2());
        printf("A2_gf: norm = %e\n", gfs.A2_gf->Norml2());
        printf("B1_gf: norm = %e\n", gfs.B1_gf->Norml2());
        printf("j2_gf: norm = %e\n", gfs.j2_gf->Norml2());

        printf("u2_gf: norm = %e\n", gfs.u2_gf->Norml2());
        printf("w1_gf: norm = %e\n", gfs.w1_gf->Norml2());
        printf("p3_gf: norm = %e\n", gfs.p3_gf->Norml2());
        printf("A1_gf: norm = %e\n", gfs.A1_gf->Norml2());
        printf("B2_gf: norm = %e\n", gfs.B2_gf->Norml2());
        printf("j1_gf: norm = %e\n", gfs.j1_gf->Norml2());
    }

    gfs.u1_gf->Update();
    gfs.w2_gf->Update();
    gfs.p0_gf->Update();
    gfs.A2_gf->Update();
    gfs.B1_gf->Update();
    gfs.j2_gf->Update();

    gfs.u2_gf->Update();
    gfs.w1_gf->Update();
    gfs.p3_gf->Update();
    gfs.A1_gf->Update();
    gfs.B2_gf->Update();
    gfs.j1_gf->Update();

    for (auto gf : gf_list)
    {
        gf->Update();
    }

    if (Mpi::Root())
    {
        printf("----------------------------------------\n");
        printf("u1_gf: norm = %e\n", gfs.u1_gf->Norml2());
        printf("w2_gf: norm = %e\n", gfs.w2_gf->Norml2());
        printf("p0_gf: norm = %e\n", gfs.p0_gf->Norml2());
        printf("A2_gf: norm = %e\n", gfs.A2_gf->Norml2());
        printf("B1_gf: norm = %e\n", gfs.B1_gf->Norml2());
        printf("j2_gf: norm = %e\n", gfs.j2_gf->Norml2());

        printf("u2_gf: norm = %e\n", gfs.u2_gf->Norml2());
        printf("w1_gf: norm = %e\n", gfs.w1_gf->Norml2());
        printf("p3_gf: norm = %e\n", gfs.p3_gf->Norml2());
        printf("A1_gf: norm = %e\n", gfs.A1_gf->Norml2());
        printf("B2_gf: norm = %e\n", gfs.B2_gf->Norml2());
        printf("j1_gf: norm = %e\n", gfs.j1_gf->Norml2());
    }
}

void MHD_solver::UpdateForms(const initializer_list<ParBilinearForm *> &bf_list,
                             const initializer_list<ParMixedBilinearForm *> &mbf_list,
                             const initializer_list<ParLinearForm *> &lf_list)
{
    for (auto bf : bf_list)
    {
        bf->Update();
    }

    for (auto mbf : mbf_list)
    {
        mbf->Update();
    }

    for (auto lf : lf_list)
    {
        lf->Update();
    }
}

void MHD_solver::UpdateFinishedFEspaces()
{
    sol_info.H1space->UpdatesFinished();
    sol_info.NDspace->UpdatesFinished();
    sol_info.RTspace->UpdatesFinished();
    sol_info.L2space->UpdatesFinished();
}
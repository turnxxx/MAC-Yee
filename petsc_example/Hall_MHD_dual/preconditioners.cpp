
#include "Hall_MHD_dual.hpp"
#include "preconditioners.hpp"

pc_hydro_half::pc_hydro_half(ProblemData *pd_,
                             MHDSolverInfo sol_info_,
                             LinearSolverInfo lin_info_,
                             BlockOperator *BlkOp_,
                             Array<int> &offsets_,
                             real_t theta_) : Solver(offsets_.Last()),
                                              pd(pd_),
                                              sol_info(sol_info_),
                                              lin_info(lin_info_),
                                              BlkOp(BlkOp_),
                                              block_trueOffsets(offsets_)
{

    ParBilinearForm L0_varf(sol_info.H1space);
    ConstantCoefficient eps_coeff_L(0.0001);
    L0_varf.AddDomainIntegrator(new MassIntegrator(eps_coeff_L));
    L0_varf.AddDomainIntegrator(new DiffusionIntegrator);
    L0_varf.Assemble();
    L0_varf.Finalize();
    L0mat = L0_varf.ParallelAssemble();

    diag_Lp = new HypreDiagScale(*L0mat);

    {
        HypreParVector Kd(MPI_COMM_WORLD, L0mat->GetGlobalNumRows(), L0mat->GetRowStarts());
        L0mat->GetDiag(Kd);
        Kd *= 1.0 / (lin_info.gamma);
        HypreParMatrix KinvB(*dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(1, 0)));
        KinvB.InvScaleRows(Kd);
        BtKinvB = ParMult(
            dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(0, 1)), &KinvB);
        Umat = ParAdd(
            dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(0, 0)), BtKinvB);
    }

    prec_u_amg = new HypreBoomerAMG(*Umat);
    prec_u_amg->SetPrintLevel(0);

    solver_u = new GMRESSolver(MPI_COMM_WORLD);
    solver_u->SetRelTol(lin_info.sub_pc_rtol);
    solver_u->SetMaxIter(10);
    solver_u->SetPreconditioner(*prec_u_amg);
    solver_u->SetOperator(*Umat);
    solver_u->SetPrintLevel(-1);
}

pc_hydro_half::~pc_hydro_half()
{

    delete diag_Lp;
    delete L0mat;
    delete BtKinvB;

    delete Umat;
    delete prec_u_amg;
    delete solver_u;
};

void pc_hydro_half::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *u_in = ExtractVector(x, block_trueOffsets, 0);
    Vector *u_out = ExtractVector(y, block_trueOffsets, 0);
    Vector *p_in = ExtractVector(x, block_trueOffsets, 1);
    Vector *p_out = ExtractVector(y, block_trueOffsets, 1);

    Vector p_temp(*p_in);

    // mass stabilization
    diag_Lp->Mult(*p_in, p_temp);
    p_temp *= -(lin_info.gamma + 1.0 / sol_info.dt);

    *p_out = p_temp;

    Vector u_temp(*u_in);
    BlkOp->GetBlock(0, 1).AddMult(*p_out, u_temp, -2.0);

    timer.Restart();
    solver_u->Mult(u_temp, *u_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_u: %d its, res: %lg . time: %lg \n", solver_u->GetNumIterations(), solver_u->GetFinalRelNorm(), timer.RealTime());

    delete u_in;
    delete u_out;
    delete p_in;
    delete p_out;
}

pc_magnetic_half::pc_magnetic_half(ProblemData *pd_,
                                   MHDSolverInfo sol_info_,
                                   LinearSolverInfo lin_info_,
                                   BlockOperator *BlkOp_,
                                   Array<int> &offsets_,
                                   real_t theta_)
    : pc_magnetic(pd_, sol_info_, lin_info_, BlkOp_, offsets_)
{

    ConstantCoefficient thetadtonRm(theta_ * sol_info.dt / pd->param.Rm);
    ParBilinearForm B_varf(sol_info.NDspace);
    B_varf.AddDomainIntegrator(new VectorFEMassIntegrator);
    if (sol_info.resistivity)
        B_varf.AddDomainIntegrator(new CurlCurlIntegrator(thetadtonRm));
    B_varf.Assemble();
    B_varf.EliminateEssentialBC(pd->ess_bdr_magnetic);
    B_varf.Finalize();
    Mat44 = B_varf.ParallelAssemble();

    solver_B = new CGSolver(MPI_COMM_WORLD);
    solver_B->SetRelTol(lin_info.sub_pc_rtol);
    solver_B->SetMaxIter(lin_info.sub_pc_maxit);
    if (sol_info.resistivity)
    {
        prec_B = new HypreAMS(*Mat44, sol_info.NDspace);
        HypreAMS *prec_B_ = dynamic_cast<HypreAMS *>(prec_B);
        prec_B_->SetPrintLevel(0);
    }
    else
    {
        prec_B = new HypreDiagScale(*Mat44);
    }
    solver_B->SetPreconditioner(*prec_B);
    solver_B->SetOperator(*Mat44);
    solver_B->SetPrintLevel(0);

    prec_J = new HypreDiagScale;
    solver_J = new CGSolver(MPI_COMM_WORLD);
    solver_J->SetRelTol(lin_info.sub_pc_rtol);
    solver_J->SetMaxIter(lin_info.sub_pc_maxit);
    solver_J->SetPreconditioner(*prec_J);
    solver_J->SetOperator(BlkOp->GetBlock(1, 1));
    solver_J->SetPrintLevel(-1);

    prec_A = new HypreDiagScale;
    solver_A = new CGSolver(MPI_COMM_WORLD);
    solver_A->SetRelTol(lin_info.sub_pc_rtol);
    solver_A->SetMaxIter(lin_info.sub_pc_maxit);
    solver_A->SetPreconditioner(*prec_A);
    solver_A->SetOperator(BlkOp->GetBlock(0, 0));
    solver_A->SetPrintLevel(-1);
};

pc_magnetic_half::~pc_magnetic_half()
{
    delete Mat44;
    delete prec_B;
    delete solver_B;

    delete prec_J;
    delete solver_J;

    delete prec_A;
    delete solver_A;
};

// Define the action of the Solver
void pc_magnetic_half::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *A_in = ExtractVector(x, block_trueOffsets, 0);
    Vector *A_out = ExtractVector(y, block_trueOffsets, 0);
    Vector *J_in = ExtractVector(x, block_trueOffsets, 1);
    Vector *J_out = ExtractVector(y, block_trueOffsets, 1);
    Vector *B_in = ExtractVector(x, block_trueOffsets, 2);
    Vector *B_out = ExtractVector(y, block_trueOffsets, 2);

    timer.Restart();
    solver_B->Mult(*B_in, *B_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_B: %d its, res: %lg . time: %lg \n",
                   solver_B->GetNumIterations(), solver_B->GetFinalNorm(), timer.RealTime());

    Vector j_temp(*J_in);
    BlkOp->GetBlock(1, 2).AddMult(*B_out, j_temp, -1.0);

    timer.Restart();
    solver_J->Mult(j_temp, *J_out);
    timer.Stop();

    if (lin_info.print_level == 1)
        mfemPrintf("solver_J: %d its, res: %lg . time: %lg \n",
                   solver_J->GetNumIterations(), solver_J->GetFinalRelNorm(), timer.RealTime());

    Vector A_temp(*A_in);
    BlkOp->GetBlock(0, 1).AddMult(*J_out, A_temp, -1.0);

    timer.Restart();
    solver_A->Mult(A_temp, *A_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_A: %d its, res: %lg . time: %lg \n",
                   solver_A->GetNumIterations(), solver_A->GetFinalRelNorm(), timer.RealTime());

    delete A_in;
    delete A_out;
    delete B_in;
    delete B_out;
    delete J_in;
    delete J_out;
};

pc_half::pc_half(ProblemData *pd_,
                 MHDSolverInfo sol_info_,
                 LinearSolverInfo lin_info_,
                 BlockOperator *BlkOp_,
                 Array<int> &offsets_,
                 real_t theta_)
    : Solver(offsets_.Last()),
      pd(pd_),
      sol_info(sol_info_),
      lin_info(lin_info_),
      BlkOp(BlkOp_),
      block_trueOffsets(offsets_)
{

    block_trueOffsets.GetSubArray(2, 4, block_trueOffsets_mag);
    for (int i = 0; i < block_trueOffsets_mag.Size(); i++)
        block_trueOffsets_mag[i] -= block_trueOffsets[2];
    MagOp = new BlockOperator(block_trueOffsets_mag);

    MagOp->SetBlock(0, 0, &BlkOp->GetBlock(2, 2));
    MagOp->SetBlock(0, 1, &BlkOp->GetBlock(2, 3));
    MagOp->SetBlock(1, 1, &BlkOp->GetBlock(3, 3));
    MagOp->SetBlock(1, 2, &BlkOp->GetBlock(3, 4));
    MagOp->SetBlock(2, 0, &BlkOp->GetBlock(4, 2));
    MagOp->SetBlock(2, 2, &BlkOp->GetBlock(4, 4));

    if (lin_info.mag_type == PETSC)
    {
        Array2D<HypreParMatrix *> mag_blocks(3, 3);
        mag_blocks = NULL;
        mag_blocks(0, 0) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(2, 2));
        mag_blocks(0, 1) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(2, 3));
        mag_blocks(1, 1) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(3, 3));
        mag_blocks(1, 2) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(3, 4));
        mag_blocks(2, 0) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(4, 2));
        mag_blocks(2, 2) = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(4, 4));

        MagMat = HypreParMatrixFromBlocks(mag_blocks);

        petscpc_Mag = new PetscPreconditioner(MPI_COMM_WORLD, *MagMat, "half_mag_");
        petscsolver_Mag = new PetscLinearSolver(MPI_COMM_WORLD, "half_mag_", false, true);
        petscsolver_Mag->SetOperator(*MagMat);
    }
    else
    {
        prec_magnetic = new pc_magnetic_half(pd, sol_info, lin_info, MagOp, block_trueOffsets_mag, theta_);
    }

    block_trueOffsets.GetSubArray(0, 3, block_trueOffsets_hydro);
    HydroOp = new BlockOperator(block_trueOffsets_hydro);

    HydroOp->SetBlock(0, 0, &BlkOp->GetBlock(0, 0));
    HydroOp->SetBlock(0, 1, &BlkOp->GetBlock(0, 1));
    HydroOp->SetBlock(1, 0, &BlkOp->GetBlock(1, 0));

    prec_hydro = new pc_hydro_half(pd, sol_info, lin_info, HydroOp, block_trueOffsets_hydro, theta_);
};

pc_half::~pc_half()
{
    delete HydroOp;
    delete prec_hydro;

    delete MagOp;

    if (lin_info.mag_type == PETSC)
    {
        delete petscsolver_Mag;
        delete petscpc_Mag;
        delete MagMat;
    }
    else
    {
        delete prec_magnetic;
    }
};

// Define the action of the Solver
void pc_half::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *up_in = ExtractVector(x, block_trueOffsets, 0, 2);
    Vector *up_out = ExtractVector(y, block_trueOffsets, 0, 2);

    Vector *AJB_in = ExtractVector(x, block_trueOffsets, 2, 3);
    Vector *AJB_out = ExtractVector(y, block_trueOffsets, 2, 3);

    if (lin_info.mag_type == PETSC)
    {
        timer.Restart();
        petscsolver_Mag->Mult(*AJB_in, *AJB_out);
        timer.Stop();
        if (lin_info.print_level == 1)
            mfemPrintf("petscsolver_Mag: %d its, res: %lg . time: %lg \n", petscsolver_Mag->GetNumIterations(), petscsolver_Mag->GetFinalNorm(), timer.RealTime());
    }
    else
    {
        timer.Restart();
        prec_magnetic->Mult(*AJB_in, *AJB_out);
        timer.Stop();
    }

    Vector *J_out = ExtractVector(*AJB_out, block_trueOffsets_mag, 1);

    Vector up_temp(*up_in);
    Vector *u_temp = ExtractVector(up_temp, block_trueOffsets, 0);

    BlkOp->GetBlock(0, 3).AddMult(*J_out, *u_temp, -1.0);

    timer.Restart();
    prec_hydro->Mult(up_temp, *up_out);
    timer.Stop();

    delete up_in;
    delete up_out;
    delete u_temp;

    delete J_out;
    delete AJB_in;
    delete AJB_out;
};

pc_integer::pc_integer(
    ProblemData *pd_,
    MHDSolverInfo sol_info_,
    LinearSolverInfo lin_info_,
    BlockOperator *BlkOp_,
    Array<int> offsets_) : Solver(offsets_.Last()),
                           pd(pd_),
                           sol_info(sol_info_),
                           lin_info(lin_info_),
                           BlkOp(BlkOp_),
                           block_trueOffsets(offsets_),
                           Mpmat(NULL),
                           UWOp(NULL),
                           solver_uw(NULL),
                           prec_uw(NULL),
                           solver_Mp(NULL),
                           prec_Mp(NULL),
                           MagOp(NULL),
                           prec_magnetic(NULL),
                           petscpc_Mag(NULL),
                           petscsolver_Mag(NULL),
                           MagMat(NULL)
{

    ParBilinearForm M3_varf(sol_info.L2space);
    M3_varf.AddDomainIntegrator(new MassIntegrator());
    M3_varf.Assemble();
    M3_varf.Finalize();
    Mpmat = M3_varf.ParallelAssemble();

    if (sol_info.viscosity)
    {
        block_trueOffsets.GetSubArray(0, 3, block_trueOffsets_uw);
        UWOp = new BlockOperator(block_trueOffsets_uw);
        UWOp->SetBlock(0, 0, &BlkOp->GetBlock(0, 0));
        UWOp->SetBlock(0, 1, &BlkOp->GetBlock(0, 1));
        UWOp->SetBlock(1, 0, &BlkOp->GetBlock(1, 0));
        UWOp->SetBlock(1, 1, &BlkOp->GetBlock(1, 1));
    }
    else
    {
        block_trueOffsets.GetSubArray(0, 2, block_trueOffsets_uw);
        UWOp = new BlockOperator(block_trueOffsets_uw);
        UWOp->SetBlock(0, 0, &BlkOp->GetBlock(0, 0));
    }

    prec_uw = new subpc_integer(pd, sol_info, lin_info, UWOp, block_trueOffsets_uw);
    solver_uw = new FGMRESSolver(MPI_COMM_WORLD);
    solver_uw->SetRelTol(lin_info.sub_pc_rtol);
    solver_uw->SetMaxIter(lin_info.sub_pc_maxit);
    solver_uw->SetPreconditioner(*prec_uw);
    solver_uw->SetOperator(*UWOp);
    solver_uw->SetPrintLevel(-1);

    prec_Mp = new HypreDiagScale;

    solver_Mp = new CGSolver(MPI_COMM_WORLD);
    solver_Mp->SetRelTol(lin_info.sub_pc_rtol);
    solver_Mp->SetMaxIter(lin_info.sub_pc_maxit);
    solver_Mp->SetPreconditioner(*prec_Mp);
    solver_Mp->SetOperator(*Mpmat);
    solver_Mp->SetPrintLevel(-1);

    if (sol_info.viscosity)
    {
        block_trueOffsets.GetSubArray(3, 4, block_trueOffsets_mag);
        for (int i = 0; i < block_trueOffsets_mag.Size(); i++)
            block_trueOffsets_mag[i] -= block_trueOffsets[3];
        MagOp = new BlockOperator(block_trueOffsets_mag);
        MagOp->SetBlock(0, 0, &BlkOp->GetBlock(3, 3));
        MagOp->SetBlock(0, 2, &BlkOp->GetBlock(3, 5));
        MagOp->SetBlock(1, 0, &BlkOp->GetBlock(4, 3));
        MagOp->SetBlock(1, 1, &BlkOp->GetBlock(4, 4));
        MagOp->SetBlock(2, 1, &BlkOp->GetBlock(5, 4));
        MagOp->SetBlock(2, 2, &BlkOp->GetBlock(5, 5));
    }
    else
    {
        block_trueOffsets.GetSubArray(2, 4, block_trueOffsets_mag);
        for (int i = 0; i < block_trueOffsets_mag.Size(); i++)
            block_trueOffsets_mag[i] -= block_trueOffsets[2];
        MagOp = new BlockOperator(block_trueOffsets_mag);
        MagOp->SetBlock(0, 0, &BlkOp->GetBlock(2, 2));
        MagOp->SetBlock(0, 2, &BlkOp->GetBlock(2, 4));
        MagOp->SetBlock(1, 0, &BlkOp->GetBlock(3, 2));
        MagOp->SetBlock(1, 1, &BlkOp->GetBlock(3, 3));
        MagOp->SetBlock(2, 1, &BlkOp->GetBlock(4, 3));
        MagOp->SetBlock(2, 2, &BlkOp->GetBlock(4, 4));
    }

    if (lin_info.mag_type == PETSC)
    {
        // todo: td update: magblocks are not udpated
        Array2D<HypreParMatrix *> magblocks(3, 3);
        magblocks = NULL;
        magblocks(0, 0) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(0, 0));
        magblocks(0, 2) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(0, 2));
        magblocks(1, 0) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(1, 0));
        magblocks(1, 1) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(1, 1));
        magblocks(2, 1) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(2, 1));
        magblocks(2, 2) = dynamic_cast<HypreParMatrix *>(&MagOp->GetBlock(2, 2));

        MagMat = HypreParMatrixFromBlocks(magblocks);

        petscpc_Mag = new PetscPreconditioner(MPI_COMM_WORLD, *MagMat, "integer_mag_");
        petscsolver_Mag = new PetscLinearSolver(MPI_COMM_WORLD, "integer_mag_", false, true);
        petscsolver_Mag->SetOperator(*MagMat);
    }
    else
    {
        prec_magnetic = new pc_magnetic_integer(pd, sol_info, lin_info, MagOp, block_trueOffsets_mag);
    }
};

pc_integer::~pc_integer()
{

    delete UWOp;
    delete prec_uw;
    delete solver_uw;

    delete prec_Mp;
    delete solver_Mp;
    delete Mpmat;

    delete MagOp;

    if (lin_info.mag_type == PETSC)
    {
        delete petscpc_Mag;
        delete petscsolver_Mag;
        delete MagMat;
    }
    else
    {
        delete prec_magnetic;
    }
};

// Define the action of the Solver
void pc_integer::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *u_in = nullptr;
    Vector *u_out = nullptr;
    Vector *w_in = nullptr;
    Vector *w_out = nullptr;
    Vector *p_in = nullptr;
    Vector *p_out = nullptr;
    Vector *uw_in = nullptr;
    Vector *uw_out = nullptr;
    Vector *ABJ_in = nullptr;
    Vector *ABJ_out = nullptr;

    if (sol_info.viscosity)
    {
        u_in = ExtractVector(x, block_trueOffsets, 0);
        u_out = ExtractVector(y, block_trueOffsets, 0);
        w_in = ExtractVector(x, block_trueOffsets, 1);
        w_out = ExtractVector(y, block_trueOffsets, 1);
        p_in = ExtractVector(x, block_trueOffsets, 2);
        p_out = ExtractVector(y, block_trueOffsets, 2);
        uw_in = ExtractVector(x, block_trueOffsets, 0, 2);
        uw_out = ExtractVector(y, block_trueOffsets, 0, 2);
        ABJ_in = ExtractVector(x, block_trueOffsets, 3, 3);
        ABJ_out = ExtractVector(y, block_trueOffsets, 3, 3);
    }
    else
    {
        u_in = ExtractVector(x, block_trueOffsets, 0);
        u_out = ExtractVector(y, block_trueOffsets, 0);
        p_in = ExtractVector(x, block_trueOffsets, 1);
        p_out = ExtractVector(y, block_trueOffsets, 1);
        uw_in = ExtractVector(x, block_trueOffsets, 0, 1);
        uw_out = ExtractVector(y, block_trueOffsets, 0, 1);
        ABJ_in = ExtractVector(x, block_trueOffsets, 2, 3);
        ABJ_out = ExtractVector(y, block_trueOffsets, 2, 3);
    }

    if (petscsolver_Mag)
    {
        timer.Restart();
        petscsolver_Mag->Mult(*ABJ_in, *ABJ_out);
        timer.Stop();
        if (lin_info.print_level == 1)
            mfemPrintf("petscsolver_Mag: %d its, res: %lg . time: %lg \n", petscsolver_Mag->GetNumIterations(), petscsolver_Mag->GetFinalNorm(), timer.RealTime());
    }
    else
    {
        timer.Restart();
        prec_magnetic->Mult(*ABJ_in, *ABJ_out);
        timer.Stop();
    }

    Vector uw_temp(*uw_in);
    Vector *u_temp = ExtractVector(uw_temp, block_trueOffsets, 0);

    Vector *J_out = ExtractVector(*ABJ_out, block_trueOffsets_mag, 2);

    BlkOp->GetBlock(0, sol_info.viscosity ? 5 : 4).AddMult(*J_out, *u_temp, -1.0);

    solver_Mp->Mult(*p_in, *p_out);
    *p_out *= -lin_info.gamma;
    BlkOp->GetBlock(0, sol_info.viscosity ? 2 : 1).AddMult(*p_out, *u_temp, -1.0);

    timer.Restart();
    solver_uw->Mult(uw_temp, *uw_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_uw: %d its, res: %lg . time: %lg \n", solver_uw->GetNumIterations(), solver_uw->GetFinalRelNorm(), timer.RealTime());

    delete u_temp;
    delete uw_in;
    delete uw_out;

    delete u_in;
    delete u_out;
    if (w_in)
        delete w_in;
    if (w_out)
        delete w_out;
    delete p_in;
    delete p_out;
    delete J_out;
    delete ABJ_in;
    delete ABJ_out;
};

void subpc_integer::SetUUoperator(HypreParMatrix *Umat_)
{
    if (Umat)
        delete Umat;
    if (sol_info.viscosity)
    {
        Umat = ParAdd(Umat_, KtMinvK);
    }
    else
    {
        Umat = new HypreParMatrix(*Umat_);
    }
    solver_u->SetOperator(*Umat);
}

subpc_integer::subpc_integer(ProblemData *pd_,
                             MHDSolverInfo sol_info_,
                             LinearSolverInfo lin_info_,
                             BlockOperator *BlkOp_,
                             Array<int> offsets_) : Solver(offsets_.Last()),
                                                    pd(pd_),
                                                    sol_info(sol_info_),
                                                    lin_info(lin_info_),
                                                    BlkOp(BlkOp_),
                                                    block_trueOffsets(offsets_)
{

    // build Umat
    HypreParMatrix *Mat00 = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(0, 0));
    HypreParMatrix *Mat11 = nullptr;
    HypreParMatrix *Mat10 = nullptr;
    HypreParMatrix *Mat01 = nullptr;
    KtMinvK = nullptr;
    if (sol_info.viscosity)
    {
        Mat11 = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(1, 1));
        Mat10 = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(1, 0));
        Mat01 = dynamic_cast<HypreParMatrix *>(&BlkOp->GetBlock(0, 1));
        HypreParVector Md(MPI_COMM_WORLD,
                          Mat11->GetGlobalNumRows(),
                          Mat11->GetRowStarts());
        Mat11->GetDiag(Md);
        Md.Neg();
        HypreParMatrix MinvK(*Mat10);
        MinvK.InvScaleRows(Md);
        KtMinvK = ParMult(Mat01, &MinvK);
        Umat = ParAdd(Mat00, KtMinvK);
    }
    else
    {
        Umat = new HypreParMatrix(*Mat00);
    }

    prec_u = new HypreADS(*Umat, sol_info.RTspace);

    solver_u = new GMRESSolver(MPI_COMM_WORLD);
    solver_u->SetRelTol(lin_info.sub_pc_rtol);
    solver_u->SetMaxIter(1);
    solver_u->SetPreconditioner(*prec_u);
    solver_u->SetOperator(*Umat);
    solver_u->SetPrintLevel(-1);

    if (sol_info.viscosity)
    {
        prec_w = new HypreDiagScale(*Mat11);
        solver_w = new CGSolver(MPI_COMM_WORLD);
        solver_w->SetRelTol(lin_info.sub_pc_rtol);
        solver_w->SetMaxIter(lin_info.sub_pc_maxit);
        solver_w->SetPreconditioner(*prec_w);
        solver_w->SetOperator(*Mat11);
        solver_w->SetPrintLevel(-1);
    }
    else
    {
        solver_w = nullptr;
        prec_w = nullptr;
    }
}

subpc_integer::~subpc_integer()
{
    delete Umat;
    if (KtMinvK)
        delete KtMinvK;
    delete prec_u;
    delete solver_u;
    if (prec_w)
        delete prec_w;
    if (solver_w)
        delete solver_w;
}

void subpc_integer::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *u_in = ExtractVector(x, block_trueOffsets, 0);
    Vector *u_out = ExtractVector(y, block_trueOffsets, 0);
    Vector *w_in = nullptr;
    Vector *w_out = nullptr;
    if (sol_info.viscosity)
    {
        w_in = ExtractVector(x, block_trueOffsets, 1);
        w_out = ExtractVector(y, block_trueOffsets, 1);
    }

    timer.Restart();
    solver_u->Mult(*u_in, *u_out);
    if (lin_info.print_level == 1)
        mfemPrintf("solver_u: %d its, res: %lg . time: %lg \n", solver_u->GetNumIterations(), solver_u->GetFinalNorm(), timer.RealTime());

    if (sol_info.viscosity)
    {
        Vector w_temp(*w_in);
        BlkOp->GetBlock(1, 0).AddMult(*u_out, w_temp, -1.0);

        timer.Restart();
        solver_w->Mult(w_temp, *w_out);
        if (lin_info.print_level == 1)
            mfemPrintf("solver_w: %d its, res: %lg . time: %lg \n", solver_w->GetNumIterations(), solver_w->GetFinalRelNorm(), timer.RealTime());
    }

    delete u_in;
    delete u_out;
    if (sol_info.viscosity)
    {
        delete w_in;
        delete w_out;
    }
};

pc_magnetic_integer::pc_magnetic_integer(
    ProblemData *pd_,
    MHDSolverInfo sol_info_,
    LinearSolverInfo lin_info_,
    BlockOperator *BlkOp_,
    Array<int> offsets_) : Solver(offsets_.Last()),
                           pd(pd_),
                           sol_info(sol_info_),
                           lin_info(lin_info_),
                           BlkOp(BlkOp_),
                           block_trueOffsets(offsets_)
{

    prec_A = new HypreDiagScale;
    solver_A = new CGSolver(MPI_COMM_WORLD);
    solver_A->SetRelTol(lin_info.sub_pc_rtol);
    solver_A->SetMaxIter(lin_info.sub_pc_maxit);
    solver_A->SetPreconditioner(*prec_A);
    solver_A->SetOperator(BlkOp->GetBlock(0, 0));
    solver_A->SetPrintLevel(-1);

    prec_B = new HypreDiagScale;
    solver_B = new CGSolver(MPI_COMM_WORLD);
    solver_B->SetRelTol(lin_info.sub_pc_rtol);
    solver_B->SetMaxIter(lin_info.sub_pc_maxit);
    solver_B->SetPreconditioner(*prec_B);
    solver_B->SetOperator(BlkOp->GetBlock(1, 1));
    solver_B->SetPrintLevel(-1);

    if (!sol_info.Hall)
    {
        ConstantCoefficient dton2Rm(sol_info.dt / (2.0 * pd->param.Rm));
        ParBilinearForm J_varf(sol_info.NDspace);
        J_varf.AddDomainIntegrator(new VectorFEMassIntegrator);
        if (sol_info.resistivity)
            J_varf.AddDomainIntegrator(new CurlCurlIntegrator(dton2Rm));
        J_varf.Assemble();
        J_varf.Finalize();
        Mat55 = J_varf.ParallelAssemble();

        if (sol_info.resistivity)
        {
            prec_J = new HypreAMS(sol_info.NDspace);
            HypreAMS *prec_J_ = dynamic_cast<HypreAMS *>(prec_J);
            prec_J_->SetPrintLevel(0);
        }
        else
        {
            prec_J = new HypreDiagScale(*Mat55);
        }
        solver_J = new CGSolver(MPI_COMM_WORLD);
        solver_J->SetRelTol(lin_info.sub_pc_rtol);
        solver_J->SetMaxIter(lin_info.sub_pc_maxit);
        solver_J->SetPreconditioner(*prec_J);
        solver_J->SetOperator(*Mat55);
        solver_J->SetPrintLevel(-1);
    }
    else
    {
        mfemPrintf("Hall term not implemented yet\n");
        exit(1);
    }
};

pc_magnetic_integer::~pc_magnetic_integer()
{

    delete prec_A;
    delete solver_A;

    delete prec_B;
    delete solver_B;

    delete Mat55;
    delete prec_J;
    delete solver_J;
};

void pc_magnetic_integer::Mult(const mfem::Vector &x, mfem::Vector &y) const
{

    StopWatch timer;

    Vector *A_in = ExtractVector(x, block_trueOffsets, 0);
    Vector *A_out = ExtractVector(y, block_trueOffsets, 0);
    Vector *B_in = ExtractVector(x, block_trueOffsets, 1);
    Vector *B_out = ExtractVector(y, block_trueOffsets, 1);
    Vector *J_in = ExtractVector(x, block_trueOffsets, 2);
    Vector *J_out = ExtractVector(y, block_trueOffsets, 2);

    timer.Restart();
    solver_J->Mult(*J_in, *J_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_J: %d its, res: %lg . time: %lg \n", solver_J->GetNumIterations(), solver_J->GetFinalNorm(), timer.RealTime());

    Vector B_temp(*B_in);
    if (sol_info.resistivity)
        BlkOp->GetBlock(1, 0).AddMult(*J_out, B_temp, sol_info.dt / (2.0 * pd->param.Rm));

    timer.Restart();
    solver_B->Mult(B_temp, *B_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_B: %d its, res: %lg . time: %lg \n", solver_B->GetNumIterations(), solver_B->GetFinalRelNorm(), timer.RealTime());

    Vector A_temp(*A_in);
    if (sol_info.resistivity)
        BlkOp->GetBlock(0, 2).AddMult(*J_out, A_temp, -1.0);

    timer.Restart();
    solver_A->Mult(A_temp, *A_out);
    timer.Stop();
    if (lin_info.print_level == 1)
        mfemPrintf("solver_A: %d its, res: %lg . time: %lg \n", solver_A->GetNumIterations(), solver_A->GetFinalRelNorm(), timer.RealTime());

    delete A_in;
    delete A_out;
    delete B_in;
    delete B_out;
    delete J_in;
    delete J_out;
}
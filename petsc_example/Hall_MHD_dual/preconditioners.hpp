
#ifndef PRECONDITIONERS_HPP
#define PRECONDITIONERS_HPP

#include "Hall_MHD_dual.hpp"
#include "Integrators.hpp"

class pc_hydro_half : public Solver
{
private:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> &block_trueOffsets;

    HypreParMatrix *L0mat;
    HypreDiagScale *diag_Lp;

    HypreParMatrix *Umat;
    HypreParMatrix *BtKinvB;
    HypreBoomerAMG *prec_u_amg;
    GMRESSolver *solver_u;

public:
    pc_hydro_half(ProblemData *pd_,
                  MHDSolverInfo sol_info_,
                  LinearSolverInfo lin_info_,
                  BlockOperator *BlkOp_,
                  Array<int> &offsets_,
                  real_t theta_);

    ~pc_hydro_half();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

    void SetOperator(const Operator &op) {};
    
    void SetUUoperator(HypreParMatrix *Umat_)
    {
        if(Umat) delete Umat;
        Umat = ParAdd(Umat_, BtKinvB);
        solver_u->SetOperator(*Umat);
    }
};

class pc_magnetic : public Solver
{
protected:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> &block_trueOffsets;

public:
    pc_magnetic(ProblemData *pd_,
                MHDSolverInfo sol_info_,
                LinearSolverInfo lin_info_,
                BlockOperator *BlkOp_,
                Array<int> &offsets_) : Solver(offsets_.Last()),
                                        pd(pd_),
                                        sol_info(sol_info_),
                                        lin_info(lin_info_),
                                        BlkOp(BlkOp_),
                                        block_trueOffsets(offsets_) {};

    ~pc_magnetic() {};

    // Define the action of the Solver
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const = 0;

    void SetOperator(const Operator &op) {};
};

class pc_magnetic_half : public pc_magnetic
{
private:
    HypreSolver *prec_B;
    CGSolver *solver_B;

    HypreParMatrix *Mat44;

    HypreDiagScale *prec_J;
    CGSolver *solver_J;

    HypreDiagScale *prec_A;
    CGSolver *solver_A;

public:
    pc_magnetic_half(ProblemData *pd_,
                     MHDSolverInfo sol_info_,
                     LinearSolverInfo lin_info_,
                     BlockOperator *BlkOp_,
                     Array<int> &offsets_,
                     real_t theta_);

    ~pc_magnetic_half();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;
};

class pc_half : public Solver
{
private:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> block_trueOffsets;

    Array<int> block_trueOffsets_hydro;
    Array<int> block_trueOffsets_mag;

    BlockOperator *HydroOp;
    // FGMRESSolver *solver_hydro;
    pc_hydro_half *prec_hydro;

    BlockOperator *MagOp;
    // FGMRESSolver *solver_Mag;
    pc_magnetic *prec_magnetic;

    HypreParMatrix *MagMat;
    PetscLinearSolver *petscsolver_Mag;
    PetscPreconditioner *petscpc_Mag;

public:
    pc_half(ProblemData *pd_,
            MHDSolverInfo sol_info_,
            LinearSolverInfo lin_info_,
            BlockOperator *BlkOp_,
            Array<int> &offsets_,
            real_t theta_);

    ~pc_half();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

    void SetOperator(const Operator &op) {};

    void SetTDOperators(HypreParMatrix *UUmat_,
                        HypreParMatrix *UJmat_,
                        HypreParMatrix *AUmat_,
                        HypreParMatrix *AJmat_)
    { 
        HydroOp->SetBlock(0, 0, UUmat_);
        if(AJmat_)
            MagOp->SetBlock(0, 1, AJmat_);
    }
};

// sub preconditioner for the (u,w) system
class subpc_integer : public Solver
{
private:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> block_trueOffsets;

    HypreParMatrix *Umat;
    HypreParMatrix *KtMinvK;
    GMRESSolver *solver_u;
    HypreADS *prec_u;

    CGSolver *solver_w;
    HypreDiagScale *prec_w;

public:
    subpc_integer(ProblemData *pd_,
                  MHDSolverInfo sol_info_,
                  LinearSolverInfo lin_info_,
                  BlockOperator *BlkOp_,
                  Array<int> offsets_);

    ~subpc_integer();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

    void SetOperator(const Operator &op) {};

    void SetUUoperator(HypreParMatrix *Umat_);
};

// Custom Preconditioner for Magnetic block in integer steps
// Only for non-Hall MHD
class pc_magnetic_integer : public Solver
{
private:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> block_trueOffsets;

    HypreDiagScale *prec_A;
    CGSolver *solver_A;

    HypreDiagScale *prec_B;
    CGSolver *solver_B;

    HypreParMatrix *Mat55;
    HypreSolver *prec_J;
    CGSolver *solver_J;

public:
    pc_magnetic_integer(ProblemData *pd_,
                        MHDSolverInfo sol_info_,
                        LinearSolverInfo lin_info_,
                        BlockOperator *BlkOp_,
                        Array<int> offsets_);

    ~pc_magnetic_integer();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

    void SetOperator(const Operator &op) {};
};

class pc_integer : public Solver
{
private:
    ProblemData *pd;
    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;

    BlockOperator *BlkOp;
    Array<int> block_trueOffsets;
    Array<int> block_trueOffsets_uw;
    Array<int> block_trueOffsets_mag;

    HypreParMatrix *Mpmat;
    HypreDiagScale *prec_Mp;
    CGSolver *solver_Mp;

    BlockOperator *UWOp;
    FGMRESSolver *solver_uw;
    subpc_integer *prec_uw;

    BlockOperator *MagOp;

    // FGMRESSolver *solver_Mag;
    pc_magnetic_integer *prec_magnetic;

    HypreParMatrix *MagMat;
    PetscLinearSolver *petscsolver_Mag;
    PetscPreconditioner *petscpc_Mag;

public:
    pc_integer(ProblemData *pd_,
               MHDSolverInfo sol_info_,
               LinearSolverInfo lin_info_,
               BlockOperator *BlkOp_,
               Array<int> offsets_);

    ~pc_integer();

    // Define the action of the Solver
    void Mult(const mfem::Vector &x, mfem::Vector &y) const;

    void SetOperator(const Operator &op) {};

    void SetTDoperators(HypreParMatrix *UUmat_,
                        HypreParMatrix *UJmat_,
                        HypreParMatrix *AUmat_,
                        HypreParMatrix *AJmat_)
    {
        UWOp->SetBlock(0, 0, UUmat_);
        if (AJmat_)
            MagOp->SetBlock(0, 2, AJmat_);

        prec_uw->SetUUoperator(UUmat_);
    }
};

#endif
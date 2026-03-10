
#include "Hall_MHD_dual.hpp"
#include "preconditioners.hpp"

class EvolutionOperator
{
protected:
    /* not owned */
    ProblemData *pd;
    ParMesh *pmesh;

    MHDSolverInfo sol_info;
    LinearSolverInfo lin_info;
    AMRInfo amr_info;
    GridFunctions gfs;
    GridFunctions old_gfs;

    BlockOperator *BigOp;

    Solver *solver;
    FGMRESSolver *fgmres_solver;
    PetscLinearSolver *petsc_solver;
    Array<int> offsets;

public:
    EvolutionOperator(ProblemData *pd_,
                      ParMesh *pmesh_,
                      MHDSolverInfo sol_info_,
                      LinearSolverInfo lin_info_,
                      AMRInfo amr_info_,
                      GridFunctions gfs_,
                      GridFunctions old_gfs_)
        : pd(pd_),
          pmesh(pmesh_),
          sol_info(sol_info_),
          lin_info(lin_info_),
          amr_info(amr_info_),
          gfs(gfs_),
        old_gfs(old_gfs_),
          BigOp(nullptr),
          solver(nullptr),
          fgmres_solver(nullptr),
          petsc_solver(nullptr)
    {
    }

    virtual ~EvolutionOperator()
    {
        if (BigOp)
            delete BigOp;
    }
    
    int GetIterCount() const
    {
        return fgmres_solver? fgmres_solver->GetNumIterations() : 0;
    }
};

class IntegerEvolutionOperator : public EvolutionOperator
{
protected:
    /* Coeffcients */
    ConstantCoefficient *oneondt;
    ConstantCoefficient *m_one;
    ConstantCoefficient *gammacoeff;
    ConstantCoefficient *oneon2Re;
    ConstantCoefficient *oneon2Rm;
    VectorFunctionCoefficient *ubdrycoeff;
    VectorFunctionCoefficient *Bbdrycoeff;
    VectorFunctionCoefficient *wbdrycoeff;
    FunctionCoefficient *pbdrycoeff;

    VectorGridFunctionCoefficient *w2_gf_coeff;
    ScalarVectorProductCoefficient *halfw2_coeff;
    ScalarVectorProductCoefficient *mhalfw2_coeff;

    VectorGridFunctionCoefficient *B1_gf_coeff;
    ScalarVectorProductCoefficient *halfB1_coeff;
    ScalarVectorProductCoefficient *halfsB1_coeff;
    ScalarVectorProductCoefficient *mhalfRHB1_coeff;

    /* Bilinear forms*/
    ParBilinearForm *UU_varf;
    ParBilinearForm *UU_old_varf;
    ParMixedBilinearForm *UJ_varf;
    ParMixedBilinearForm *UW_varf;
    ParBilinearForm *WW_varf;
    ParMixedBilinearForm *WU_varf;
    ParMixedBilinearForm *PU_varf;
    ParMixedBilinearForm *UP_varf;
    ParBilinearForm *AA_varf;
    ParBilinearForm *AJ_varf;
    ParMixedBilinearForm *AU_varf;
    ParBilinearForm *BB_varf;
    ParMixedBilinearForm *BA_varf;
    ParBilinearForm *JJ_varf;
    ParMixedBilinearForm *JB_varf;

    /* Matrices */
    HypreParMatrix *UUmat;
    HypreParMatrix *UJmat;
    HypreParMatrix *WUmat;
    HypreParMatrix *UWmat;
    HypreParMatrix *PUmat;
    HypreParMatrix *UPmat;
    HypreParMatrix *WWmat;
    HypreParMatrix *AAmat;
    HypreParMatrix *AJmat;
    HypreParMatrix *AUmat;
    HypreParMatrix *BBmat;
    HypreParMatrix *BAmat;
    HypreParMatrix *JJmat;
    HypreParMatrix *JBmat;

    HypreParMatrix *WUmat_e;
    HypreParMatrix *PUmat_e;
    HypreParMatrix *AUmat_e;
    HypreParMatrix *UWmat_e;
    HypreParMatrix *UUmat_e;
    HypreParMatrix *WWmat_e;

    /* preconditioner and solver */
    pc_integer *pc;

public:
    IntegerEvolutionOperator(ProblemData *pd_,
                             ParMesh *pmesh_,
                             MHDSolverInfo solver_info_,
                             LinearSolverInfo lin_solver_info_,
                             AMRInfo amr_info_,
                             GridFunctions gfs_,
                             GridFunctions old_gfs_);

    ~IntegerEvolutionOperator();

    void AssembleOperators();

    void AssembleTDOperators();

    void Step(real_t &t_int, real_t dt);
};

class HalfEvolutionOperator : public EvolutionOperator
{
protected:
    /* Coeffcients */
    ConstantCoefficient *oneondt;
    ConstantCoefficient *m_one;
    VectorFunctionCoefficient *ubdrycoeff;
    VectorFunctionCoefficient *Bbdrycoeff;
    VectorFunctionCoefficient *wbdrycoeff;
    FunctionCoefficient *pbdrycoeff;

    ConstantCoefficient *theta_coeff;
    SumCoefficient *thetam1_coeff;
    ProductCoefficient *mtheta_coeff;
    ProductCoefficient *mthetam1_coeff;
    VectorGridFunctionCoefficient *w1_gf_coeff;
    ScalarVectorProductCoefficient *thetaw1_coeff;
    ScalarVectorProductCoefficient *thetam1_w1_coeff;

    ConstantCoefficient *one_on_Re;
    ProductCoefficient *theta_on_Re;
    ProductCoefficient *thetam1_on_Re;

    ConstantCoefficient *one_on_Rm;
    ProductCoefficient *theta_on_Rm;
    ProductCoefficient *thetam1_on_Rm;

    VectorGridFunctionCoefficient *B2_gf_coeff;
    ConstantCoefficient *s_coeff;
    ScalarVectorProductCoefficient *thetaB2_coeff;
    ScalarVectorProductCoefficient *sB2_coeff;
    ScalarVectorProductCoefficient *thetasB2_coeff;
    ScalarVectorProductCoefficient *thetam1sB2_coeff;

    ConstantCoefficient *RH_coeff;
    ScalarVectorProductCoefficient *RH_B2_coeff;
    ScalarVectorProductCoefficient *mthetaRH_B2_coeff;
    ScalarVectorProductCoefficient *mthetam1RH_B2_coeff;

    /* Bilinear forms*/
    ParBilinearForm *UU_varf;
    ParBilinearForm *UU_old_varf;
    ParMixedBilinearForm *PU_varf;
    ParMixedBilinearForm *UP_varf;
    ParBilinearForm *PP_varf;
    ParMixedBilinearForm *UJ_varf;
    ParMixedBilinearForm *UJ_old_varf;
    ParBilinearForm *AA_varf;
    ParBilinearForm *AJ_varf;
    ParBilinearForm *AJ_old_varf;
    ParMixedBilinearForm *AU_varf;
    ParBilinearForm *BB_varf;
    ParMixedBilinearForm *BA_varf;
    ParBilinearForm *JJ_varf;
    ParMixedBilinearForm *JB_varf;

    /* Matrices */
    HypreParMatrix *UUmat;
    HypreParMatrix *PUmat;
    HypreParMatrix *UPmat;
    HypreParMatrix *PPmat;
    HypreParMatrix *UJmat;
    HypreParMatrix *AAmat;
    HypreParMatrix *AJmat;
    HypreParMatrix *AUmat;
    HypreParMatrix *BBmat;
    HypreParMatrix *BAmat;
    HypreParMatrix *JJmat;
    HypreParMatrix *JBmat;

    HypreParMatrix *PUmat_e;
    HypreParMatrix *AUmat_e;
    HypreParMatrix *UUmat_e;
    HypreParMatrix *UPmat_e;
    HypreParMatrix *PPmat_e;
    HypreParMatrix *JBmat_e;
    HypreParMatrix *BBmat_e;

    /* preconditioner and solver */
    pc_half *pc;

public:
    HalfEvolutionOperator(ProblemData *pd_,
                          ParMesh *pmesh_,
                          MHDSolverInfo solver_info_,
                          LinearSolverInfo lin_solver_info_,
                          AMRInfo amr_info_,
                          GridFunctions gfs_,
                          GridFunctions old_gfs_);

    ~HalfEvolutionOperator();

    void AssembleOperators(real_t theta, real_t dt);

    void AssembleTDOperators(real_t theta, real_t dt);

    void Step(real_t &t_half, real_t dt, real_t theta);
};

class MHD_solver
{
public:
    ProblemData *pd;
    ParMesh *pmesh;

    MHDSolverInfo sol_info;

    LinearSolverInfo lin_sol_int;
    LinearSolverInfo lin_sol_half;

    AMRInfo amr_info;

    // visualization
    bool visualization;

    H1_FECollection *H1_coll;
    ND_FECollection *ND_coll;
    RT_FECollection *RT_coll;
    L2_FECollection *L2_coll;

    // solutions (GridFunctions)
    GridFunctions gfs;
    
    GridFunctions old_gfs;

    ParaViewDataCollection *paraview_dc;

    // Evolution operators
    IntegerEvolutionOperator *integer_evo;
    HalfEvolutionOperator *half_evo;

public:
    MHD_solver(ProblemData *pd_,
               ParMesh *pmesh_,
               MHDSolverInfo sol_info_,
               LinearSolverInfo lin_sol_int_,
               LinearSolverInfo lin_sol_half_,
               AMRInfo amr_info_,
               bool visualization_);

    ~MHD_solver();

    // please set all the parameters before calling this.
    void Init();

    // integer time steps:
    // given w2, update u2, w1, p3
    // return number of iterations
    int IntegerStep(real_t &t_int, real_t dt, bool update = true);

    void CalcEnergy(const ParGridFunction &u_gf, const ParGridFunction &B_gf, real_t &Energy_kinetic, real_t &Energy_magnetic);

    real_t GFInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2);
    real_t GFComponentInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2, int component = 0);
    real_t GFDotCurlGF(const ParGridFunction &gf_1, const ParGridFunction &gf_2);

    // half integer time steps:
    // given w1, update u1, w2, p2
    // return number of iterations
    int HalfStep(real_t &t_half, real_t dt, real_t theta = 0.5, bool update = true, bool first_step = false);

    void InitializeByProjection();

    void ComputeError(real_t t_int, real_t t_half, real_t dt);
    
    // compute error at t = t_f - 0.5*dt
    void ComputeErrorPrimalDual(real_t t, bool print, ofstream *error_out=nullptr);

    void UpdateAllFEspaces();

    void UpdateFinishedFEspaces();

    void UpdateAllSolutions(const initializer_list<ParGridFunction *> &gf_list);

    void UpdateForms(const initializer_list<ParBilinearForm *> &bf_list,
                     const initializer_list<ParMixedBilinearForm *> &mbf_list,
                     const initializer_list<ParLinearForm *> &lf_list);

    void SetupParaview(const char *paraview_dir);
    void OutputParaview(const char *paraview_dir, int cycle, real_t t);
};
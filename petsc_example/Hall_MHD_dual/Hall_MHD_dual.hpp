#ifndef HALL_MHD_DUAL_HPP
#define HALL_MHD_DUAL_HPP

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unistd.h>   

using namespace std;
using namespace mfem;


// useful macros
#define mfemPrintf(...) { \
      if (Mpi::Root()) { \
            printf(__VA_ARGS__); \
      } \
}

#define SLEEP_HERE { \
      { \
            int flag = 1; \
            printf("pid %d:\n", getpid()); \
            while(flag){ \
            sleep(1); \
            } \
      } \
}

typedef enum 
{
    MFEM,
    PETSC
} SolverType;

typedef struct
{
    int order;
    int dim;
    real_t dt;
    real_t t_final;
    bool Hall;
    bool viscosity;
    bool resistivity;
    
    ParFiniteElementSpace *H1space;
    ParFiniteElementSpace *NDspace;
    ParFiniteElementSpace *RTspace;
    ParFiniteElementSpace *L2space;
    
} MHDSolverInfo;

typedef struct
{
    SolverType type;
    int maxit;
    real_t rtol;
    real_t atol;
    real_t gamma;
    int print_level;
    bool iterative_mode;

    real_t sub_pc_rtol;
    real_t sub_pc_atol;
    int sub_pc_maxit;
    
    SolverType mag_type;
} LinearSolverInfo;

typedef struct
{
    ParGridFunction *u2_gf;
    ParGridFunction *w1_gf;
    ParGridFunction *p3_gf;
    ParGridFunction *A1_gf;
    ParGridFunction *B2_gf;
    ParGridFunction *j1_gf;
    
    ParGridFunction *u1_gf;
    ParGridFunction *w2_gf;
    ParGridFunction *p0_gf;
    ParGridFunction *A2_gf;
    ParGridFunction *B1_gf;
    ParGridFunction *j2_gf;
} GridFunctions;

typedef struct
{
    bool amr;
    int max_amr_iter_init;
    int max_elements_init;
    real_t refine_frac_init;
    real_t coarse_frac_init;

    int max_amr_iter;
    real_t refine_frac;
    real_t coarse_frac;
    int max_elements;
    real_t total_err_goal;
    
} AMRInfo;

typedef struct ParamList_ {
    real_t Re, Rm, s, RH;
} ParamList;

typedef enum BDRYTYPE_ {
    TANGENT_PRESSURE,
    NORMAL_VORTICITY    
} BDRYTYPE;

#define GETPARAM(param) \
    real_t Re = param.Re; \
    real_t Rm = param.Rm; \
    real_t s = param.s; \
    real_t RH = param.RH; \

// param dependent functions
template<typename ParamType, typename Func>
class Pfunc;

template<typename ParamType, typename R, typename... Args>
class Pfunc<ParamType, R(Args...)>{
    ParamType param;
    std::function<R(Args..., ParamType)> func;
    
public:
    Pfunc(std::function<R(Args..., ParamType)>func_, ParamType param_): func(func_), param(param_){}
    void SetParam(ParamType param_) { param = param_; };
    R operator()(Args... args){
        return func(args..., param);
    }
};

#define PFUNCV_TD_DEF(func_f, F_)                       \
static void func_f(const Vector & xi, real_t t, Vector & u, ParamList param) \
{                                               \
    GETPARAM(param);                            \
    real_t x(xi(0));                            \
    real_t y(xi(1));                            \
    real_t z(xi(2));                            \
    F_(u, t, x, y, z);                         \
}                                               \

#define FUNCV_TD_DEF(func_f, F_)                       \
static void func_f(const Vector & xi, real_t t, Vector & u) \
{                                               \
    real_t x(xi(0));                            \
    real_t y(xi(1));                            \
    real_t z(xi(2));                            \
    F_(u, t, x, y, z);                         \
}                                               \

#define FUNCV_DEF(func_f, F_)                       \
static void func_f(const Vector & xi, Vector & u) \
{                                               \
    real_t x(xi(0));                            \
    real_t y(xi(1));                            \
    real_t z(xi(2));                            \
    F_(u, x, y, z);                         \
}                                               \

#define FUNCS_DEF(func_f, F_)                       \
static real_t func_f(const Vector & xi) \
{                                               \
    real_t x(xi(0));                            \
    real_t y(xi(1));                            \
    real_t z(xi(2));                            \
    real_t u;                                  \
    F_(u, x, y, z);                         \
    return u;                                  \
}                                               \

#define FUNCS_TD_DEF(func_f, F_)                       \
static real_t func_f(const Vector & xi, real_t t) \
{                                               \
    real_t x(xi(0));                            \
    real_t y(xi(1));                            \
    real_t z(xi(2));                            \
    real_t u;                                  \
    F_(u, t, x, y, z);                         \
    return u;                                  \
}                                               \

typedef std::function <void(const Vector &, Vector &)> Vfunc;
typedef std::function <void(const Vector &, real_t, Vector &)> TDVfunc;
typedef std::function <real_t(const Vector &)> Sfunc;
typedef std::function <real_t(const Vector &, real_t)> TDSfunc;


// problem_data
// including: parameters (Re, Rm, s)
//           initial conditions (u0, A0)
//           boundary conditions (u_bc, b_bc)
//           exact solutions (u_ex, A_ex) (if available)
class ProblemData
{
public:
    ParamList param;
    
    bool has_exact_solution;
    
    // initial conditions
    Vfunc u0_fun;
    Vfunc w0_fun;
    Sfunc p0_fun;
    Vfunc A0_fun;
    Vfunc B0_fun;
    Vfunc j0_fun;
    
    // boundary conditions
    TDVfunc ubdry_fun;
    TDVfunc wbdry_fun;
    TDSfunc pbdry_fun;
    TDVfunc Bbdry_fun;
    
    // essential boundary conditions
    Array<int> ess_bdr_tangent;
    Array<int> ess_bdr_normal;
    Array<int> ess_bdr_magnetic;
    
    /*
    1. z=0
    2. y=0
    3. x=1
    4. y=1
    5. x=0
    6. z=1
    */
    
    // source terms
    Pfunc<ParamList, void(const Vector &, real_t, Vector &)> fu_fun;
    Pfunc<ParamList, void(const Vector &, real_t, Vector &)> fA_fun;
     
     // exact solutions
    TDVfunc u_fun;
    TDVfunc w_fun;
    TDSfunc p_fun;
    TDVfunc A_fun;
    TDVfunc B_fun;
    TDVfunc j_fun;
    TDVfunc curlw_fun;
    TDVfunc gradp_fun;
    TDSfunc divA_fun;
    TDVfunc curlj_fun;
    
    // periodic boundary conditions
    bool periodic = false;
    Vfunc Bstab_fun;
    
    // with exact solution
    ProblemData(ParamList param_,
                Vfunc u0_, Vfunc w0_, Sfunc p0_, Vfunc A0_, Vfunc B0_, Vfunc j0_,
                TDVfunc ubdry_, TDVfunc wbdry_, TDSfunc pbdry_, TDVfunc Bbdry_, 
                Array<int> ess_bdr_tangent_, Array<int> ess_bdr_normal_, Array<int> ess_bdr_magnetic_, 
                std::function<void(const Vector & xi, real_t t, Vector & f, ParamList param)> fu_,
                std::function<void(const Vector & xi, real_t t, Vector & f, ParamList param)> fA_,
                TDVfunc u_, TDVfunc w_, TDSfunc p_, TDVfunc A_, TDVfunc B_, TDVfunc j_,
                TDVfunc curlw_, TDVfunc gradp_, TDSfunc divA_, TDVfunc curlj_):
                param(param_),
                has_exact_solution(true),
                u0_fun(u0_),
                w0_fun(w0_),
                p0_fun(p0_),
                A0_fun(A0_),
                B0_fun(B0_),
                j0_fun(j0_),
                ubdry_fun(ubdry_),
                wbdry_fun(wbdry_),
                pbdry_fun(pbdry_),
                Bbdry_fun(Bbdry_),
                ess_bdr_tangent(ess_bdr_tangent_),
                ess_bdr_normal(ess_bdr_normal_),
                ess_bdr_magnetic(ess_bdr_magnetic_),
                fu_fun(fu_, param_),
                fA_fun(fA_, param_),
                u_fun(u_),
                w_fun(w_),
                p_fun(p_),
                A_fun(A_),
                B_fun(B_),
                j_fun(j_),
                curlw_fun(curlw_),
                gradp_fun(gradp_),
                divA_fun(divA_),
                curlj_fun(curlj_){}
                
    // without exact solution
    ProblemData(ParamList param_,
                Vfunc u0_, Vfunc w0_, Sfunc p0_, Vfunc A0_, Vfunc B0_, Vfunc j0_,
                TDVfunc ubdry_, TDVfunc wbdry_, TDSfunc pbdry_, TDVfunc Bbdry_,
                Array<int> ess_bdr_tangent_, Array<int> ess_bdr_normal_, Array<int> ess_bdr_magnetic_,
                std::function<void(const Vector & xi, real_t t, Vector & f, ParamList param)> fu_,
                std::function<void(const Vector & xi, real_t t, Vector & f, ParamList param)> fA_):
                param(param_),
                has_exact_solution(false),
                u0_fun(u0_),
                w0_fun(w0_),
                p0_fun(p0_),
                A0_fun(A0_),
                B0_fun(B0_),
                j0_fun(j0_),
                ubdry_fun(ubdry_),
                wbdry_fun(wbdry_),
                pbdry_fun(pbdry_),
                Bbdry_fun(Bbdry_),
                ess_bdr_tangent(ess_bdr_tangent_),
                ess_bdr_normal(ess_bdr_normal_),
                ess_bdr_magnetic(ess_bdr_magnetic_),
                fu_fun(fu_, param_),
                fA_fun(fA_, param_){}
                
                
    // set periodic boundary conditions
    void SetPeriodic(Vfunc Bstab_){
        periodic = true;
        Bstab_fun = Bstab_;
    }
};


// extract subvector from a big vector (from start to start+size)
// should be deleted by caller
Vector *ExtractVector(const Vector &x, Array<int> offsets, int start, int size=1);

Array<real_t> GetPlotTime(const char *time_file);

bool CheckPlot(const Array<real_t> &plot_time, real_t t);

void EvaluateGFAtPoint(const ParGridFunction &gf, const Vector &point, Vector &val);

#endif
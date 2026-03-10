
#include "Hall_MHD_dual.hpp"

#define _B0_ (1.0) 
#define _GAM_ (0.01)

typedef struct {
    real_t m;
} WhistlerParam;

#define u0_(u, x, y, z) {                            \
    u[0] = 0.0; \
    u[1] = 0.0;                      \
    u[2] = 0.0;            \
}

#define ubdry_(u, t, x, y, z) {                            \
    u[0] = 0.0; \
    u[1] = 0.0; \
    u[2] = 0.0; \
}

#define w0_(w, x, y, z) {                            \
    w[0] = 0.0; \
    w[1] = 0.0; \
    w[2] = 0.0; \
}

#define wbdry_(w, t, x, y, z) {                            \
    w[0] = 0.0; \
    w[1] = 0.0; \
    w[2] = 0.0; \
}

#define p0_(u, x, y, z) {                            \
    u = 0.0;                                 \
}

// unused
#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;                                 \
}

#define A0_(A, x, y, z){  \
    real_t m = param.m; \
    A[0] = _GAM_/(2.0*M_PI*m)*sin(2.0*M_PI*m*z); \
    A[1] = _GAM_/(2.0*M_PI*m)*cos(2.0*M_PI*m*z); \
    A[2] = 0.0;       \
}

#define Bstab_(B, x, y, z){  \
    B[0] = 0.0; \
    B[1] = 0.0; \
    B[2] = _B0_; \
}

#define B0_(B, x, y, z){  \
    real_t m = param.m; \
    B[0] = _GAM_*sin(2.0*M_PI*m*z); \
    B[1] = _GAM_*cos(2.0*M_PI*m*z); \
    B[2] = _B0_; \
}

#define Bbdry_(B, t, x, y, z){  \
    real_t m = param.m; \
    B[0] = 0.0; \
    B[1] = 0.0;  \
    B[2] = _B0_; \
}

#define j0_(j, x, y, z){  \
    real_t m = param.m; \
    j[0] = _GAM_*(2.0*M_PI*m)*sin(2.0*M_PI*m*z); \
    j[1] = _GAM_*(2.0*M_PI*m)*cos(2.0*M_PI*m*z); \
    j[2] = 0.0; \
}

#define fu_(f, t, x, y, z) {                            \
    f[0] = 0.0;                                 \
    f[1] = 0.0;                                 \
    f[2] = 0.0;                                 \
}

#define fA_(f, t, x, y, z){ \
    f[0] = 0.0;                                 \
    f[1] = 0.0;                                 \
    f[2] = 0.0;                             \
}

FUNCV_DEF(func_u0, u0_);
FUNCV_DEF(func_w0, w0_);
FUNCS_DEF(func_p0, p0_);
FUNCV_DEF(func_Bstab, Bstab_);

static void func_A0_(const Vector &xi, Vector &A, WhistlerParam param)
{
    real_t x(xi(0)), y(xi(1)), z(xi(2));
    A0_(A, x, y, z);
}
Pfunc<WhistlerParam, void(const Vector &, Vector &)> func_A0(func_A0_, WhistlerParam{1.0});

static void func_B0_(const Vector &xi, Vector &B, WhistlerParam param)
{
    real_t x(xi(0)), y(xi(1)), z(xi(2));
    B0_(B, x, y, z);
}
Pfunc<WhistlerParam, void(const Vector &, Vector &)> func_B0(func_B0_, WhistlerParam{1.0});

static void func_J0_(const Vector &xi, Vector &j, WhistlerParam param)
{
    real_t x(xi(0)), y(xi(1)), z(xi(2));
    j0_(j, x, y, z);
}
Pfunc<WhistlerParam, void(const Vector &, Vector &)> func_j0(func_J0_, WhistlerParam{1.0});

static void func_Bbdry_(const Vector &xi, real_t t, Vector &B, WhistlerParam param)
{
    real_t x(xi(0)), y(xi(1)), z(xi(2));
    Bbdry_(B, t, x, y, z);
}
Pfunc<WhistlerParam, void(const Vector &, real_t , Vector &)> func_Bbdry(func_Bbdry_, WhistlerParam{1.0});

FUNCV_TD_DEF(func_ubdry, ubdry_);
FUNCV_TD_DEF(func_wbdry, wbdry_);
FUNCS_TD_DEF(func_pbdry, pbdry_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetWhistlerProblemData(ParamList param, real_t m_)
{
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({0,0,0,0,0,0});
    Array<int> ess_bdr_magnetic({0,0,0,0,0,0});
    func_A0.SetParam(WhistlerParam{m_});
    func_B0.SetParam(WhistlerParam{m_});
    func_j0.SetParam(WhistlerParam{m_});
    func_Bbdry.SetParam(WhistlerParam{m_});
    ProblemData *pd = new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
        func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
        ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
        func_fu, func_fA);
    pd->SetPeriodic(func_Bstab);
    return pd;
}
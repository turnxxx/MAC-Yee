
#include "Hall_MHD_dual.hpp"

static real_t nu_ = 0.0;
    
#define u_(u, t, x, y, z) {                            \
    u[0] = sin(x)*cos(y)*exp(-2*nu_*t);                           \
    u[1] = -cos(x)*sin(y)*exp(-2*nu_*t);                      \
    u[2] = 0.0;                           \
}
    
#define u0_(u, x, y, z) {                            \
    u[0] = sin(x)*cos(y);                           \
    u[1] = -cos(x)*sin(y);                      \
    u[2] = 0.0;                           \
}

#define gradu_(u, t, x, y, z) {   \
    u[0] = cos(x)*cos(y)*exp(-2*nu_*t); \
    u[1] = -sin(x)*sin(y)*exp(-2*nu_*t); \
    u[2] = 0.0; \
    u[3] = sin(x)*sin(y)*exp(-2*nu_*t); \
    u[4] = -cos(x)*cos(y)*exp(-2*nu_*t); \
    u[5] = 0.0; \
    u[6] = 0.0; \
    u[7] = 0.0; \
    u[8] = 0.0; \
}

#define uLaplace_(u, t, x, y, z) {                     \
    u[0] = -2.0*sin(x)*cos(y)*exp(-2*nu_*t);  \
    u[1] = 2.0*cos(x)*sin(y)*exp(-2*nu_*t);   \
    u[2] = 0.0;  \
}

#define w_(w, t, x, y, z) {                        \
    real_t gradu[9];                            \
    gradu_(gradu, t, x, y, z);                  \
    w[0] = gradu[7] - gradu[5];     \
    w[1] = gradu[2] - gradu[6];     \
    w[2] = gradu[3] - gradu[1];     \
}

#define curlw_(w, t, x, y, z){      \
    real_t ulap[3];                 \
    uLaplace_(ulap, t, x, y, z);    \
    w[0] = -ulap[0];                \
    w[1] = -ulap[1];                \
    w[2] = -ulap[2];                \
}

#define w0_(w, x, y, z) {                            \
    real_t t = 0.0;             \
    w_(w, t, x, y, z);          \
}

#define p_(u, t, x, y, z) {                            \
    u = 0.0;   \
}

#define p0_(u, x, y, z) {                            \
    u = 0.0;   \
}

#define gradp_(u, t, x, y, z) {      \
    u[0] = 0.0;      \
    u[1] = 0.0;       \
    u[2] = 0.0;       \
}

#define A_(A, t, x, y, z){  \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = sin(x)*sin(y)*exp(-2*nu_*t); \
}

#define divA_(u, t, x, y, z){   \
    u = 0.0; \
}

#define A0_(A, x, y, z){  \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = sin(x)*sin(y); \
}

#define fu_(f, t, x, y, z) {          \
    f[0] = 0.0;      \
    f[1] = 0.0;      \
    f[2] = 0.0;      \
}

#define fA_(f, t, x, y, z){ \
    f[0] = 0.0; \
    f[1] = 0.0; \
    f[2] = 0.0; \
}

FUNCV_TD_DEF(func_u, u_);

FUNCV_TD_DEF(func_w, w_);
FUNCV_TD_DEF(func_curlw, curlw_);

FUNCS_TD_DEF(func_p, p_);
FUNCV_TD_DEF(func_gradp, gradp_);

FUNCV_TD_DEF(func_A, A_);
FUNCS_TD_DEF(func_divA, divA_);

FUNCV_TD_DEF(func_B, u_);
FUNCV_TD_DEF(func_j, w_);
FUNCV_TD_DEF(func_curlj, curlw_);

FUNCV_DEF(func_u0, u0_);
FUNCV_DEF(func_w0, w0_);
FUNCS_DEF(func_p0, p0_);
FUNCV_DEF(func_A0, A0_);
FUNCV_DEF(func_B0, u0_);
FUNCV_DEF(func_j0, w0_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetTGVORTEXProblemData(ParamList param, bool viscosity)
{
    nu_ = viscosity ? 1.0/param.Re : 0.0;
    // periodic boundary conditions
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({0,0,0,0,0,0});
    Array<int> ess_bdr_magnetic({0,0,0,0,0,0});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_u, func_w, func_p, func_B,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA,
     func_u, func_w, func_p, func_A, func_B, func_j, func_curlw, func_gradp, func_divA, func_curlj);
}
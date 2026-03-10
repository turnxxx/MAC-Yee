
#include "Hall_MHD_dual.hpp"

static real_t visc_ = 0.0;
static real_t resist_ = 0.0;
static real_t Hall_ = 0.0;

#define F_(x) (0.5*sin(M_PI*(x))*sin(M_PI*(x))/M_PI)
#define dF_(x) (0.5*sin(2.*M_PI*(x)))
#define ddF_(x) (M_PI*cos(2.0*M_PI*(x)))
#define dddF_(x) (-2.*M_PI*M_PI*sin(2.0*M_PI*(x)))

    
#define u_(u, t, x, y, z) {                            \
    u[0] = (F_(x)*dF_(y)*F_(z) - F_(x)*F_(y)*dF_(z))*(1.0-t);                           \
    u[1] = (F_(x)*F_(y)*dF_(z) - dF_(x)*F_(y)*F_(z))*(1.0-t);                      \
    u[2] = (dF_(x)*F_(y)*F_(z) - F_(x)*dF_(y)*F_(z))*(1.0-t);                           \
}

#define u0_(u, x, y, z) {                            \
    u[0] = (F_(x)*dF_(y)*F_(z) - F_(x)*F_(y)*dF_(z));                           \
    u[1] = (F_(x)*F_(y)*dF_(z) - dF_(x)*F_(y)*F_(z));                      \
    u[2] = (dF_(x)*F_(y)*F_(z) - F_(x)*dF_(y)*F_(z));                           \
}

#define uLaplace_(u, t, x, y, z) {                     \
    u[0] = (1.0-t)*(ddF_(x)*dF_(y)*F_(z) + F_(x)*dddF_(y)*F_(z) + F_(x)*dF_(y)*ddF_(z) - ddF_(x)*F_(y)*dF_(z) - F_(x)*ddF_(y)*dF_(z) - F_(x)*F_(y)*dddF_(z));  \
    u[1] = (1.0-t)*(ddF_(x)*F_(y)*dF_(z) + F_(x)*ddF_(y)*dF_(z) + F_(x)*F_(y)*dddF_(z) - dddF_(x)*F_(y)*F_(z) - dF_(x)*ddF_(y)*F_(z) - dF_(x)*F_(y)*ddF_(z));   \
    u[2] = (1.0-t)*(dddF_(x)*F_(y)*F_(z) + dF_(x)*ddF_(y)*F_(z) + dF_(x)*F_(y)*ddF_(z) - ddF_(x)*dF_(y)*F_(z) - F_(x)*dddF_(y)*F_(z) - F_(x)*dF_(y)*ddF_(z));  \
}
    
#define ut_(u, t, x, y, z) {                           \
    u[0] = -(F_(x)*dF_(y)*F_(z) - F_(x)*F_(y)*dF_(z));                           \
    u[1] = -(F_(x)*F_(y)*dF_(z) - dF_(x)*F_(y)*F_(z));                      \
    u[2] = -(dF_(x)*F_(y)*F_(z) - F_(x)*dF_(y)*F_(z));                           \
}

#define gradu_(u, t, x, y, z) {                           \
    u[0] = (1.0-t)*(dF_(x)*dF_(y)*F_(z) - dF_(x)*F_(y)*dF_(z)); \
    u[1] = (1.0-t)*(F_(x)*ddF_(y)*F_(z) - F_(x)*dF_(y)*dF_(z)); \
    u[2] = (1.0-t)*(F_(x)*dF_(y)*dF_(z) - F_(x)*F_(y)*ddF_(z)); \
    u[3] = (1.0-t)*(dF_(x)*F_(y)*dF_(z) - ddF_(x)*F_(y)*F_(z)); \
    u[4] = (1.0-t)*(F_(x)*dF_(y)*dF_(z) - dF_(x)*dF_(y)*F_(z)); \
    u[5] = (1.0-t)*(F_(x)*F_(y)*ddF_(z) - dF_(x)*F_(y)*dF_(z)); \
    u[6] = (1.0-t)*(ddF_(x)*F_(y)*F_(z) - dF_(x)*dF_(y)*F_(z)); \
    u[7] = (1.0-t)*(dF_(x)*dF_(y)*F_(z) - F_(x)*ddF_(y)*F_(z)); \
    u[8] = (1.0-t)*(dF_(x)*F_(y)*dF_(z) - F_(x)*dF_(y)*dF_(z)); \
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
    u = sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z)*(1.0-t);                                 \
}

#define p0_(u, x, y, z) {                            \
    u = sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);                                 \
}

#define gradp_(u, t, x, y, z) {                            \
    u[0] = 2.*M_PI*cos(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z)*(1.0-t);                                 \
    u[1] = 2.*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z)*(1.0-t);                                 \
    u[2] = 2.*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z)*(1.0-t);                                 \
}

#define A_(A, t, x, y, z){  \
    A[0] = 0.5*sin(x)*(cos(y) - cos(z))*(1.-t); \
    A[1] = 0.5*sin(y)*(cos(z) - cos(x))*(1.-t); \
    A[2] = 0.5*sin(z)*(cos(x) - cos(y))*(1.-t); \
}

#define divA_(u, t, x, y, z){   \
    u = 0.0; \
}

#define A0_(A, x, y, z){  \
    A[0] = 0.5*sin(x)*(cos(y) - cos(z)); \
    A[1] = 0.5*sin(y)*(cos(z) - cos(x)); \
    A[2] = 0.5*sin(z)*(cos(x) - cos(y)); \
}

#define At_(A, t, x, y, z){  \
    A[0] = -0.5*sin(x)*(cos(y) - cos(z)); \
    A[1] = -0.5*sin(y)*(cos(z) - cos(x)); \
    A[2] = -0.5*sin(z)*(cos(x) - cos(y)); \
}

#define B_(B, t, x, y, z){  \
    B[0] = sin(y)*sin(z)*(1.0-t); \
    B[1] = sin(z)*sin(x)*(1.0-t); \
    B[2] = sin(x)*sin(y)*(1.0-t); \
}

#define B0_(B, x, y, z){  \
    B[0] = sin(y)*sin(z); \
    B[1] = sin(z)*sin(x); \
    B[2] = sin(x)*sin(y); \
}

#define uxB_(uxB, t, x, y, z){ \
    real_t uu[3]; \
    u_(uu, t, x, y, z); \
    real_t B[3]; \
    B_(B, t, x, y, z); \
    uxB[0] = uu[1]*B[2] - uu[2]*B[1]; \
    uxB[1] = uu[2]*B[0] - uu[0]*B[2]; \
    uxB[2] = uu[0]*B[1] - uu[1]*B[0]; \
}

#define jxB_(jxB, t, x, y, z){ \
    real_t j[3]; \
    j_(j, t, x, y, z); \
    real_t B[3]; \
    B_(B, t, x, y, z); \
    jxB[0] = j[1]*B[2] - j[2]*B[1]; \
    jxB[1] = j[2]*B[0] - j[0]*B[2]; \
    jxB[2] = j[0]*B[1] - j[1]*B[0]; \
}

#define j_(j, t, x, y, z){  \
    j[0] = (1.-t)*sin(x)*(cos(y)-cos(z)); \
    j[1] = (1.-t)*sin(y)*(cos(z)-cos(x)); \
    j[2] = (1.-t)*sin(z)*(cos(x)-cos(y)); \
}

#define curlj_(j, t, x, y, z){  \
    j[0] = 2.0*sin(y)*sin(z)*(1.0-t); \
    j[1] = 2.0*sin(z)*sin(x)*(1.0-t); \
    j[2] = 2.0*sin(x)*sin(y)*(1.0-t); \
}

#define j0_(j, x, y, z){  \
    j[0] = sin(x)*(cos(y)-cos(z)); \
    j[1] = sin(y)*(cos(z)-cos(x)); \
    j[2] = sin(z)*(cos(x)-cos(y)); \
}

#define fu_(f, t, x, y, z) {                            \
    real_t ut[3];                               \
    ut_(ut, t, x, y, z);                        \
    real_t ulaplace[3];                         \
    uLaplace_(ulaplace, t, x, y, z);            \
    real_t gradp[3];                            \
    gradp_(gradp, t, x, y, z);                  \
    real_t w[3];                                \
    w_(w, t, x, y, z);                          \
    real_t uu[3];                                \
    u_(uu, t, x, y, z);                          \
    real_t j[3];                                \
    j_(j, t, x, y, z);                          \
    real_t B[3];                                \
    B_(B, t, x, y, z);                          \
    f[0] = ut[0]                                \
    - visc_*ulaplace[0]                        \
    + w[1]*uu[2] - w[2]*uu[1]                   \
    + gradp[0]                               \
    - s*(j[1]*B[2] - j[2]*B[1]);               \
    f[1] = ut[1]                                \
    - visc_*ulaplace[1]                        \
    + w[2]*uu[0] - w[0]*uu[2]                   \
    + gradp[1]                                 \
    - s*(j[2]*B[0] - j[0]*B[2]);               \
    f[2] = ut[2]                                \
    - visc_*ulaplace[2]                        \
    + w[0]*uu[1] - w[1]*uu[0]                   \
    + gradp[2]                                 \
    - s*(j[0]*B[1] - j[1]*B[0]);               \
}

#define fA_(f, t, x, y, z){ \
    real_t At[3]; \
    At_(At, t, x, y, z); \
    real_t j[3]; \
    j_(j, t, x, y, z); \
    real_t uxB[3]; \
    uxB_(uxB, t, x, y, z); \
    real_t jxB[3]; \
    jxB_(jxB, t, x, y, z); \
    f[0] = At[0] + resist_*j[0] - uxB[0] + Hall_*jxB[0]; \
    f[1] = At[1] + resist_*j[1] - uxB[1] + Hall_*jxB[1]; \
    f[2] = At[2] + resist_*j[2] - uxB[2] + Hall_*jxB[2]; \
}

FUNCV_TD_DEF(func_u, u_);
FUNCV_TD_DEF(func_ut, ut_);
FUNCV_TD_DEF(func_uLaplace, uLaplace_);

FUNCV_TD_DEF(func_w, w_);
FUNCV_TD_DEF(func_curlw, curlw_);

FUNCS_TD_DEF(func_p, p_);
FUNCV_TD_DEF(func_gradp, gradp_);

FUNCV_TD_DEF(func_A, A_);
FUNCS_TD_DEF(func_divA, divA_);

FUNCV_TD_DEF(func_B, B_);
FUNCV_TD_DEF(func_j, j_);
FUNCV_TD_DEF(func_curlj, curlj_);

FUNCV_DEF(func_u0, u0_);
FUNCV_DEF(func_w0, w0_);
FUNCS_DEF(func_p0, p0_);
FUNCV_DEF(func_A0, A0_);
FUNCV_DEF(func_B0, B0_);
FUNCV_DEF(func_j0, j0_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetSpatialProblemData(ParamList param, bool Hall, bool viscosity, bool resistivity)
{
    Hall_ = Hall?param.RH:0.0;
    visc_ = viscosity?1.0/param.Re:0.0;
    resist_ = resistivity?1.0/param.Rm:0.0; 
    
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({1,1,1,1,1,1});
    Array<int> ess_bdr_magnetic({1,1,1,1,1,1});
    // Array<int> ess_bdr_normal({0,0,0,0,0,0});
    // Array<int> ess_bdr_magnetic({0,0,0,0,0,0});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_u, func_w, func_p, func_B,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA,
     func_u, func_w, func_p, func_A, func_B, func_j, func_curlw, func_gradp, func_divA, func_curlj);
}
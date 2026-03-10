
#include "Hall_MHD_dual.hpp"

static real_t X0=5.0;
static real_t Y0=5.0;
    
#define u_(u, t, x, y, z) {                            \
    real_t r = sqrt((x-X0)*(x-X0) + (y-Y0)*(y-Y0)); \
    u[0] = exp(0.5*(1-r*r))*(Y0-y);                           \
    u[1] = exp(0.5*(1-r*r))*(x-X0);                      \
    u[2] = 0.0;                           \
}
    
#define u0_(u, x, y, z) {                            \
    real_t t = 0.0;             \
    u_(u, t, x, y, z);          \
}

#define gradu_(u, t, x, y, z) {   \
    u[0] = exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(X0-x)*(Y0-y); \
    u[1] = -exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(Y0*y*2.0-Y0*Y0-y*y+1.0); \
    u[2] = 0.0; \
    u[3] = exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(X0*x*2.0-X0*X0-x*x+1.0); \
    u[4] = -exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(X0-x)*(Y0-y); \
    u[5] = 0.0; \
    u[6] = 0.0; \
    u[7] = 0.0; \
    u[8] = 0.0; \
}

#define uLaplace_(u, t, x, y, z) {                     \
    u[0] = exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(Y0-y)*(X0*x*-2.0-Y0*y*2.0+X0*X0+Y0*Y0+x*x+y*y-4.0);  \
    u[1] = -exp(pow(X0-x,2.0)*(-1.0/2.0)-pow(Y0-y,2.0)/2.0+1.0/2.0)*(X0-x)*(X0*x*-2.0-Y0*y*2.0+X0*X0+Y0*Y0+x*x+y*y-4.0);   \
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
    real_t t = 0.0;             \
    p_(u, t, x, y, z);          \
}

#define gradp_(u, t, x, y, z) {      \
    u[0] = 0.0;      \
    u[1] = 0.0; \
    u[2] = 0.0;       \
}

// #define p_(u, t, x, y, z) {                            \
//     real_t r = sqrt((x-U0*t-X0)*(x-U0*t-X0) + (y-U0*t-Y0)*(y-U0*t-Y0)); \
//     u = 1.0+0.5*exp(1.0)*(1-r*r*exp(-r*r));   \
//     real_t uu[3]; \
//     u_(uu, t, x, y, z); \
//     u += 0.5*(uu[0]*uu[0]+uu[1]*uu[1]+uu[2]*uu[2]); \
//     u -= 2.359140914229523; \
// }

// #define p0_(u, x, y, z) {                            \
//     real_t t = 0.0;             \
//     p_(u, t, x, y, z);          \
// }

// #define gradp_(u, t, x, y, z) {      \
//     u[0] = exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*(X0*2.0-x*2.0)*(-1.0/2.0)+exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0))*(X0*2.0-x*2.0+U0*t*2.0)*1.359140914229523+(exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*pow(X0-x,2.0)*(X0*2.0-x*2.0+U0*t*2.0))/2.0+(exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*pow(Y0-y,2.0)*(X0*2.0-x*2.0+U0*t*2.0))/2.0-exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0))*(pow(X0-x+U0*t,2.0)+pow(Y0-y+U0*t,2.0))*(X0*2.0-x*2.0+U0*t*2.0)*1.359140914229523;      \
//     u[1] = exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*(Y0*2.0-y*2.0)*(-1.0/2.0)+exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0))*(Y0*2.0-y*2.0+U0*t*2.0)*1.359140914229523+(exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*pow(X0-x,2.0)*(Y0*2.0-y*2.0+U0*t*2.0))/2.0+(exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0)+1.0)*pow(Y0-y,2.0)*(Y0*2.0-y*2.0+U0*t*2.0))/2.0-exp(-pow(X0-x+U0*t,2.0)-pow(Y0-y+U0*t,2.0))*(pow(X0-x+U0*t,2.0)+pow(Y0-y+U0*t,2.0))*(Y0*2.0-y*2.0+U0*t*2.0)*1.359140914229523; \
//     u[2] = 0.0;       \
// }

#define A_(A, t, x, y, z){  \
    real_t r = sqrt((x-X0)*(x-X0) + (y-Y0)*(y-Y0)); \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = exp(0.5*(1.0-r*r)); \
}

#define divA_(u, t, x, y, z){   \
    u = 0.0; \
}

#define A0_(A, x, y, z){  \
    real_t r = sqrt((x-X0)*(x-X0) + (y-Y0)*(y-Y0)); \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = exp(0.5*(1.0-r*r)); \
}

#define B_(B, t, x, y, z){  \
    real_t r = sqrt((x-X0)*(x-X0) + (y-Y0)*(y-Y0)); \
    B[0] = exp(0.5*(1.0-r*r))*(Y0-y); \
    B[1] = exp(0.5*(1.0-r*r))*(x-X0); \
    B[2] = 0.0; \
}

#define B0_(B, x, y, z){  \
    real_t r = sqrt((x-X0)*(x-X0) + (y-Y0)*(y-Y0)); \
    B[0] = exp(0.5*(1.0-r*r))*(Y0-y); \
    B[1] = exp(0.5*(1.0-r*r))*(x-X0); \
    B[2] = 0.0; \
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

FUNCV_TD_DEF(func_B, B_);
FUNCV_TD_DEF(func_j, w_);
FUNCV_TD_DEF(func_curlj, curlw_);

FUNCV_DEF(func_u0, u0_);
FUNCV_DEF(func_w0, w0_);
FUNCS_DEF(func_p0, p0_);
FUNCV_DEF(func_A0, A0_);
FUNCV_DEF(func_B0, B0_);
FUNCV_DEF(func_j0, w0_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetMHDVORTEXProblemData(ParamList param)
{
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
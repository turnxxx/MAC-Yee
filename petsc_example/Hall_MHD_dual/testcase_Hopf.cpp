
#include "Hall_MHD_dual.hpp"

// static real_t s_ = 1.0;
// static real_t w1_ = 3.0;
// static real_t w2_ = 2.0;
    
#define u0_(u, x, y, z) {                            \
    u[0] = 0.0;                           \
    u[1] = 0.0;                      \
    u[2] = 0.0;                           \
}

#define ubdry_(u, t, x, y, z) {                            \
    u[0] = 0.0;                           \
    u[1] = 0.0;                      \
    u[2] = 0.0;                           \
}

#define w0_(w, x, y, z) {                            \
    w[0] = 0.0;                           \
    w[1] = 0.0;                      \
    w[2] = 0.0;                           \
}

#define wbdry_(w, t, x, y, z) {                            \
    w[0] = 0.0;                           \
    w[1] = 0.0;                      \
    w[2] = 0.0;                           \
}


#define p0_(u, x, y, z) {                            \
    u = 0.0;   \
}

#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;   \
}

#define A0_(A, x, y, z){  \
    real_t r = sqrt(x*x + y*y + z*z); \
    A[0] = (x*z-y)/(2.0*pow(1.0+r*r,2)); \
    A[1] = (y*z+x)/(2.0*pow(1.0+r*r,2)); \
    A[2] = (2.0*z*z+1.0-r*r)/(4.0*pow(1.0+r*r,2)); \
}

#define B0_(B, x, y, z){  \
    real_t r = sqrt(x*x + y*y + z*z); \
    B[0] = 2.0*(x*z-y)/pow(1.0+r*r,3); \
    B[1] = 2.0*(y*z+x)/pow(1.0+r*r,3); \
    B[2] = (2.0*z*z+1.0-r*r)/pow(1.0+r*r,3); \
}

#define Bbdry_(B, t, x, y, z){  \
    real_t r = sqrt(x*x + y*y + z*z); \
    B[0] = 2.0*(x*z-y)/pow(1.0+r*r,3); \
    B[1] = 2.0*(y*z+x)/pow(1.0+r*r,3); \
    B[2] = (2.0*z*z+1.0-r*r)/pow(1.0+r*r,3); \
}

#define j0_(j, x, y, z){  \
    j[0] = 1.0/pow(x*x+y*y+z*z+1.0,4.0)*(y*-5.0+x*z*6.0+(x*x)*y+y*(z*z)+y*y*y)*2.0; \
    j[1] = 1.0/pow(x*x+y*y+z*z+1.0,4.0)*(x*-5.0-y*z*6.0+x*(y*y)+x*(z*z)+x*x*x)*-2.0; \
    j[2] = ((x*x)*2.0+(y*y)*2.0-z*z-1.0)*1.0/pow(x*x+y*y+z*z+1.0,4.0)*-4.0; \
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

FUNCV_DEF(func_u0, u0_);
FUNCV_DEF(func_w0, w0_);
FUNCS_DEF(func_p0, p0_);
FUNCV_DEF(func_A0, A0_);
FUNCV_DEF(func_B0, B0_);
FUNCV_DEF(func_j0, j0_);

FUNCV_TD_DEF(func_ubdry, ubdry_);
FUNCV_TD_DEF(func_wbdry, wbdry_);
FUNCS_TD_DEF(func_pbdry, pbdry_);
FUNCV_TD_DEF(func_Bbdry, Bbdry_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetHopfProblemData(ParamList param)
{
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({1,1,1,1,1,1});
    Array<int> ess_bdr_magnetic({1,1,1,1,1,1});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
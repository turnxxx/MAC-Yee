
#include "Hall_MHD_dual.hpp"
    
#define u0_(u, x, y, z) {                            \
    u[0] = 2.0*cos(y);                           \
    u[1] = -2.0*sin(x);                      \
    u[2] = 0.0;                           \
}

#define ubdry_(u, t, x, y, z) {                            \
    u[0] = 2.0*cos(y);                           \
    u[1] = -2.0*sin(x);                      \
    u[2] = 0.0;                           \
}

#define w0_(w, x, y, z) {                            \
    w[0] = 0.0;                           \
    w[1] = 0.0;                      \
    w[2] = -2.0*cos(x)+2.0*sin(y);                           \
}

#define wbdry_(w, t, x, y, z) {                            \
    w[0] = 0.0;                           \
    w[1] = 0.0;                      \
    w[2] = -2.0*cos(x)+2.0*sin(y);                           \
}


#define p0_(u, x, y, z) {                            \
    u = 0.0;   \
}

#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;   \
}

#define A0_(A, x, y, z){  \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = cos(2.0*y)-2.0*cos(x); \
}

#define B0_(B, x, y, z){  \
    B[0] = -2.0*sin(2.0*y); \
    B[1] = -2.0*sin(x); \
    B[2] = 0.0; \
}

#define Bbdry_(B, t, x, y, z){  \
    B[0] = -2.0*sin(2.0*y); \
    B[1] = -2.0*sin(x); \
    B[2] = 0.0; \
}

#define j0_(j, x, y, z){  \
    j[0] = 0.0; \
    j[1] = 0.0; \
    j[2] = -2.0*cos(x)+4.0*cos(2.0*y); \
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

ProblemData *GetOrszagTangKrausProblemData(ParamList param)
{
    // periodic boundary conditions
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({0,0,0,0,0,0});
    Array<int> ess_bdr_magnetic({0,0,0,0,0,0});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
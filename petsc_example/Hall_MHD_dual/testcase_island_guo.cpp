
#include "Hall_MHD_dual.hpp"

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
    u = cos(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);                                 \
}

// unused
#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;                                 \
}

#define A0_(A, x, y, z){  \
    A[0] = 0.0; \
    A[1] = 2.5/M_PI*cos(M_PI*x); \
    A[2] = 0.0;       \
}

#define B0_(B, x, y, z){  \
    B[0] = 0.0; \
    B[1] = 0.0; \
    B[2] = -2.5*sin(M_PI*x); \
}

#define Bbdry_(B, t, x, y, z){  \
    B[0] = 0.0; \
    B[1] = 0.0;  \
    B[2] = -2.5*sin(M_PI*x); \
}

#define j0_(j, x, y, z){  \
    j[0] = 0.0; \
    j[1] = 2.5*M_PI*cos(M_PI*x); \
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
FUNCV_DEF(func_A0, A0_);
FUNCV_DEF(func_B0, B0_);
FUNCV_DEF(func_j0, j0_);

FUNCV_TD_DEF(func_ubdry, ubdry_);
FUNCV_TD_DEF(func_Bbdry, Bbdry_);
FUNCV_TD_DEF(func_wbdry, wbdry_);
FUNCS_TD_DEF(func_pbdry, pbdry_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetIslandGuoProblemData(ParamList param)
{
    Array<int> ess_bdr_tangent;
    Array<int> ess_bdr_normal({1,1,1,1,1,1});
    Array<int> ess_bdr_magnetic({1,1,1,1,1,1});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
        ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
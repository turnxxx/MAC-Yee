
#include "Hall_MHD_dual.hpp"

#define _DELTA_ (0.07957747154595)
#define _BX_ (1.0)
#define _CN_ (0.001)

#define u0_(u, x, y, z) {                            \
    u[0] = 1.5*tanh((2.0*z-1.0)*28.0) \
    -_CN_*2.0*(z-0.5)*28.*28.*exp(-(z-0.5)*(z-0.5)*28.*28.)*cos(5.0*M_PI*x); \
    u[1] = 0.0;                      \
    u[2] = _CN_*5.0*M_PI*exp(-(z-0.5)*(z-0.5)*28.*28.)*sin(5.0*M_PI*x);            \
}

#define ubdry_(u, t, x, y, z) {                            \
    u[0] = 0.0; \
    u[1] = 0.0; \
    u[2] = 0.0; \
}

#define w0_(w, x, y, z) {                            \
    w[0] = 0.0; \
    w[1] = 56.*1.5/pow(cosh(28.*(2.*z-1.)),2); \
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

#define A0_(A, x, y, z){  \
    A[0] = 0.0; \
    A[1] = -z; \
    A[2] = 0.0; \
}

#define B0_(B, x, y, z){  \
    B[0] = 1.0; \
    B[1] = 0.0; \
    B[2] = 0.0; \
}

#define Bbdry_(B, t, x, y, z){  \
    B0_(B, x, y, z);                            \
}

#define j0_(j, x, y, z){  \
    j[0] = 0.0; \
    j[1] = 0.0; \
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
    f[2] = 0.0;                                 \
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

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetKHProblemData(ParamList param)
{
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_Bbdry,
     func_fu, func_fA);
}
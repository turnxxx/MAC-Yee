
#include "Hall_MHD_dual.hpp"

#define _DELTA_ (1.0/(2.0*M_PI))
#define _EPSILON_ (0.2)
#define _GAM_ (0.01)

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
    u = (1.0-_EPSILON_*_EPSILON_)/2.0*(1.0+1.0/pow(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_),2));                                 \
}

// unused
#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;                                 \
}

#define A0_(A, x, y, z){  \
    A[0] = 0.0; \
    A[1] = 0.0; \
    A[2] = _DELTA_*log(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_)) \
    +2.0*_GAM_/M_PI/M_PI*cos(M_PI*x)*cos(0.5*M_PI*y);       \
}

#define B0_(B, x, y, z){  \
    B[0] = sinh(y/_DELTA_)/(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_)) \
         - _GAM_/M_PI*cos(M_PI*x)*sin(0.5*M_PI*y); \
    B[1] = _EPSILON_*sin(x/_DELTA_)/(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_))  \
    + _GAM_/M_PI*2.0*sin(M_PI*x)*cos(0.5*M_PI*y); \
    B[2] = 0.0; \
}

#define Bbdry_(B, t, x, y, z){  \
    B[0] = sinh(y/_DELTA_)/(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_)); \
    B[1] = _EPSILON_*sin(x/_DELTA_)/(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_));  \
    B[2] = 0.0; \
}

#define j0_(j, x, y, z){  \
    j[0] = 0.0; \
    j[1] = 0.0; \
    j[2] = (_EPSILON_*_EPSILON_-1.0)/(_DELTA_*pow(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_),2)) ; \
}

#define fu_(f, t, x, y, z) {                            \
    f[0] = 0.0;                                 \
    f[1] = 0.0;                                 \
    f[2] = 0.0;                                 \
}

#define fA_(f, t, x, y, z){ \
    f[0] = 0.0;                                 \
    f[1] = 0.0;                                 \
    f[2] = 1.0/Rm*(_EPSILON_*_EPSILON_-1.0)/(_DELTA_*pow(cosh(y/_DELTA_)+_EPSILON_*cos(x/_DELTA_),2));                             \
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

ProblemData *GetIslandProblemData(ParamList param)
{
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({0,1,1,1,1,0});
    Array<int> ess_bdr_magnetic({0,1,1,1,1,0});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
        ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
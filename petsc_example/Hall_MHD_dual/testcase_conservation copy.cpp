
#include "Hall_MHD_dual.hpp"

#define F_(x) (32.0*(x*x*x*(x-1.0)*(x-1.0)*(x-1.0)))
#define dF_(x) (32.0*(3.*x*x*(x-1.0)*(x-1.0)*(x-1.0) + x*x*x*3.*(x-1.0)*(x-1.0)))
#define ddF_(x) (32.0*(6.*x*(x-1.0)*(x-1.0)*(x-1.0) + 9.*x*x*(x-1.0)*(x-1.0) + 9.*x*x*(x-1.0)*(x-1.0) + 6.*x*x*x*(x-1.0)))

#define ubdry_(u, t, x, y, z) {                            \
    u[0] = 0.0;                                 \
    u[1] = 0.0;                                 \
    u[2] = 0.0;                                 \
}

#define _UX_ (1.0)
#define _UY_ (1.0)
#define _UZ_ (1.0)

#define _ALPHA_(u, x, y, z) {                            \
    u[0] = F_(x)*F_(y)*F_(z);                                 \
}

#define _NABLA_ALPHA_(u, x, y, z) {                            \
    u[0] = dF_(x)*F_(y)*F_(z);                                 \
    u[1] = F_(x)*dF_(y)*F_(z);                                 \
    u[2] = F_(x)*F_(y)*dF_(z);                                 \
}

#define _BETA_(u, x, y, z) {                            \
    u[0] = y;                                 \
    u[1] = -x;                                 \
    u[2] = 1;                                 \
}

#define _CURL_BETA_(u, x, y, z) {                            \
    u[0] = 0.0;                                 \
    u[1] = 0.0;                                 \
    u[2] = -2.0;                                 \
}

#define _CURL_NABLAALPHA_TIMES_BETA_(u, x, y, z) {                            \
    u[0] = -y*(ddF_(x)*F_(y)*F_(z)+F_(x)*ddF_(y)*F_(z)+F_(x)*F_(y)*ddF_(z)) \
    +y*(ddF_(x)*F_(y)*F_(z))-x*(dF_(x)*dF_(y)*F_(z)) + (dF_(x)*F_(y)*dF_(z)) \
    -(F_(x)*dF_(y)*F_(z)) \
    ;                                 \
    u[1] = x*(ddF_(x)*F_(y)*F_(z)+F_(x)*ddF_(y)*F_(z)+F_(x)*F_(y)*ddF_(z))  \
    +y*(dF_(x)*dF_(y)*F_(z)) - x*(F_(x)*ddF_(y)*F_(z)) + (F_(x)*dF_(y)*dF_(z)) \
    +(dF_(x)*F_(y)*F_(z)) \
    ;                                 \
    u[2] = -1.0*(ddF_(x)*F_(y)*F_(z)+F_(x)*ddF_(y)*F_(z)+F_(x)*F_(y)*ddF_(z)) \
    +y*(dF_(x)*F_(y)*dF_(z)) - x*(F_(x)*dF_(y)*dF_(z)) + (F_(x)*F_(y)*ddF_(z)) \
    ;                                 \
}

#define u0_(u, x, y, z) {                            \
    real_t alpha[1];                            \
    _ALPHA_(alpha, x, y, z);                            \
    real_t nabla_alpha[3];                            \
    _NABLA_ALPHA_(nabla_alpha, x, y, z);                            \
    real_t beta[3];                            \
    _BETA_(beta, x, y, z);                            \
    real_t curl_beta[3];                            \
    _CURL_BETA_(curl_beta, x, y, z);                            \
    u[0] = nabla_alpha[1]*beta[2] - nabla_alpha[2]*beta[1] + alpha[0]*curl_beta[0];                           \
    u[1] = nabla_alpha[2]*beta[0] - nabla_alpha[0]*beta[2] + alpha[0]*curl_beta[1];                           \
    u[2] = nabla_alpha[0]*beta[1] - nabla_alpha[1]*beta[0] + alpha[0]*curl_beta[2];                           \
}

#define wbdry_(w, t, x, y, z) {                            \
    w[0] = 0.0;                                 \
    w[1] = 0.0;                                 \
    w[2] = 0.0;                                 \
}

#define w0_(w, x, y, z) {                            \
    real_t nabla_alpha[3];                            \
    _NABLA_ALPHA_(nabla_alpha, x, y, z);                            \
    real_t curl_beta[3];                            \
    _CURL_BETA_(curl_beta, x, y, z);                            \
    real_t curl_nablaalpha_times_beta[3];                            \
    _CURL_NABLAALPHA_TIMES_BETA_(curl_nablaalpha_times_beta, x, y, z);                            \
    w[0] = nabla_alpha[1]*curl_beta[2] - nabla_alpha[2]*curl_beta[1] + curl_nablaalpha_times_beta[0];                           \
    w[1] = nabla_alpha[2]*curl_beta[0] - nabla_alpha[0]*curl_beta[2] + curl_nablaalpha_times_beta[1];                           \
    w[2] = nabla_alpha[0]*curl_beta[1] - nabla_alpha[1]*curl_beta[0] + curl_nablaalpha_times_beta[2];                           \
}

#define p0_(u, x, y, z) {                            \
    u = sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);                                 \
}

#define pbdry_(u, t, x, y, z) {                            \
    u = 0.0;                                 \
}

#define A0_(A, x, y, z){  \
    A[0] = y*F_(x)*F_(y)*F_(z); \
    A[1] = -x*F_(x)*F_(y)*F_(z); \
    A[2] = F_(x)*F_(y)*F_(z); \
}

#define Bbdry_(B, t, x, y, z){  \
    B[0] = 0.0; \
    B[1] = 0.0; \
    B[2] = 0.0; \
}

#define B0_(B, x, y, z){  \
    B[0] = F_(x)*dF_(y)*F_(z) + x*F_(x)*F_(y)*dF_(z); \
    B[1] = y*F_(x)*F_(y)*dF_(z) - dF_(x)*F_(y)*F_(z); \
    B[2] = -(F_(x)+x*dF_(x))*F_(y)*F_(z) - (F_(y)+y*dF_(y))*F_(x)*F_(z); \
}

#define j0_(j, x, y, z){  \
    j[0] = -(F_(x)+x*dF_(x))*dF_(y)*F_(z) - (2.*dF_(y)+y*ddF_(y))*F_(x)*F_(z) \
         - y*F_(x)*F_(y)*ddF_(z) + dF_(x)*F_(y)*dF_(z); \
    j[1] = F_(x)*dF_(y)*dF_(z) + x*F_(x)*F_(y)*ddF_(z) \
         - (2.*dF_(x)+x*ddF_(x))*F_(y)*F_(z) + (F_(y)+y*dF_(y))*dF_(x)*F_(z); \
    j[2] = y*dF_(x)*F_(y)*dF_(z) - ddF_(x)*F_(y)*F_(z) \
         - F_(x)*ddF_(y)*F_(z) + x*F_(x)*dF_(y)*dF_(z); \
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
FUNCS_TD_DEF(func_pbdry, pbdry_);
FUNCV_TD_DEF(func_wbdry, wbdry_);

PFUNCV_TD_DEF(func_fu, fu_);
PFUNCV_TD_DEF(func_fA, fA_);

ProblemData *GetConservationProblemData(ParamList param)
{
    Array<int> ess_bdr_tangent({0,0,0,0,0,0});
    Array<int> ess_bdr_normal({1,1,1,1,1,1});
    Array<int> ess_bdr_magnetic({1,1,1,1,1,1});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
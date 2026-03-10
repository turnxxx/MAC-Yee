
#include "Hall_MHD_dual.hpp"

#define u0_(u, x, y, z) {                            \
    u[0] = (x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4;                           \
    u[1] = (x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*-9.8304E+4;                           \
    u[2] = (x*x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4;                           \
}

#define ubdry_(u, t, x, y, z) {                            \
    u0_(u, x, y, z);                                 \
}

#define w0_(w, x, y, z) {                            \
    w[0] = (x*x)*(y*y*y)*z*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*1.96608E+5+(x*x)*(y*y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y*y)*(z*z*z)*(x*1.4E+1-1.1E+1)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*3.2768E+4+(x*x*x)*(y*y*y)*(z*z*z)*(y*2.0-2.0)*pow(x-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4+(x*x*x)*(y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*9.8304E+4-(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(x*2.0+z*2.0-x*y*2.0-x*z*4.0+(x*x)*y*2.0-1.0)*9.8304E+4; \
    w[1] = (x*x*x)*(y*y)*z*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*1.96608E+5+(x*x*x)*(y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4+(x*x*x)*(y*y*y)*(z*z*z)*(y*1.4E+1-1.1E+1)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*3.2768E+4-(x*x*x)*(y*y*y)*(z*z*z)*(x*2.0-2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4-(x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(y*2.0+z*2.0+x*y*2.0-y*z*4.0-x*(y*y)*2.0-1.0)*9.8304E+4; \
    w[2] = x*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*-1.96608E+5-(x*x*x)*y*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*1.96608E+5-(x*x)*(y*y*y)*(z*z)*(x*2.0-2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*(y*2.0-2.0)*pow(x-1.0,3.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(x-z*2.0-x*y*2.0-x*z*2.0+(z*z)*2.0+x*y*z*4.0)*9.8304E+4+(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(y+z*2.0-x*y*2.0-y*z*2.0-(z*z)*2.0+x*y*z*4.0)*9.8304E+4; \
}

#define wbdry_(u, t, x, y, z) {                            \
    w0_(u, x, y, z);                                 \
}

#define p0_(u, x, y, z) {                            \
    u = sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);                                 \
}

#define pbdry_(u, t, x, y, z) {                            \
    p0_(u, x, y, z);                                 \
}

#define A0_(A, x, y, z){  \
    A[0] = (x*x*x)*(y*y*y*y)*(z*z*z)*pow(x-1.0,3.0)*pow(y-1.0,3.0)*pow(z-1.0,3.0)*3.2768E+4; \
    A[1] = (x*x*x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,3.0)*pow(y-1.0,3.0)*pow(z-1.0,3.0)*-3.2768E+4; \
    A[2] = (x*x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,3.0)*pow(y-1.0,3.0)*pow(z-1.0,3.0)*3.2768E+4; \
}

#define B0_(B, x, y, z){  \
    B[0] = (x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4; \
    B[1] = (x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*-9.8304E+4; \
    B[2] = (x*x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4; \
}

// #define Bbdry_(B, t, x, y, z){  \
//     B0_(B, x, y, z);                                 \
// }

#define Bbdry_(B, t, x, y, z){  \
    B[0] = 0.0; \
    B[1] = 0.0; \
    B[2] = 0.0; \
}

#define j0_(J, x, y, z){  \
    J[0] = (x*x)*(y*y*y)*z*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*1.96608E+5+(x*x)*(y*y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y*y)*(z*z*z)*(x*1.4E+1-1.1E+1)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*3.2768E+4+(x*x*x)*(y*y*y)*(z*z*z)*(y*2.0-2.0)*pow(x-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4+(x*x*x)*(y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*9.8304E+4-(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(x*2.0+z*2.0-x*y*2.0-x*z*4.0+(x*x)*y*2.0-1.0)*9.8304E+4; \
    J[1] = (x*x*x)*(y*y)*z*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*1.96608E+5+(x*x*x)*(y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4+(x*x*x)*(y*y*y)*(z*z*z)*(y*1.4E+1-1.1E+1)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*3.2768E+4-(x*x*x)*(y*y*y)*(z*z*z)*(x*2.0-2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*3.2768E+4-(x*x)*(y*y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,3.0)*(x*1.1E+1+y*1.1E+1-x*y*1.4E+1-8.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(y*2.0+z*2.0+x*y*2.0-y*z*4.0-x*(y*y)*2.0-1.0)*9.8304E+4; \
    J[2] = x*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*-1.96608E+5-(x*x*x)*y*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*1.96608E+5-(x*x)*(y*y*y)*(z*z)*(x*2.0-2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*(y*2.0-2.0)*pow(x-1.0,3.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*9.8304E+4-(x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(x-z*2.0-x*y*2.0-x*z*2.0+(z*z)*2.0+x*y*z*4.0)*9.8304E+4+(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(y+z*2.0-x*y*2.0-y*z*2.0-(z*z)*2.0+x*y*z*4.0)*9.8304E+4; \
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
    Array<int> ess_bdr_normal({0,0,0,0,0,0});
    Array<int> ess_bdr_magnetic({0,0,0,0,0,0});
    return new ProblemData(param, func_u0, func_w0, func_p0, func_A0, func_B0, func_j0,
     func_ubdry, func_wbdry, func_pbdry, func_Bbdry,
     ess_bdr_tangent, ess_bdr_normal, ess_bdr_magnetic,
     func_fu, func_fA);
}
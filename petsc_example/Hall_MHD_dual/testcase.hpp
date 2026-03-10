#ifndef TESTCASE_H
#define TESTCASE_H

#include "Hall_MHD_dual.hpp"

typedef enum {
    TEMPORAL=0,
    SPATIAL=1,
    SPATIAL2=10,
    SPATIALTEST=3,
    CONSERVATION=2,
    ISLAND=5,
    ADAPTIVE=6,
    ISLAND_GUO=7,
    WHISTLER=8,
    MHD_VORTEX=9,
    LOOP=11,
    ORSZAG_TANG=12,
    ORSZAG_TANG_KRAUS=13,
    TG_VORTEX=14,
    HOPF=15,
    ORSZAG_TANG_3D=17,
    FRICTION=18,
    ISLAND_ORIGIN=19,
    ISLAND_ADLER=20
} ProblemType;

ProblemData *GetTemporalProblemData(ParamList param, bool Hall, bool viscosity, bool resistivity);

ProblemData *GetSpatialProblemData(ParamList param, bool Hall, bool viscosity, bool resistivity);

ProblemData *GetSpatial2ProblemData(ParamList param, bool Hall, bool viscosity, bool resistivity);

ProblemData *GetSpatialTestProblemData(ParamList param, bool Hall, bool viscosity, bool resistivity);

ProblemData *GetConservationProblemData(ParamList param);

ProblemData *GetIslandProblemData(ParamList param);

ProblemData *GetAdaptiveProblemData(ParamList param);

ProblemData *GetIslandGuoProblemData(ParamList param);

ProblemData *GetWhistlerProblemData(ParamList param, real_t m_);

ProblemData *GetMHDVORTEXProblemData(ParamList param);

ProblemData *GetLoopProblemData(ParamList param);

ProblemData *GetOrszagTangProblemData(ParamList param);

ProblemData *GetOrszagTangKrausProblemData(ParamList param);

ProblemData *GetTGVORTEXProblemData(ParamList param, bool viscosity);

ProblemData *GetHopfProblemData(ParamList param);

ProblemData *GetOrszagTang3DProblemData(ParamList param);

ProblemData *GetFrictionProblemData(ParamList param);

ProblemData *GetIslandOriginProblemData(ParamList param);

ProblemData *GetIslandAdlerProblemData(ParamList param);

#endif
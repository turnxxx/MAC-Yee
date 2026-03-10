#ifndef TOOLS_H
#define TOOLS_H

#include "Hall_MHD_dual.hpp"

void EvaluateVectorGFAtPoint(const ParGridFunction &gf, const Vector &point, Vector &val);

void Zeromean(ParGridFunction &p);

real_t CheckDivergenceFree(ParGridFunction *u_gf);

void weak_curl(const ParGridFunction &u, ParGridFunction &w, Array<int> bdr_attr);

void strong_curl(const ParGridFunction &u1, ParGridFunction &w2);

#endif
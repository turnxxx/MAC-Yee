#pragma once

#include <petsc.h>
#include <petscdm.h>
#include <petscdmstag.h>
#include "ref_sol.h"

struct EvaluationResult {
    PetscReal err_u = 0.0;
    PetscReal err_omega = 0.0;
    PetscReal err_p = 0.0;
    PetscReal div_u_l2 = 0.0;
    PetscReal div_u_linf = 0.0;
};

class Evaluation {
public:
    Evaluation();
    ~Evaluation();

    // gridScale <= 0 时自动使用单元体积 dx*dy*dz
    // time_u_omega: 速度/涡量参考时间；time_p: 压力参考时间（可与前者不同）
    PetscErrorCode compute_error(DM dmSol, Vec sol, RefSol refSol,
                                 PetscReal time_u_omega, PetscReal time_p,
                                 PetscReal gridScale, EvaluationResult *result = nullptr);

    // 压力三时刻诊断：在 t_center-dt/2, t_center, t_center+dt/2 分别计算压力误差
    PetscErrorCode diagnose_pressure_three_times(DM dmSol, Vec sol, RefSol refSol,
                                                 PetscReal t_center, PetscReal dt,
                                                 PetscReal gridScale, const char *label = nullptr);

private:
    PetscErrorCode compute_error_u(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                   PetscReal *l2ErrOut = nullptr, PetscReal *divL2Out = nullptr, PetscReal *divLinfOut = nullptr);
    PetscErrorCode compute_error_omega(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                       PetscReal *l2ErrOut = nullptr);
    PetscErrorCode compute_error_p(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                   PetscReal *l2ErrOut = nullptr);
};
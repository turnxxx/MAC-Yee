#pragma once

#include <petsc.h>
#include <petscdm.h>
#include <petscdmstag.h>
#include "ref_sol.h"

struct EvaluationResult {
    PetscReal err_u = 0.0;
    PetscReal err_omega = 0.0;
    PetscReal err_p = 0.0;
    PetscReal err_grad_p = 0.0;
    PetscReal div_u_l2 = 0.0;
    PetscReal div_u_linf = 0.0;
};

struct InvariantResult {
    PetscReal H1 = 0.0; // H1 = <u1, omega1> * h^3
    PetscReal H2 = 0.0; // H2 = <u2, omega2> * h^3
    PetscReal K1 = 0.0; // K1 = 0.5 * <u1, u1> * h^3
    PetscReal K2 = 0.0; // K2 = 0.5 * <u2, u2> * h^3
};

struct GradPressureOmegaInnerProductResult {
    PetscReal gradP3_dot_omega2 = 0.0; // <grad p3, omega2> * h^3
    PetscReal gradP0_dot_omega1 = 0.0; // <grad p0, omega1> * h^3
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

    // 计算离散螺旋度（helicity）：
    // H1 = <u1, omega1> * h^3, H2 = <u2, omega2> * h^3
    // gridScale <= 0 时自动使用单元体积 dx*dy*dz
    PetscErrorCode compute_helicity(DM dmSol1, Vec sol1, DM dmSol2, Vec sol2,
                                    PetscReal gridScale, PetscReal *H1Out,
                                    PetscReal *H2Out);

    // 计算离散动能（kinetic energy）：
    // K1 = 0.5 * <u1, u1> * h^3, K2 = 0.5 * <u2, u2> * h^3
    // gridScale <= 0 时自动使用单元体积 dx*dy*dz
    PetscErrorCode compute_kinetic_energy(DM dmSol1, Vec sol1, DM dmSol2, Vec sol2,
                                          PetscReal gridScale, PetscReal *K1Out,
                                          PetscReal *K2Out);

    // 一次性计算 H1/H2/K1/K2
    // sol1_prev: 若非 NULL，则螺旋度使用论文 (5.20) 的时间平均定义:
    //   H1 = <(u1_new + u1_old)/2, omega1> * h^3
    //   H2 = <u2, (omega2_new + omega2_old)/2> * h^3
    PetscErrorCode compute_invariants(DM dmSol1, Vec sol1, DM dmSol2, Vec sol2,
                                      PetscReal gridScale,
                                      InvariantResult *result = nullptr,
                                      Vec sol1_prev = NULL);

    // 计算：
    //   <grad p3, omega2> * h^3  和  <grad p0, omega1> * h^3
    // 其中梯度离散与现有 grad(p0)/grad(p3) 组装保持一致
    PetscErrorCode compute_grad_pressure_omega_inner_products(
        DM dmSol1, Vec sol1, DM dmSol2, Vec sol2, PetscReal gridScale,
        GradPressureOmegaInnerProductResult *result = nullptr);

private:
    PetscErrorCode compute_error_u(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                   PetscReal *l2ErrOut = nullptr, PetscReal *divL2Out = nullptr, PetscReal *divLinfOut = nullptr);
    PetscErrorCode compute_error_omega(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                       PetscReal *l2ErrOut = nullptr);
    PetscErrorCode compute_error_p(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                   PetscReal *l2ErrOut = nullptr);
    PetscErrorCode compute_error_grad_p(DM dmSol, Vec sol, RefSol refSol, PetscReal time, PetscReal gridScale,
                                        PetscReal *l2ErrOut = nullptr);
};
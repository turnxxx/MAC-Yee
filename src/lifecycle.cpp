#include "../include/DUAL_MAC.h"
#include "../include/evaluation.h"

DUAL_MAC::DUAL_MAC()
    : Nx(0), Ny(0), Nz(0), Nt(0), xmin(0.0), xmax(1.0), ymin(0.0), ymax(1.0),
      zmin(0.0), zmax(1.0), dx(0.0), dy(0.0), dz(0.0), time(0.0), dt(0.0),
      nu(0.0), pinPressure(PETSC_TRUE), stab_alpha(1.0), stab_gamma(1.0),
      dmSol_1(NULL), dmSol_2(NULL), slot_p0(0), slot_u1(0), slot_omega1(0),
      slot_omega2(0), slot_u2(0), slot_p3(0), A(NULL), sol(NULL), rhs(NULL),
      sol_ref(NULL), sol1_cached(NULL), sol2_cached(NULL),
      hasRefSol(PETSC_FALSE), ksp(NULL), pc(NULL), snes(NULL), err_abs(0.0),
      err_rel(0.0), err_u1(0.0), err_omega2(0.0), err_p0(0.0), err_grad_p0(0.0),
      err_u2(0.0), err_omega1(0.0), err_p3(0.0), err_grad_p3(0.0) {}

DUAL_MAC::~DUAL_MAC() { (void)destroy(); }

PetscErrorCode DUAL_MAC::compute_error(PetscReal time) {
  PetscFunctionBeginUser;

  if (!hasRefSol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "未设置参考解，无法计算误差。\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!dmSol_1 || !dmSol_2 || !sol1_cached || !sol2_cached) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "当前没有可用的最终解，请先执行 solve()。\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Evaluation evaluator;
  const PetscReal cellVolume =
      (dx > 0.0 && dy > 0.0 && dz > 0.0) ? (dx * dy * dz) : -1.0;
  const PetscReal time_oneform =
      time + 0.5 * dt; // 1-form (u1, omega2, p0) 位于半整数时刻
  const PetscReal time_twoform = time; // 2-form (u2, omega1, p3) 位于整数时刻

  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "误差评估时刻：2-form(u2/omega1)@t=%.6f, 2-form(p3)@t-dt/2=%.6f; "
      "1-form(u1/omega2)@t+dt/2=%.6f, 1-form(p0)@t=%.6f\n",
      (double)time_twoform, (double)(time_twoform - 0.5 * dt),
      (double)time_oneform, (double)time_twoform));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-------- 2-form 系统 --------\n"));
  EvaluationResult res2;
  PetscCall(evaluator.compute_error(dmSol_2, sol2_cached, refSol_cached,
                                    time_twoform, time_twoform - 0.5 * dt,
                                    cellVolume, &res2));
  PetscCall(evaluator.diagnose_pressure_three_times(
      dmSol_2, sol2_cached, refSol_cached, time_twoform, dt, cellVolume,
      "2-form pressure"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-------- 1-form 系统 --------\n"));
  EvaluationResult res1;
  PetscCall(evaluator.compute_error(dmSol_1, sol1_cached, refSol_cached,
                                    time_oneform, time_twoform, cellVolume,
                                    &res1));
  PetscCall(evaluator.diagnose_pressure_three_times(
      dmSol_1, sol1_cached, refSol_cached, time_twoform, dt, cellVolume,
      "1-form pressure"));

  // 缓存本次误差，供外部（例如多层加密收敛率统计）读取
  err_u1 = res1.err_u;
  err_omega2 = res1.err_omega;
  err_p0 = res1.err_p;
  err_grad_p0 = res1.err_grad_p;
  err_u2 = res2.err_u;
  err_omega1 = res2.err_omega;
  err_p3 = res2.err_p;
  err_grad_p3 = res2.err_grad_p;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DUAL_MAC::get_pressure_means(PetscReal *mean_p0,
                                            PetscReal *mean_p3) {
  PetscFunctionBeginUser;
  if (mean_p0)
    *mean_p0 = 0.0;
  if (mean_p3)
    *mean_p3 = 0.0;

  if (!dmSol_1 || !dmSol_2 || !sol1_cached || !sol2_cached) {
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "[get_pressure_means] 没有可用的缓存解，请先执行 solve()。\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  MPI_Comm comm = PetscObjectComm((PetscObject)dmSol_1);

  // ===== p0 均值（1-form，顶点 BACK_DOWN_LEFT）=====
  {
    PetscInt dof0, dof1, dof2, dof3;
    PetscCall(DMStagGetDOF(dmSol_1, &dof0, &dof1, &dof2, &dof3));
    if (dof0 > 0 && mean_p0) {
      PetscInt startx, starty, startz, nx, ny, nz;
      PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny,
                                 &nz, NULL, NULL, NULL));
      PetscInt slot_p;
      PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN_LEFT, 0, &slot_p));

      Vec localSol;
      PetscScalar ****arr;
      PetscCall(DMGetLocalVector(dmSol_1, &localSol));
      PetscCall(DMGlobalToLocal(dmSol_1, sol1_cached, INSERT_VALUES, localSol));
      PetscCall(DMStagVecGetArrayRead(dmSol_1, localSol, &arr));

      PetscScalar localSum = 0.0;
      PetscInt localCount = 0;
      for (PetscInt ez = startz; ez < startz + nz; ++ez)
        for (PetscInt ey = starty; ey < starty + ny; ++ey)
          for (PetscInt ex = startx; ex < startx + nx; ++ex) {
            localSum += arr[ez][ey][ex][slot_p];
            localCount++;
          }

      PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localSol, &arr));
      PetscCall(DMRestoreLocalVector(dmSol_1, &localSol));

      PetscScalar globalSum = 0.0;
      PetscInt globalCount = 0;
      MPI_Allreduce(&localSum, &globalSum, 1, MPIU_SCALAR, MPI_SUM, comm);
      MPI_Allreduce(&localCount, &globalCount, 1, MPIU_INT, MPI_SUM, comm);
      if (globalCount > 0)
        *mean_p0 = PetscRealPart(globalSum) / (PetscReal)globalCount;
    }
  }

  // ===== p3 均值（2-form，单元中心 ELEMENT）=====
  {
    PetscInt dof0, dof1, dof2, dof3;
    PetscCall(DMStagGetDOF(dmSol_2, &dof0, &dof1, &dof2, &dof3));
    if (dof3 > 0 && mean_p3) {
      PetscInt startx, starty, startz, nx, ny, nz;
      PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz, &nx, &ny,
                                 &nz, NULL, NULL, NULL));
      PetscInt slot_p;
      PetscCall(DMStagGetLocationSlot(dmSol_2, ELEMENT, 0, &slot_p));

      Vec localSol;
      PetscScalar ****arr;
      PetscCall(DMGetLocalVector(dmSol_2, &localSol));
      PetscCall(DMGlobalToLocal(dmSol_2, sol2_cached, INSERT_VALUES, localSol));
      PetscCall(DMStagVecGetArrayRead(dmSol_2, localSol, &arr));

      PetscScalar localSum = 0.0;
      PetscInt localCount = 0;
      for (PetscInt ez = startz; ez < startz + nz; ++ez)
        for (PetscInt ey = starty; ey < starty + ny; ++ey)
          for (PetscInt ex = startx; ex < startx + nx; ++ex) {
            localSum += arr[ez][ey][ex][slot_p];
            localCount++;
          }

      PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localSol, &arr));
      PetscCall(DMRestoreLocalVector(dmSol_2, &localSol));

      PetscScalar globalSum = 0.0;
      PetscInt globalCount = 0;
      MPI_Allreduce(&localSum, &globalSum, 1, MPIU_SCALAR, MPI_SUM, comm);
      MPI_Allreduce(&localCount, &globalCount, 1, MPIU_INT, MPI_SUM, comm);
      if (globalCount > 0)
        *mean_p3 = PetscRealPart(globalSum) / (PetscReal)globalCount;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DUAL_MAC::destroy() {
  PetscFunctionBeginUser;

  if (A)
    PetscCall(MatDestroy(&A));
  if (sol)
    PetscCall(VecDestroy(&sol));
  if (rhs)
    PetscCall(VecDestroy(&rhs));
  if (sol_ref)
    PetscCall(VecDestroy(&sol_ref));
  if (sol1_cached)
    PetscCall(VecDestroy(&sol1_cached));
  if (sol2_cached)
    PetscCall(VecDestroy(&sol2_cached));

  if (ksp)
    PetscCall(KSPDestroy(&ksp));
  if (snes)
    PetscCall(SNESDestroy(&snes));
  pc = NULL;

  if (dmSol_1)
    PetscCall(DMDestroy(&dmSol_1));
  if (dmSol_2)
    PetscCall(DMDestroy(&dmSol_2));

  hasRefSol = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

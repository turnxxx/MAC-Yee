#include "../include/DUAL_MAC.h"
#include "../include/evaluation.h"
#include "../include/ref_sol.h"
#include "petscdmstag.h"
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "petscviewerhdf5.h"
#include <cstring>
#include <vector>
//#include"petscviewerascii.h"

// omega2 = ∇_h × u1  (edge→face)
// 使用与耦合方程 (5.5) 完全相同的离散旋度 stencil，
// 保证初始涡量与时间推进中耦合方程强制的离散旋度一致。
static PetscErrorCode
compute_discrete_curl_edge_to_face(DM dm, Vec u1_global, Vec omega2_global,
                                   PetscReal hx, PetscReal hy, PetscReal hz) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));
  Vec localU1;
  PetscScalar ****arrU1;
  PetscCall(DMGetLocalVector(dm, &localU1));
  PetscCall(DMGlobalToLocal(dm, u1_global, INSERT_VALUES, localU1));
  PetscCall(DMStagVecGetArrayRead(dm, localU1, &arrU1));

  Vec localO2;
  PetscScalar ****arrO2;
  PetscCall(DMGetLocalVector(dm, &localO2));
  PetscCall(VecZeroEntries(localO2));
  PetscCall(DMStagVecGetArray(dm, localO2, &arrO2));

  PetscInt su1x, su1y, su1z, so2x, so2y, so2z;
  PetscCall(DMStagGetLocationSlot(dm, BACK_DOWN, 0, &su1x));
  PetscCall(DMStagGetLocationSlot(dm, BACK_LEFT, 0, &su1y));
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT, 0, &su1z));
  PetscCall(DMStagGetLocationSlot(dm, LEFT, 0, &so2x));
  PetscCall(DMStagGetLocationSlot(dm, DOWN, 0, &so2y));
  PetscCall(DMStagGetLocationSlot(dm, BACK, 0, &so2z));

  for (PetscInt ez = startz; ez < startz + nz; ++ez)
    for (PetscInt ey = starty; ey < starty + ny; ++ey)
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        arrO2[ez][ey][ex][so2x] =
            (arrU1[ez][ey + 1][ex][su1z] - arrU1[ez][ey][ex][su1z]) / hy -
            (arrU1[ez + 1][ey][ex][su1y] - arrU1[ez][ey][ex][su1y]) / hz;
        arrO2[ez][ey][ex][so2y] =
            (arrU1[ez + 1][ey][ex][su1x] - arrU1[ez][ey][ex][su1x]) / hz -
            (arrU1[ez][ey][ex + 1][su1z] - arrU1[ez][ey][ex][su1z]) / hx;
        arrO2[ez][ey][ex][so2z] =
            (arrU1[ez][ey][ex + 1][su1y] - arrU1[ez][ey][ex][su1y]) / hx -
            (arrU1[ez][ey + 1][ex][su1x] - arrU1[ez][ey][ex][su1x]) / hy;
      }

  PetscCall(DMStagVecRestoreArrayRead(dm, localU1, &arrU1));
  PetscCall(DMRestoreLocalVector(dm, &localU1));
  PetscCall(DMStagVecRestoreArray(dm, localO2, &arrO2));
  PetscCall(DMLocalToGlobal(dm, localO2, INSERT_VALUES, omega2_global));
  PetscCall(DMRestoreLocalVector(dm, &localO2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// omega1 = ∇_h × u2  (face→edge)
// 使用与耦合方程 (5.2) 完全相同的离散旋度 stencil。
static PetscErrorCode
compute_discrete_curl_face_to_edge(DM dm, Vec u2_global, Vec omega1_global,
                                   PetscReal hx, PetscReal hy, PetscReal hz) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));
  Vec localU2;
  PetscScalar ****arrU2;
  PetscCall(DMGetLocalVector(dm, &localU2));
  PetscCall(DMGlobalToLocal(dm, u2_global, INSERT_VALUES, localU2));
  PetscCall(DMStagVecGetArrayRead(dm, localU2, &arrU2));

  Vec localO1;
  PetscScalar ****arrO1;
  PetscCall(DMGetLocalVector(dm, &localO1));
  PetscCall(VecZeroEntries(localO1));
  PetscCall(DMStagVecGetArray(dm, localO1, &arrO1));

  PetscInt su2x, su2y, su2z, so1x, so1y, so1z;
  PetscCall(DMStagGetLocationSlot(dm, LEFT, 0, &su2x));
  PetscCall(DMStagGetLocationSlot(dm, DOWN, 0, &su2y));
  PetscCall(DMStagGetLocationSlot(dm, BACK, 0, &su2z));
  PetscCall(DMStagGetLocationSlot(dm, BACK_DOWN, 0, &so1x));
  PetscCall(DMStagGetLocationSlot(dm, BACK_LEFT, 0, &so1y));
  PetscCall(DMStagGetLocationSlot(dm, DOWN_LEFT, 0, &so1z));

  for (PetscInt ez = startz; ez < startz + nz; ++ez)
    for (PetscInt ey = starty; ey < starty + ny; ++ey)
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        arrO1[ez][ey][ex][so1x] =
            (arrU2[ez][ey][ex][su2z] - arrU2[ez][ey - 1][ex][su2z]) / hy -
            (arrU2[ez][ey][ex][su2y] - arrU2[ez - 1][ey][ex][su2y]) / hz;
        arrO1[ez][ey][ex][so1y] =
            (arrU2[ez][ey][ex][su2x] - arrU2[ez - 1][ey][ex][su2x]) / hz -
            (arrU2[ez][ey][ex][su2z] - arrU2[ez][ey][ex - 1][su2z]) / hx;
        arrO1[ez][ey][ex][so1z] =
            (arrU2[ez][ey][ex][su2y] - arrU2[ez][ey][ex - 1][su2y]) / hx -
            (arrU2[ez][ey][ex][su2x] - arrU2[ez - 1][ey][ex][su2x]) / hy;
      }

  PetscCall(DMStagVecRestoreArrayRead(dm, localU2, &arrU2));
  PetscCall(DMRestoreLocalVector(dm, &localU2));
  PetscCall(DMStagVecRestoreArray(dm, localO1, &arrO1));
  PetscCall(DMLocalToGlobal(dm, localO1, INSERT_VALUES, omega1_global));
  PetscCall(DMRestoreLocalVector(dm, &localO1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 线性求解器选择开关：
// - 全局开关：-use_graddiv <bool> / -graddiv_gamma <real>
// - 分前缀开关：-one_use_graddiv, -one_graddiv_gamma 等
static PetscErrorCode solve_linear_system_with_switch(
    Mat A, Vec rhs, Vec sol, DM dm, const char *optionsPrefix,
    PetscBool pinPressure, PetscReal dt, PetscReal alphaExternal,
    PetscReal gammaExternal) {
  PetscFunctionBeginUser;
  PetscBool useGradDivLocal = PETSC_FALSE, useGradDivGlobal = PETSC_FALSE;
  PetscBool hasUseLocal = PETSC_FALSE, hasUseGlobal = PETSC_FALSE;
  PetscReal gammaLocal = 0.0, gammaGlobal = 0.0;
  PetscBool hasGammaLocal = PETSC_FALSE, hasGammaGlobal = PETSC_FALSE;

  if (optionsPrefix && optionsPrefix[0] != '\0') {
    PetscCall(PetscOptionsGetBool(NULL, optionsPrefix, "-use_graddiv",
                                  &useGradDivLocal, &hasUseLocal));
    PetscCall(PetscOptionsGetReal(NULL, optionsPrefix, "-graddiv_gamma",
                                  &gammaLocal, &hasGammaLocal));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_graddiv", &useGradDivGlobal,
                                &hasUseGlobal));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-graddiv_gamma", &gammaGlobal,
                                &hasGammaGlobal));

  const PetscBool useGradDiv =
      hasUseLocal
          ? useGradDivLocal
          : (hasUseGlobal ? useGradDivGlobal
                          : (gammaExternal > 0.0 ? PETSC_TRUE : PETSC_FALSE));
  const PetscReal gamma = hasGammaLocal
                              ? gammaLocal
                              : (hasGammaGlobal ? gammaGlobal : gammaExternal);
  const PetscBool attachPressureNullspace =
      pinPressure ? PETSC_FALSE : PETSC_TRUE;
  const PetscBool isTwoSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "two_", 4) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;

  // 仅 2-form(two_) 使用显式 gamma-graddiv 路径；
  // 1-form(one_) 稳定化由 solve_linear_system_basic 内部 oneCoeff 负责。
  if (useGradDiv && isTwoSystem) {
    PetscCall(solve_linear_system_graddiv(A, rhs, sol, dm, gamma, optionsPrefix,
                                          attachPressureNullspace, dt,
                                          alphaExternal, gammaExternal));
  } else {
    PetscCall(solve_linear_system_basic(A, rhs, sol, dm, optionsPrefix,
                                        attachPressureNullspace, dt,
                                        alphaExternal, gammaExternal));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// one_ 子步压力诊断：
// 比较“去掉 p0 后的动量残差”与“p0 梯度贡献”的平衡程度。
static PetscErrorCode diagnose_oneform_pressure_balance(Mat A, Vec rhs, Vec sol,
                                                        DM dmSol_1,
                                                        PetscReal time) {
  PetscFunctionBeginUser;
  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol_1, &dof0, &dof1, &dof2, &dof3));
  if (!(dof0 > 0 && dof1 > 0))
    PetscFunctionReturn(PETSC_SUCCESS);

  Vec solNoP = NULL, localNoP = NULL;
  PetscScalar ****arrNoP = NULL;
  Vec rFull = NULL, rNoP = NULL, gradEff = NULL, sumVec = NULL;
  IS isUMomentum = NULL;
  Vec rFullU = NULL, rNoPU = NULL, gradU = NULL, sumU = NULL;

  PetscCall(VecDuplicate(sol, &solNoP));
  PetscCall(VecCopy(sol, solNoP));

  PetscCall(DMGetLocalVector(dmSol_1, &localNoP));
  PetscCall(DMGlobalToLocal(dmSol_1, solNoP, INSERT_VALUES, localNoP));
  PetscCall(DMStagVecGetArray(dmSol_1, localNoP, &arrNoP));
  PetscInt sx, sy, sz, nx, ny, nz;
  PetscCall(
      DMStagGetCorners(dmSol_1, &sx, &sy, &sz, &nx, &ny, &nz, NULL, NULL, NULL));
  PetscInt slotP0;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN_LEFT, 0, &slotP0));
  for (PetscInt ez = sz; ez < sz + nz; ++ez)
    for (PetscInt ey = sy; ey < sy + ny; ++ey)
      for (PetscInt ex = sx; ex < sx + nx; ++ex)
        arrNoP[ez][ey][ex][slotP0] = 0.0;
  PetscCall(DMStagVecRestoreArray(dmSol_1, localNoP, &arrNoP));
  PetscCall(DMLocalToGlobal(dmSol_1, localNoP, INSERT_VALUES, solNoP));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localNoP));

  PetscCall(VecDuplicate(rhs, &rFull));
  PetscCall(VecDuplicate(rhs, &rNoP));
  PetscCall(VecDuplicate(rhs, &gradEff));
  PetscCall(VecDuplicate(rhs, &sumVec));

  PetscCall(MatMult(A, sol, rFull));
  PetscCall(VecAXPY(rFull, -1.0, rhs));
  PetscCall(MatMult(A, solNoP, rNoP));
  PetscCall(VecAXPY(rNoP, -1.0, rhs));

  // gradEff = A*(sol-solNoP) ; sumVec = rNoP + gradEff
  PetscCall(VecCopy(rFull, gradEff));
  PetscCall(VecAXPY(gradEff, -1.0, rNoP));
  PetscCall(VecCopy(rNoP, sumVec));
  PetscCall(VecAXPY(sumVec, 1.0, gradEff));

  // 构造 one_ 动量方程（u1 三分量）对应的行索引集。
  std::vector<DMStagStencil> stU;
  stU.reserve((size_t)nx * (size_t)ny * (size_t)nz * 3);
  for (PetscInt ez = sz; ez < sz + nz; ++ez) {
    for (PetscInt ey = sy; ey < sy + ny; ++ey) {
      for (PetscInt ex = sx; ex < sx + nx; ++ex) {
        DMStagStencil s;
        s.i = ex;
        s.j = ey;
        s.k = ez;
        s.c = 0;
        s.loc = BACK_DOWN;
        stU.push_back(s);
        s.loc = BACK_LEFT;
        stU.push_back(s);
        s.loc = DOWN_LEFT;
        stU.push_back(s);
      }
    }
  }
  PetscCall(DMStagCreateISFromStencils(dmSol_1, (PetscInt)stU.size(), stU.data(),
                                       &isUMomentum));

  PetscCall(VecGetSubVector(rFull, isUMomentum, &rFullU));
  PetscCall(VecGetSubVector(rNoP, isUMomentum, &rNoPU));
  PetscCall(VecGetSubVector(gradEff, isUMomentum, &gradU));
  PetscCall(VecGetSubVector(sumVec, isUMomentum, &sumU));

  PetscReal nFull2 = 0.0, nNoP2 = 0.0, nGrad2 = 0.0, nBal2 = 0.0;
  PetscReal nFullInf = 0.0, nNoPInf = 0.0, nGradInf = 0.0, nBalInf = 0.0;
  PetscCall(VecNorm(rFullU, NORM_2, &nFull2));
  PetscCall(VecNorm(rNoPU, NORM_2, &nNoP2));
  PetscCall(VecNorm(gradU, NORM_2, &nGrad2));
  PetscCall(VecNorm(sumU, NORM_2, &nBal2));
  PetscCall(VecNorm(rFullU, NORM_INFINITY, &nFullInf));
  PetscCall(VecNorm(rNoPU, NORM_INFINITY, &nNoPInf));
  PetscCall(VecNorm(gradU, NORM_INFINITY, &nGradInf));
  PetscCall(VecNorm(sumU, NORM_INFINITY, &nBalInf));

  const PetscReal relBal =
      nBal2 / (nNoP2 + nGrad2 + (PetscReal)1.0e-30);
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "[Diag][one_ pressure balance] t=%.6f\n"
      "  ||R_full||_u2=%.12e, ||R_no_p||_u2=%.12e, ||G(p0)||_u2=%.12e\n"
      "  ||R_no_p + G||_u2=%.12e, rel=%.12e\n"
      "  ||R_full||_uinf=%.12e, ||R_no_p||_uinf=%.12e, "
      "||G(p0)||_uinf=%.12e, ||R_no_p+G||_uinf=%.12e\n",
      (double)time, (double)nFull2, (double)nNoP2, (double)nGrad2,
      (double)nBal2, (double)relBal, (double)nFullInf, (double)nNoPInf,
      (double)nGradInf, (double)nBalInf));

  PetscCall(VecRestoreSubVector(rFull, isUMomentum, &rFullU));
  PetscCall(VecRestoreSubVector(rNoP, isUMomentum, &rNoPU));
  PetscCall(VecRestoreSubVector(gradEff, isUMomentum, &gradU));
  PetscCall(VecRestoreSubVector(sumVec, isUMomentum, &sumU));
  PetscCall(ISDestroy(&isUMomentum));
  PetscCall(VecDestroy(&sumVec));
  PetscCall(VecDestroy(&gradEff));
  PetscCall(VecDestroy(&rNoP));
  PetscCall(VecDestroy(&rFull));
  PetscCall(VecDestroy(&solNoP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if DUAL_MAC_DEBUG
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#endif

#if DUAL_MAC_DEBUG
static PetscErrorCode debug_check_vec_finite(Vec v, const char *tag) {
  PetscFunctionBeginUser;
  if (!v)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt lo = 0, hi = 0;
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));

  const PetscScalar *arr = NULL;
  PetscInt nLocal = 0;
  PetscCall(VecGetLocalSize(v, &nLocal));
  PetscCall(VecGetArrayRead(v, &arr));

  const PetscInt kSentinel = std::numeric_limits<PetscInt>::max();
  PetscInt localFirst = kSentinel;
  PetscScalar localVal = 0.0;
  for (PetscInt i = 0; i < nLocal; ++i) {
    const PetscScalar val = arr[i];
#if defined(PETSC_USE_COMPLEX)
    const bool finiteVal = std::isfinite((double)PetscRealPart(val)) &&
                           std::isfinite((double)PetscImaginaryPart(val));
#else
    const bool finiteVal = std::isfinite((double)PetscRealPart(val));
#endif
    if (!finiteVal) {
      localFirst = lo + i;
      localVal = val;
      break;
    }
  }
  PetscCall(VecRestoreArrayRead(v, &arr));

  MPI_Comm comm = PetscObjectComm((PetscObject)v);
  PetscInt globalFirst = kSentinel;
  {
    PetscMPIInt ierr =
        MPI_Allreduce(&localFirst, &globalFirst, 1, MPIU_INT, MPI_MIN, comm);
    if (ierr)
      SETERRQ(comm, PETSC_ERR_LIB,
              "MPI_Allreduce failed for first invalid index");
  }

  if (globalFirst == kSentinel) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG][finite] %s: all finite\n",
                          tag ? tag : "vec"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscMPIInt rank = 0, owner = PETSC_MPI_INT_MAX;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (localFirst == globalFirst)
    owner = rank;
  {
    PetscMPIInt ierr =
        MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, comm);
    if (ierr)
      SETERRQ(comm, PETSC_ERR_LIB, "MPI_Allreduce failed for owner rank");
  }
  if (rank != owner)
    localVal = 0.0;
  {
    PetscMPIInt ierr = MPI_Bcast(&localVal, 1, MPIU_SCALAR, owner, comm);
    if (ierr)
      SETERRQ(comm, PETSC_ERR_LIB, "MPI_Bcast failed for invalid value");
  }

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "[DEBUG][finite] %s: first invalid at global idx=%" PetscInt_FMT
      ", value=(%.16e, %.16e)\n",
      tag ? tag : "vec", globalFirst, (double)PetscRealPart(localVal),
      (double)PetscImaginaryPart(localVal)));
#else
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "[DEBUG][finite] %s: first invalid at global idx=%" PetscInt_FMT
      ", value=%.16e\n",
      tag ? tag : "vec", globalFirst, (double)PetscRealPart(localVal)));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode dump_matrix_ascii_matlab_debug(Mat A, const char *tag,
                                                     PetscInt step_id) {
  PetscFunctionBeginUser;

  std::error_code ec;
  std::filesystem::create_directories("mat", ec);
  if (ec) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 创建 mat 目录失败: %s\n",
                          ec.message().c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::ostringstream oss;
  oss << "mat/" << tag << "_step_" << std::setw(4) << std::setfill('0')
      << step_id << ".m";
  const std::string filename = oss.str();

  PetscViewer viewer = NULL;
  PetscCall(PetscObjectSetName((PetscObject)A, tag));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
  PetscCall(MatView(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 已输出矩阵(%s): %s\n", tag,
                        filename.c_str()));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode dump_vector_ascii_matlab_debug(Vec v, const char *tag,
                                                     PetscInt step_id) {
  PetscFunctionBeginUser;

  std::error_code ec;
  std::filesystem::create_directories("vec", ec);
  if (ec) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 创建 vec 目录失败: %s\n",
                          ec.message().c_str()));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::ostringstream oss;
  oss << "vec/" << tag << "_step_" << std::setw(4) << std::setfill('0')
      << step_id << ".m";
  const std::string filename = oss.str();

  PetscViewer viewer = NULL;
  PetscCall(PetscObjectSetName((PetscObject)v, tag));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
  PetscCall(VecView(v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 已输出向量(%s): %s\n", tag,
                        filename.c_str()));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode debug_report_curl_consistency(DM dmSol_1, Vec sol1,
                                                    DM dmSol_2, Vec sol2,
                                                    PetscReal hx, PetscReal hy,
                                                    PetscReal hz,
                                                    PetscReal cellVolume) {
  PetscFunctionBeginUser;

  // ---- 检查 omega2 - curl(u1) ----
  Vec localSol1 = NULL;
  PetscScalar ****arrSol1 = NULL;
  PetscCall(DMGetLocalVector(dmSol_1, &localSol1));
  PetscCall(DMGlobalToLocal(dmSol_1, sol1, INSERT_VALUES, localSol1));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localSol1, &arrSol1));

  PetscInt sx1, sy1, sz1, nx1, ny1, nz1;
  PetscCall(DMStagGetCorners(dmSol_1, &sx1, &sy1, &sz1, &nx1, &ny1, &nz1, NULL,
                             NULL, NULL));

  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z));
  PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));

  PetscReal localSqOmega2Curl = 0.0;
  for (PetscInt ez = sz1; ez < sz1 + nz1; ++ez) {
    for (PetscInt ey = sy1; ey < sy1 + ny1; ++ey) {
      for (PetscInt ex = sx1; ex < sx1 + nx1; ++ex) {
        const PetscReal curl_x =
            (PetscRealPart(arrSol1[ez][ey + 1][ex][slot_u1_z]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_z])) /
                hy -
            (PetscRealPart(arrSol1[ez + 1][ey][ex][slot_u1_y]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_y])) /
                hz;
        const PetscReal curl_y =
            (PetscRealPart(arrSol1[ez + 1][ey][ex][slot_u1_x]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_x])) /
                hz -
            (PetscRealPart(arrSol1[ez][ey][ex + 1][slot_u1_z]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_z])) /
                hx;
        const PetscReal curl_z =
            (PetscRealPart(arrSol1[ez][ey][ex + 1][slot_u1_y]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_y])) /
                hx -
            (PetscRealPart(arrSol1[ez][ey + 1][ex][slot_u1_x]) -
             PetscRealPart(arrSol1[ez][ey][ex][slot_u1_x])) /
                hy;

        const PetscReal d0 =
            PetscRealPart(arrSol1[ez][ey][ex][slot_omega2_x]) - curl_x;
        const PetscReal d1 =
            PetscRealPart(arrSol1[ez][ey][ex][slot_omega2_y]) - curl_y;
        const PetscReal d2 =
            PetscRealPart(arrSol1[ez][ey][ex][slot_omega2_z]) - curl_z;
        localSqOmega2Curl += d0 * d0 + d1 * d1 + d2 * d2;
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localSol1, &arrSol1));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localSol1));

  // ---- 检查 omega1 - curl(u2) ----
  Vec localSol2 = NULL;
  PetscScalar ****arrSol2 = NULL;
  PetscCall(DMGetLocalVector(dmSol_2, &localSol2));
  PetscCall(DMGlobalToLocal(dmSol_2, sol2, INSERT_VALUES, localSol2));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localSol2, &arrSol2));

  PetscInt sx2, sy2, sz2, nx2, ny2, nz2;
  PetscCall(DMStagGetCorners(dmSol_2, &sx2, &sy2, &sz2, &nx2, &ny2, &nz2, NULL,
                             NULL, NULL));

  PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, LEFT, 0, &slot_u2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN, 0, &slot_u2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK, 0, &slot_u2_z));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z));

  PetscReal localSqOmega1Curl = 0.0;
  for (PetscInt ez = sz2; ez < sz2 + nz2; ++ez) {
    for (PetscInt ey = sy2; ey < sy2 + ny2; ++ey) {
      for (PetscInt ex = sx2; ex < sx2 + nx2; ++ex) {
        // 在 dmSol_2 上，u2 分量位于面自由度：u2_x@LEFT, u2_y@DOWN, u2_z@BACK；
        // 这里在同一 (ex,ey,ez) 处构造 curl(u2)，目标位置是棱自由度 omega1：
        // omega1_x@BACK_DOWN, omega1_y@BACK_LEFT, omega1_z@DOWN_LEFT。
        // 例如 d(u2_z)/dy 使用 BACK 面在 y 方向相邻两点 [ey] 与 [ey-1] 的差分，
        // d(u2_y)/dz 使用 DOWN 面在 z 方向相邻两点 [ez] 与 [ez-1] 的差分。
        const PetscReal curl_x =
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_z]) -
             PetscRealPart(arrSol2[ez][ey - 1][ex][slot_u2_z])) /
                hy -
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_y]) -
             PetscRealPart(arrSol2[ez - 1][ey][ex][slot_u2_y])) /
                hz;
        // curl_y = d(u2_x)/dz - d(u2_z)/dx，
        // 分别来自 LEFT 面在 z 向差分与 BACK 面在 x 向差分。
        const PetscReal curl_y =
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_x]) -
             PetscRealPart(arrSol2[ez - 1][ey][ex][slot_u2_x])) /
                hz -
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_z]) -
             PetscRealPart(arrSol2[ez][ey][ex - 1][slot_u2_z])) /
                hx;
        // curl_z = d(u2_y)/dx - d(u2_x)/dy，
        // 分别来自 DOWN 面在 x 向差分与 LEFT 面在 y 向差分。
        const PetscReal curl_z =
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_y]) -
             PetscRealPart(arrSol2[ez][ey][ex - 1][slot_u2_y])) /
                hx -
            (PetscRealPart(arrSol2[ez][ey][ex][slot_u2_x]) -
             PetscRealPart(arrSol2[ez][ey - 1][ex][slot_u2_x])) /
                hy;

        const PetscReal d0 =
            PetscRealPart(arrSol2[ez][ey][ex][slot_omega1_x]) - curl_x;
        const PetscReal d1 =
            PetscRealPart(arrSol2[ez][ey][ex][slot_omega1_y]) - curl_y;
        const PetscReal d2 =
            PetscRealPart(arrSol2[ez][ey][ex][slot_omega1_z]) - curl_z;
        localSqOmega1Curl += d0 * d0 + d1 * d1 + d2 * d2;
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localSol2, &arrSol2));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localSol2));

  PetscReal globalSqOmega2Curl = 0.0, globalSqOmega1Curl = 0.0;
  PetscCallMPI(MPI_Allreduce(&localSqOmega2Curl, &globalSqOmega2Curl, 1,
                             MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localSqOmega1Curl, &globalSqOmega1Curl, 1,
                             MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  const PetscReal scale = (cellVolume > 0.0) ? cellVolume : 1.0;
  const PetscReal l2Omega2Curl = PetscSqrtReal(globalSqOmega2Curl * scale);
  const PetscReal l2Omega1Curl = PetscSqrtReal(globalSqOmega1Curl * scale);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "[DEBUG][consistency] ||omega2-curl(u1)||_L2 = %.12e, "
                        "||omega1-curl(u2)||_L2 = %.12e\n",
                        (double)l2Omega2Curl, (double)l2Omega1Curl));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// 根据论文的 Starting procedure 设置初始值
// 给定初始值 u_2^{h,0} 和 omega_1^{h,0}（整数时间步）
// 需要计算 u_1^{h,0}, omega_2^{h,0}（t=0时刻）
// 然后使用显式 Euler 格式计算 u_1^{h,1/2} 和 omega_2^{h,1/2}
// 参数：
//   refSol: 参考解对象，提供初始速度场和涡度场
//   u1_0: 输出参数，存储 u_1^{h,0}（1形式速度场，在 dmSol_1 中）
//   u2_0: 输出参数，存储 u_2^{h,0}（2形式速度场，在 dmSol_2 中）
//   omega1_0: 输出参数，存储 omega_1^{h,0}（1形式涡度场，在 dmSol_2 中）
//   omega2_0: 输出参数，存储 omega_2^{h,0}（2形式涡度场，在 dmSol_1 中）
PetscErrorCode DUAL_MAC::setup_initial_solution(RefSol refSol, Vec u1_0,
                                                Vec u2_0, Vec omega1_0,
                                                Vec omega2_0) {
  PetscFunctionBeginUser;
  DUAL_MAC_DEBUG_LOG("[DEBUG] 初始解设置开始\n");

  // 使用传入的 RefSol 对象来设置初始值
  // 根据论文的 Starting procedure：
  // 1. 设置 u_2^{h,0} 和 omega_1^{h,0}（整数时间步的初始值）
  // 2. 计算 u_1^{h,0} 和 omega_2^{h,0}（t=0时刻）
  // 3. 使用显式 Euler 格式计算 u_1^{h,1/2} 和 omega_2^{h,1/2}

  // 获取网格信息
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 获取 product 坐标数组
  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol_1, &cArrX, &cArrY, &cArrZ));

  // 获取 product 坐标的 slot 索引
  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol_1, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol_1, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol_1, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT, &icz_prev));

  // 使用传入的初始值向量参数
  // u1_0: 存储 u_1^{h,0}（在 dmSol_1 中）
  // u2_0: 存储 u_2^{h,0}（在 dmSol_2 中）
  // omega1_0: 存储 omega_1^{h,0}（在 dmSol_2 中）
  // omega2_0: 存储 omega_2^{h,0}（在 dmSol_1 中）

  // 获取本地向量数组用于设置初始值
  // 注意：这里我们需要知道如何访问 sol 向量的不同部分
  // 根据代码结构，sol 是一个组合向量，包含所有变量
  // 我们需要通过 DMStagVecGetArray 来访问不同 slot 的数据

  // 获取 u1_0 和 omega2_0 的本地数组（都在 dmSol_1 中）
  Vec localU1_0;
  PetscScalar ****arrU1_0;
  PetscCall(DMGetLocalVector(dmSol_1, &localU1_0));
  PetscCall(DMGlobalToLocal(dmSol_1, u1_0, INSERT_VALUES, localU1_0));
  PetscCall(DMStagVecGetArray(dmSol_1, localU1_0, &arrU1_0));

  Vec localOmega2_0;
  PetscScalar ****arrOmega2_0;
  PetscCall(DMGetLocalVector(dmSol_1, &localOmega2_0));
  PetscCall(DMGlobalToLocal(dmSol_1, omega2_0, INSERT_VALUES, localOmega2_0));
  PetscCall(DMStagVecGetArray(dmSol_1, localOmega2_0, &arrOmega2_0));

  // 获取 slot 索引
  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z));

  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));

  // 获取 u2_0 和 omega1_0 的本地数组（都在 dmSol_2 中）
  Vec localU2_0;
  PetscScalar ****arrU2_0;
  PetscCall(DMGetLocalVector(dmSol_2, &localU2_0));
  PetscCall(DMGlobalToLocal(dmSol_2, u2_0, INSERT_VALUES, localU2_0));
  PetscCall(DMStagVecGetArray(dmSol_2, localU2_0, &arrU2_0));

  Vec localOmega1_0;
  PetscScalar ****arrOmega1_0;
  PetscCall(DMGetLocalVector(dmSol_2, &localOmega1_0));
  PetscCall(DMGlobalToLocal(dmSol_2, omega1_0, INSERT_VALUES, localOmega1_0));
  PetscCall(DMStagVecGetArray(dmSol_2, localOmega1_0, &arrOmega1_0));

  PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, LEFT, 0, &slot_u2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN, 0, &slot_u2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK, 0, &slot_u2_z));

  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z));

  // ===== 步骤1：设置 u_2^{h,0} 和 omega_1^{h,0}（整数时间步的初始值）=====
  // 这些值应该从参考解或用户提供的初始条件中获取
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // 设置 u_2^{h,0}（在面上）
        // x 方向面（LEFT）
        PetscScalar x = cArrX[ex][icx_prev];
        PetscScalar y = cArrY[ey][icy_center];
        PetscScalar z = cArrZ[ez][icz_center];
        arrU2_0[ez][ey][ex][slot_u2_x] = refSol.uxRef(x, y, z, 0.0);

        // y 方向面（DOWN）
        x = cArrX[ex][icx_center];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_center];
        arrU2_0[ez][ey][ex][slot_u2_y] = refSol.uyRef(x, y, z, 0.0);

        // z 方向面（BACK）
        x = cArrX[ex][icx_center];
        y = cArrY[ey][icy_center];
        z = cArrZ[ez][icz_prev];
        arrU2_0[ez][ey][ex][slot_u2_z] = refSol.uzRef(x, y, z, 0.0);

        // 设置 omega_1^{h,0}（在棱上，从参考解中获取）
        // x 方向棱（BACK_DOWN）
        x = cArrX[ex][icx_center];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_prev];
        arrOmega1_0[ez][ey][ex][slot_omega1_x] = refSol.omegaxRef(x, y, z, 0.0);

        // y 方向棱（BACK_LEFT）
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_center];
        z = cArrZ[ez][icz_prev];
        arrOmega1_0[ez][ey][ex][slot_omega1_y] = refSol.omegayRef(x, y, z, 0.0);

        // z 方向棱（DOWN_LEFT）
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_center];
        arrOmega1_0[ez][ey][ex][slot_omega1_z] = refSol.omegazRef(x, y, z, 0.0);
      }
    }
  }

  // ===== 步骤2：设置 u_1^{h,0} 和 omega_2^{h,0} =====
  // 注意：u1 在棱上，必须按棱坐标直接从参考解赋值，不能直接拷贝 u2（u2 在面上）
  // omega_2^{h,0} 在面上，直接从参考解中获取
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        // x 方向棱（BACK_DOWN）
        PetscScalar x = cArrX[ex][icx_center];
        PetscScalar y = cArrY[ey][icy_prev];
        PetscScalar z = cArrZ[ez][icz_prev];
        arrU1_0[ez][ey][ex][slot_u1_x] = refSol.uxRef(x, y, z, 0.0);

        // y 方向棱（BACK_LEFT）
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_center];
        z = cArrZ[ez][icz_prev];
        arrU1_0[ez][ey][ex][slot_u1_y] = refSol.uyRef(x, y, z, 0.0);

        // z 方向棱（DOWN_LEFT）
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_center];
        arrU1_0[ez][ey][ex][slot_u1_z] = refSol.uzRef(x, y, z, 0.0);
      }
    }
  }

  // 设置 omega_2^{h,0}（在面上，从参考解中获取）
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // x 方向面（LEFT）的 omega_2^x
        PetscScalar x = cArrX[ex][icx_prev];
        PetscScalar y = cArrY[ey][icy_center];
        PetscScalar z = cArrZ[ez][icz_center];
        arrOmega2_0[ez][ey][ex][slot_omega2_x] = refSol.omegaxRef(x, y, z, 0.0);

        // y 方向面（DOWN）的 omega_2^y
        x = cArrX[ex][icx_center];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_center];
        arrOmega2_0[ez][ey][ex][slot_omega2_y] = refSol.omegayRef(x, y, z, 0.0);

        // z 方向面（BACK）的 omega_2^z
        x = cArrX[ex][icx_center];
        y = cArrY[ey][icy_center];
        z = cArrZ[ez][icz_prev];
        arrOmega2_0[ez][ey][ex][slot_omega2_z] = refSol.omegazRef(x, y, z, 0.0);
      }
    }
  }

  // 注意：omega_1^{h,0} 已经从参考解中设置（见上面的代码）
  // 如果需要验证一致性，可以通过以下旋度计算来检查：
  // omega_1^{h,0} 应该等于 ∇×u_2^{h,0}
  // 但根据论文的 Starting procedure，omega_1^{h,0} 是给定的初始值
  // 因此这里不再重新计算，直接使用从参考解中获取的值

  // 将本地值写回全局向量
  PetscCall(DMLocalToGlobal(dmSol_1, localU1_0, INSERT_VALUES, u1_0));
  PetscCall(DMLocalToGlobal(dmSol_1, localOmega2_0, INSERT_VALUES, omega2_0));
  PetscCall(DMLocalToGlobal(dmSol_2, localU2_0, INSERT_VALUES, u2_0));
  PetscCall(DMLocalToGlobal(dmSol_2, localOmega1_0, INSERT_VALUES, omega1_0));

  // ===== 步骤3：使用 Starting procedure 计算 u_1^{h,1/2} 和 omega_2^{h,1/2}
  // ===== 根据论文公式： u_1^{h,1/2}/dt + ∇_h P_0^{h,0} = R_1 f^0 - ω_1^{h,0} ×
  // ω_1^{h,0} - (1/Re) ∇_h × ω_2^{h,0} + u_1^{h,0}/dt ∇_h · u_1^{h,1/2} = 0
  // ω_2^{h,1/2} = ∇_h × u_1^{h,1/2}

  // 注意：这里需要求解一个线性系统来得到 u_1^{h,1/2} 和 P_0^{h,0}
  // 为了简化，这里我们只设置基本结构，实际求解需要在时间循环中完成

  // 释放数组
  PetscCall(DMStagVecRestoreArray(dmSol_1, localU1_0, &arrU1_0));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localU1_0));
  PetscCall(DMStagVecRestoreArray(dmSol_1, localOmega2_0, &arrOmega2_0));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2_0));
  PetscCall(DMStagVecRestoreArray(dmSol_2, localU2_0, &arrU2_0));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localU2_0));
  PetscCall(DMStagVecRestoreArray(dmSol_2, localOmega1_0, &arrOmega1_0));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1_0));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmSol_1, &cArrX, &cArrY,
                                                     &cArrZ));

  DUAL_MAC_DEBUG_LOG("[DEBUG] 初始解设置完成\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}
// 根据论文的 Starting procedure，使用显式欧拉法计算二分之一时刻的初值
// 求解系统：
//   u₁^{h,1/2}/dt + ∇P₀^{h,0} = R₁f^0 - ω₁^{h,0}×u₁^{h,0} - (1/Re)∇×ω₂^{h,0} +
//   u₁^{h,0}/dt ∇·u₁^{h,1/2} = 0
// 然后计算：ω₂^{h,1/2} = ∇×u₁^{h,1/2}
PetscErrorCode DUAL_MAC::compute_half_solution(Vec u1_0, Vec omega1_0,
                                               Vec omega2_0, Vec u1_half,
                                               Vec omega2_half,
                                               ExternalForce externalForce) {
  PetscFunctionBeginUser;
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1/2时刻解计算开始\n");

  // 计算雷诺数
  PetscReal Re = 1.0 / this->nu;

  // 使用 half 专用 DM（仅包含 u1 + p0），避免把 omega2 的自由度放进 half
  // 线性系统
  DM dmHalf = NULL;
  PetscCall(DMStagCreateCompatibleDMStag(dmSol_1, 1, 1, 0, 0, &dmHalf));
  PetscCall(DMSetUp(dmHalf));
  PetscCall(DMStagSetUniformCoordinatesProduct(dmHalf, this->xmin, this->xmax,
                                               this->ymin, this->ymax,
                                               this->zmin, this->zmax));

  // ===== 1. 创建矩阵和右端项向量 =====
  Mat A;
  Vec rhs, sol_half;
  PetscCall(DMCreateMatrix(dmHalf, &A));
  PetscCall(DMCreateGlobalVector(dmHalf, &rhs));
  PetscCall(DMCreateGlobalVector(dmHalf, &sol_half));
  PetscCall(MatZeroEntries(A));

  // ===== 2. 组装矩阵 =====
  // 2.1 时间导数矩阵：2/dt * I（对 u₁），半步从 t=0 到 t=dt/2
  const PetscReal half_dt = 0.5 * this->dt;
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装时间导数矩阵开始\n");
  PetscCall(assemble_u1_dt_matrix(dmHalf, A, half_dt));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装时间导数矩阵完成\n");

  // 2.2 压力梯度矩阵：∇（对 P₀）
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装压力梯度矩阵开始\n");
  PetscCall(assemble_p0_gradient_matrix(dmHalf, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装压力梯度矩阵完成\n");

  // 2.3 散度矩阵：∇·（对 u₁）
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装散度矩阵开始\n");
  PetscCall(assemble_u1_divergence_matrix(dmHalf, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 组装散度矩阵完成\n");

  // ===== 3. 组装右端项 =====
  PetscCall(VecZeroEntries(rhs));

  // 3.1 时间导数项：u₁^{h,0} / (dt/2)
  // 将主 DM 中的 u1 拷贝到 half DM 的 rhs（p0 分量保持为0）
  PetscCall(VecZeroEntries(rhs));
  PetscCall(extract_u1_from_solution(dmSol_1, u1_0, dmHalf, rhs));
  PetscCall(VecScale(rhs, 1.0 / half_dt));

  // 3.2 外力项：R₁ f^0
  PetscCall(assemble_force1_vector(dmHalf, rhs, externalForce, 0.0));

  // 3.3 对流项：-ω₁^{h,0} × u₁^{h,0}
  PetscCall(assemble_omega1_u1_conv_rhs(dmHalf, rhs, dmSol_2, omega1_0, dmSol_1,
                                        u1_0));

  // 3.4 旋度项：-1/Re * ∇×ω₂^{h,0}
  // 使用 assemble_omega2_curl_matrix 的逻辑，但直接计算到右端项
  PetscCall(assemble_omega2_curl_rhs(dmHalf, rhs, dmSol_1, omega2_0, Re));

  // ===== 4. 矩阵和向量最终组装 =====
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(rhs));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(rhs));

  // 启用 pinPressure 时，固定 1-form 的 p0(0,0,0)=0 对应右端项
  if (this->pinPressure) {
    PetscInt startx_pin, starty_pin, startz_pin, nx_pin, ny_pin, nz_pin;
    PetscCall(DMStagGetCorners(dmHalf, &startx_pin, &starty_pin, &startz_pin,
                               &nx_pin, &ny_pin, &nz_pin, NULL, NULL, NULL));
    const PetscBool owns_anchor =
        (0 >= startx_pin && 0 < startx_pin + nx_pin) &&
                (0 >= starty_pin && 0 < starty_pin + ny_pin) &&
                (0 >= startz_pin && 0 < startz_pin + nz_pin)
            ? PETSC_TRUE
            : PETSC_FALSE;

    if (owns_anchor) {
      DMStagStencil prow;
      PetscScalar pval = 0.0;
      prow.i = 0;
      prow.j = 0;
      prow.k = 0;
      prow.loc = BACK_DOWN_LEFT;
      prow.c = 0;
      PetscCall(DMStagVecSetValuesStencil(dmHalf, rhs, 1, &prow, &pval,
                                          INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyEnd(rhs));
  }

#if DUAL_MAC_DEBUG
  PetscCall(dump_matrix_ascii_matlab_debug(A, "half_assembled", 0));
  PetscCall(dump_vector_ascii_matlab_debug(rhs, "half_rhs", 0));
#endif

  // ===== 5. 求解线性系统 =====
  //    求解后会自动对 P0 压力进行均值归零处理
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 线性求解开始\n");
  PetscCall(solve_linear_system_with_switch(
      A, rhs, sol_half, dmHalf, "half_", this->pinPressure, half_dt,
      this->stab_alpha, this->stab_gamma));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1/2步] 线性求解完成\n");
#if DUAL_MAC_DEBUG
  PetscCall(debug_check_vec_finite(sol_half, "half_sol_after_solve"));
  PetscCall(dump_vector_ascii_matlab_debug(sol_half, "half_sol_raw", 0));
#endif

  // ===== 6. 提取 u₁^{h,1/2} =====
  // sol_half 包含 u₁ 和 P₀，需要将 u₁ 提取并回写到主 DM 的 u1_half
  PetscCall(VecZeroEntries(u1_half));
  PetscCall(extract_u1_from_solution(dmHalf, sol_half, dmSol_1, u1_half));
#if DUAL_MAC_DEBUG
  PetscCall(debug_check_vec_finite(u1_half, "u1_half_after_extract"));
#endif

  // ===== 7. 计算 omega_2^{h,1/2} = ∇×u₁^{h,1/2} =====
  PetscCall(compute_curl_u1_to_omega2(dmSol_1, u1_half, omega2_half));
#if DUAL_MAC_DEBUG
  PetscCall(debug_check_vec_finite(omega2_half, "omega2_half_after_curl"));
#endif

  // ===== 8. 清理资源 =====
  PetscCall(VecDestroy(&sol_half));
  PetscCall(VecDestroy(&rhs));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dmHalf));

  DUAL_MAC_DEBUG_LOG("[DEBUG] 1/2时刻解计算完成\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：组装 -ω₁ × u₁ 项到右端项（Starting procedure 中的对流项）
PetscErrorCode DUAL_MAC::assemble_omega1_u1_conv_rhs(DM dmRhs, Vec rhs,
                                                     DM dmSol_2, Vec omega1_0,
                                                     DM dmU1, Vec u1_0) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmRhs, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 获取 omega1_0 的本地数组（使用 dmSol_2，因为 ω₁ 存储在 dmSol_2 中）
  Vec localOmega1;
  PetscScalar ****arrOmega1;
  PetscCall(DMGetLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMGlobalToLocal(dmSol_2, omega1_0, INSERT_VALUES, localOmega1));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localOmega1, &arrOmega1));

  // 获取 u1_0 的本地数组（由 dmU1 指定存储DM）
  Vec localU1;
  PetscScalar ****arrU1;
  PetscCall(DMGetLocalVector(dmU1, &localU1));
  PetscCall(DMGlobalToLocal(dmU1, u1_0, INSERT_VALUES, localU1));
  PetscCall(DMStagVecGetArrayRead(dmU1, localU1, &arrU1));

  // 获取 slot 索引
  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z));

  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(DMStagGetLocationSlot(dmU1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmU1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmU1, DOWN_LEFT, 0, &slot_u1_z));

  // 遍历所有单元
  // 周期边界下，DMGlobalToLocal 已经提供了 ghost 值（含周期映射），
  // 这里统一使用前向差分，不再在边界截断导数项。
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的 -ω₁ × u₁ 项 =====
        // (ω₁ × u₁)_x = ω₁ʸ u₁ᶻ - ω₁ᶻ u₁ʸ
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN;
        row.c = 0;

        // 插值 ω₁ 到 x 方向棱的位置（与
        // assemble_rhs1_vector/assemble_u1_conv_matrix 中的逻辑相同）
        PetscScalar omega1_y =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_y] +
                    arrOmega1[ez][ey][ex + 1][slot_omega1_y] +
                    arrOmega1[ez][ey - 1][ex][slot_omega1_y] +
                    arrOmega1[ez][ey - 1][ex + 1][slot_omega1_y]);

        PetscScalar omega1_z =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_z] +
                    arrOmega1[ez][ey][ex + 1][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex + 1][slot_omega1_z]);

        // u₁ᶻ、u₁ʸ 插值到 x 方向棱
        PetscScalar u1_z = 0.25 * (arrU1[ez][ey][ex][slot_u1_z] +
                                   arrU1[ez][ey][ex + 1][slot_u1_z] +
                                   arrU1[ez - 1][ey][ex][slot_u1_z] +
                                   arrU1[ez - 1][ey][ex + 1][slot_u1_z]);

        PetscScalar u1_y = 0.25 * (arrU1[ez][ey][ex][slot_u1_y] +
                                   arrU1[ez][ey][ex + 1][slot_u1_y] +
                                   arrU1[ez][ey - 1][ex][slot_u1_y] +
                                   arrU1[ez][ey - 1][ex + 1][slot_u1_y]);

        // - (ω₁ × u₁)_x
        PetscScalar rhs_val = -(omega1_y * u1_z - omega1_z * u1_y);
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的 -ω₁ × u₁ 项 =====
        row.loc = BACK_LEFT;
        // 插值逻辑（与 assemble_rhs1_vector/assemble_u1_conv_matrix
        // 中的逻辑相同）
        PetscScalar omega1_z_y =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_z] +
                    arrOmega1[ez][ey + 1][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey + 1][ex][slot_omega1_z]);

        PetscScalar omega1_x_y =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_x] +
                    arrOmega1[ez][ey][ex - 1][slot_omega1_x] +
                    arrOmega1[ez][ey + 1][ex][slot_omega1_x] +
                    arrOmega1[ez][ey + 1][ex - 1][slot_omega1_x]);

        // u₁ˣ、u₁ᶻ 插值到 y 方向棱
        PetscScalar u1_x = 0.25 * (arrU1[ez][ey][ex][slot_u1_x] +
                                   arrU1[ez][ey][ex - 1][slot_u1_x] +
                                   arrU1[ez][ey + 1][ex][slot_u1_x] +
                                   arrU1[ez][ey + 1][ex - 1][slot_u1_x]);

        PetscScalar u1_z_y = 0.25 * (arrU1[ez][ey][ex][slot_u1_z] +
                                     arrU1[ez][ey + 1][ex][slot_u1_z] +
                                     arrU1[ez - 1][ey][ex][slot_u1_z] +
                                     arrU1[ez - 1][ey + 1][ex][slot_u1_z]);

        // (ω₁ × u₁)_y = ω₁ᶻ u₁ˣ - ω₁ˣ u₁ᶻ
        rhs_val = -(omega1_z_y * u1_x - omega1_x_y * u1_z_y);
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的 -ω₁ × u₁ 项 =====
        row.loc = DOWN_LEFT;
        // 插值逻辑（与 assemble_rhs1_vector/assemble_u1_conv_matrix
        // 中的逻辑相同）
        PetscScalar omega1_x_z =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_x] +
                    arrOmega1[ez + 1][ey][ex][slot_omega1_x] +
                    arrOmega1[ez][ey][ex - 1][slot_omega1_x] +
                    arrOmega1[ez + 1][ey][ex - 1][slot_omega1_x]);

        PetscScalar omega1_y_z =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_y] +
                    arrOmega1[ez + 1][ey][ex][slot_omega1_y] +
                    arrOmega1[ez][ey - 1][ex][slot_omega1_y] +
                    arrOmega1[ez + 1][ey - 1][ex][slot_omega1_y]);

        // u₁ʸ、u₁ˣ 插值到 z 方向棱
        PetscScalar u1_y_z = 0.25 * (arrU1[ez][ey][ex][slot_u1_y] +
                                     arrU1[ez + 1][ey][ex][slot_u1_y] +
                                     arrU1[ez][ey - 1][ex][slot_u1_y] +
                                     arrU1[ez + 1][ey - 1][ex][slot_u1_y]);

        PetscScalar u1_x_z = 0.25 * (arrU1[ez][ey][ex][slot_u1_x] +
                                     arrU1[ez + 1][ey][ex][slot_u1_x] +
                                     arrU1[ez][ey][ex - 1][slot_u1_x] +
                                     arrU1[ez + 1][ey][ex - 1][slot_u1_x]);

        // (ω₁ × u₁)_z = ω₁ˣ u₁ʸ - ω₁ʸ u₁ˣ
        rhs_val = -(omega1_x_z * u1_y_z - omega1_y_z * u1_x_z);
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));
      }
    }
  }

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localOmega1, &arrOmega1));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMStagVecRestoreArrayRead(dmU1, localU1, &arrU1));
  PetscCall(DMRestoreLocalVector(dmU1, &localU1));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：组装 -1/Re * ∇×ω₂ 项到右端项
PetscErrorCode DUAL_MAC::assemble_omega2_curl_rhs(DM dmRhs, Vec rhs,
                                                  DM dmOmega2, Vec omega2_0,
                                                  PetscReal Re) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmRhs, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  PetscScalar coeff = -1.0 / Re; // -1/Re 系数

  // 获取 omega2_0 的本地数组
  Vec localOmega2;
  PetscScalar ****arrOmega2;
  PetscCall(DMGetLocalVector(dmOmega2, &localOmega2));
  PetscCall(DMGlobalToLocal(dmOmega2, omega2_0, INSERT_VALUES, localOmega2));
  PetscCall(DMStagVecGetArrayRead(dmOmega2, localOmega2, &arrOmega2));

  // 获取 slot 索引
  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmOmega2, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmOmega2, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmOmega2, BACK, 0, &slot_omega2_z));

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的旋度项 =====
        // -(1/Re) * (∇ × ω₂)_x = -(1/Re) * (∂ω₂ᶻ/∂y - ∂ω₂ʸ/∂z)
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN;
        row.c = 0;

        PetscScalar curl_x = 0.0;
        // ∂ω₂ᶻ/∂y ≈ (ω₂ᶻ|_{ey} - ω₂ᶻ|_{ey-1}) / hy
        curl_x += (arrOmega2[ez][ey][ex][slot_omega2_z] -
                   arrOmega2[ez][ey - 1][ex][slot_omega2_z]) /
                  hy;
        // -∂ω₂ʸ/∂z ≈ -(ω₂ʸ|_{ez} - ω₂ʸ|_{ez-1}) / hz
        curl_x -= (arrOmega2[ez][ey][ex][slot_omega2_y] -
                   arrOmega2[ez - 1][ey][ex][slot_omega2_y]) /
                  hz;

        PetscScalar rhs_val = coeff * curl_x;
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的旋度项 =====
        // -(1/Re) * (∇ × ω₂)_y = -(1/Re) * (∂ω₂ˣ/∂z - ∂ω₂ᶻ/∂x)
        row.loc = BACK_LEFT;

        PetscScalar curl_y = 0.0;
        // ∂ω₂ˣ/∂z ≈ (ω₂ˣ|_{ez} - ω₂ˣ|_{ez-1}) / hz
        curl_y += (arrOmega2[ez][ey][ex][slot_omega2_x] -
                   arrOmega2[ez - 1][ey][ex][slot_omega2_x]) /
                  hz;
        // -∂ω₂ᶻ/∂x ≈ -(ω₂ᶻ|_{ex} - ω₂ᶻ|_{ex-1}) / hx
        curl_y -= (arrOmega2[ez][ey][ex][slot_omega2_z] -
                   arrOmega2[ez][ey][ex - 1][slot_omega2_z]) /
                  hx;

        rhs_val = coeff * curl_y;
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的旋度项 =====
        // -(1/Re) * (∇ × ω₂)_z = -(1/Re) * (∂ω₂ʸ/∂x - ∂ω₂ˣ/∂y)
        row.loc = DOWN_LEFT;

        PetscScalar curl_z = 0.0;
        // ∂ω₂ʸ/∂x ≈ (ω₂ʸ|_{ex} - ω₂ʸ|_{ex-1}) / hx
        curl_z += (arrOmega2[ez][ey][ex][slot_omega2_y] -
                   arrOmega2[ez][ey][ex - 1][slot_omega2_y]) /
                  hx;
        // -∂ω₂ˣ/∂y ≈ -(ω₂ˣ|_{ey} - ω₂ˣ|_{ey-1}) / hy
        curl_z -= (arrOmega2[ez][ey][ex][slot_omega2_x] -
                   arrOmega2[ez][ey - 1][ex][slot_omega2_x]) /
                  hy;

        rhs_val = coeff * curl_z;
        PetscCall(DMStagVecSetValuesStencil(dmRhs, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));
      }
    }
  }

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmOmega2, localOmega2, &arrOmega2));
  PetscCall(DMRestoreLocalVector(dmOmega2, &localOmega2));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：从解向量中提取 u₁
// 支持源DM和目标DM不同：用于 half 专用DM 与主DM之间的数据转存
PetscErrorCode DUAL_MAC::extract_u1_from_solution(DM dmSolSrc, Vec sol,
                                                  DM dmSolDst, Vec u1_half) {
  PetscFunctionBeginUser;

  // 获取本地数组
  Vec localSol, localU1Half;
  PetscScalar ****arrSol, ****arrU1Half;
  PetscCall(DMGetLocalVector(dmSolSrc, &localSol));
  PetscCall(DMGlobalToLocal(dmSolSrc, sol, INSERT_VALUES, localSol));
  PetscCall(DMStagVecGetArrayRead(dmSolSrc, localSol, &arrSol));

  PetscCall(DMGetLocalVector(dmSolDst, &localU1Half));
  PetscCall(DMGlobalToLocal(dmSolDst, u1_half, INSERT_VALUES, localU1Half));
  PetscCall(DMStagVecGetArray(dmSolDst, localU1Half, &arrU1Half));

  // 获取源/目标DM中的 u1 slot
  PetscInt slot_u1_src_x, slot_u1_src_y, slot_u1_src_z;
  PetscInt slot_u1_dst_x, slot_u1_dst_y, slot_u1_dst_z;
  PetscCall(DMStagGetLocationSlot(dmSolSrc, BACK_DOWN, 0, &slot_u1_src_x));
  PetscCall(DMStagGetLocationSlot(dmSolSrc, BACK_LEFT, 0, &slot_u1_src_y));
  PetscCall(DMStagGetLocationSlot(dmSolSrc, DOWN_LEFT, 0, &slot_u1_src_z));
  PetscCall(DMStagGetLocationSlot(dmSolDst, BACK_DOWN, 0, &slot_u1_dst_x));
  PetscCall(DMStagGetLocationSlot(dmSolDst, BACK_LEFT, 0, &slot_u1_dst_y));
  PetscCall(DMStagGetLocationSlot(dmSolDst, DOWN_LEFT, 0, &slot_u1_dst_z));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSolDst, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 复制 u₁ 部分（dof1 的第0个分量）
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        arrU1Half[ez][ey][ex][slot_u1_dst_x] =
            arrSol[ez][ey][ex][slot_u1_src_x];
        arrU1Half[ez][ey][ex][slot_u1_dst_y] =
            arrSol[ez][ey][ex][slot_u1_src_y];
        arrU1Half[ez][ey][ex][slot_u1_dst_z] =
            arrSol[ez][ey][ex][slot_u1_src_z];
      }
    }
  }

  // 写回全局向量
  PetscCall(DMLocalToGlobal(dmSolDst, localU1Half, INSERT_VALUES, u1_half));

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSolSrc, localSol, &arrSol));
  PetscCall(DMRestoreLocalVector(dmSolSrc, &localSol));
  PetscCall(DMStagVecRestoreArray(dmSolDst, localU1Half, &arrU1Half));
  PetscCall(DMRestoreLocalVector(dmSolDst, &localU1Half));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：计算 omega_2 = ∇×u₁
// 使用与 assemble_u1_omega2_coupling_matrix 相同的旋度计算逻辑
PetscErrorCode DUAL_MAC::compute_curl_u1_to_omega2(DM dmSol_1, Vec u1_half,
                                                   Vec omega2_half) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 获取本地数组
  Vec localU1Half, localOmega2Half;
  PetscScalar ****arrU1Half, ****arrOmega2Half;
  PetscCall(DMGetLocalVector(dmSol_1, &localU1Half));
  PetscCall(DMGlobalToLocal(dmSol_1, u1_half, INSERT_VALUES, localU1Half));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localU1Half, &arrU1Half));

  PetscCall(DMGetLocalVector(dmSol_1, &localOmega2Half));
  PetscCall(
      DMGlobalToLocal(dmSol_1, omega2_half, INSERT_VALUES, localOmega2Half));
  PetscCall(DMStagVecGetArray(dmSol_1, localOmega2Half, &arrOmega2Half));

  // 获取 slot 索引
  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z));

  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向面 (LEFT) 的旋度 =====
        // (∇×u₁)ₓ = ∂u₁ᶻ/∂y - ∂u₁ʸ/∂z
        PetscScalar curl_x = 0.0;
        // ∂u₁ᶻ/∂y ≈ (u₁ᶻ|_{y=next} - u₁ᶻ|_{y=prev}) / hy
        curl_x += (arrU1Half[ez][ey + 1][ex][slot_u1_z] -
                   arrU1Half[ez][ey][ex][slot_u1_z]) /
                  hy;
        // -∂u₁ʸ/∂z ≈ -(u₁ʸ|_{z=next} - u₁ʸ|_{z=prev}) / hz
        curl_x -= (arrU1Half[ez + 1][ey][ex][slot_u1_y] -
                   arrU1Half[ez][ey][ex][slot_u1_y]) /
                  hz;
        arrOmega2Half[ez][ey][ex][slot_omega2_x] = curl_x;

        // ===== y 方向面 (DOWN) 的旋度 =====
        // (∇×u₁)ᵧ = ∂u₁ˣ/∂z - ∂u₁ᶻ/∂x
        PetscScalar curl_y = 0.0;
        // ∂u₁ˣ/∂z ≈ (u₁ˣ|_{z=next} - u₁ˣ|_{z=prev}) / hz
        curl_y += (arrU1Half[ez + 1][ey][ex][slot_u1_x] -
                   arrU1Half[ez][ey][ex][slot_u1_x]) /
                  hz;
        // -∂u₁ᶻ/∂x ≈ -(u₁ᶻ|_{x=next} - u₁ᶻ|_{x=prev}) / hx
        curl_y -= (arrU1Half[ez][ey][ex + 1][slot_u1_z] -
                   arrU1Half[ez][ey][ex][slot_u1_z]) /
                  hx;
        arrOmega2Half[ez][ey][ex][slot_omega2_y] = curl_y;

        // ===== z 方向面 (BACK) 的旋度 =====
        // (∇×u₁)ᵧ = ∂u₁ʸ/∂x - ∂u₁ˣ/∂y
        PetscScalar curl_z = 0.0;
        // ∂u₁ʸ/∂x ≈ (u₁ʸ|_{x=next} - u₁ʸ|_{x=prev}) / hx
        curl_z += (arrU1Half[ez][ey][ex + 1][slot_u1_y] -
                   arrU1Half[ez][ey][ex][slot_u1_y]) /
                  hx;
        // -∂u₁ˣ/∂y ≈ -(u₁ˣ|_{y=next} - u₁ˣ|_{y=prev}) / hy
        curl_z -= (arrU1Half[ez][ey + 1][ex][slot_u1_x] -
                   arrU1Half[ez][ey][ex][slot_u1_x]) /
                  hy;
        arrOmega2Half[ez][ey][ex][slot_omega2_z] = curl_z;
      }
    }
  }

  // 写回全局向量
  PetscCall(
      DMLocalToGlobal(dmSol_1, localOmega2Half, INSERT_VALUES, omega2_half));

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localU1Half, &arrU1Half));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localU1Half));
  PetscCall(DMStagVecRestoreArray(dmSol_1, localOmega2Half, &arrOmega2Half));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2Half));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 1-形式系统时间推进：给定半步旧解 sol1_old 和整数步新解 sol2_new，求半步新解
// sol1_new sol1_old : (u1^{k-1/2}, ω2^{k-1/2}, P0^{k-1/2}) 对应的 DM dmSol_1
// 全局向量 sol1_new : (u1^{k+1/2}, ω2^{k+1/2}, P0^{k})    求解得到，存回同一 DM
// 的全局向量 sol2_old : 整数步旧解 (u2^{k-1}, ω1^{k-1}, P3^{k-3/2}) ——
// 这里只作为接口占位，不直接使用 sol2_new : 整数步新解 (u2^{k},   ω1^{k},
// P3^{k-1/2})，其中 ω1^{k} 用作对流场
PetscErrorCode DUAL_MAC::time_evolve1(DM dmSol_1, DM dmSol_2, Vec sol1_old,
                                      Vec sol1_new, Vec sol2_old, Vec sol2_new,
                                      ExternalForce externalForce,
                                      PetscReal time) {
  PetscFunctionBeginUser;

  // 1. 创建矩阵和右端项
  Mat A;
  Vec rhs;
  PetscCall(DMCreateMatrix(dmSol_1, &A));
  PetscCall(DMCreateGlobalVector(dmSol_1, &rhs));

  // 2. 组装 1-形式系统矩阵与右端项
  //    u1_prev      = sol1_old (dmSol_1 中的 u1^{k-1/2})
  //    omega1_known = sol2_new (dmSol_2 中的 ω1^{k})
  //    omega2_prev  = sol1_old (dmSol_1 中的 ω2^{k-1/2})
  //    根据论文方程(5.4)，外力项是 R₁ f^k，即整数时间步 k 的外力
  //    假设传入的 time 是当前整数时间步 k，则直接使用
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 子步矩阵组装开始\n");
  PetscCall(assemble_1form_system_matrix(dmSol_1, dmSol_2, A, rhs, sol1_old,
                                         sol2_new, sol1_old, externalForce,
                                         time, this->dt));
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 子步矩阵组装完成\n");
#if DUAL_MAC_DEBUG
  {
    const PetscInt step_id = (PetscInt)(time / this->dt);
    PetscCall(dump_matrix_ascii_matlab_debug(A, "oneform_assembled", step_id));
    PetscCall(dump_vector_ascii_matlab_debug(rhs, "oneform_rhs", step_id));
  }
#endif

  // 3. 求解线性系统，得到新的 1-形式解 sol1_new
  //    求解后会自动对 P0 压力进行均值归零处理
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 子步线性求解开始\n");
  PetscCall(solve_linear_system_with_switch(
      A, rhs, sol1_new, dmSol_1, "one_", this->pinPressure, this->dt,
      this->stab_alpha, this->stab_gamma));
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 子步线性求解完成\n");
  PetscCall(diagnose_oneform_pressure_balance(A, rhs, sol1_new, dmSol_1, time));

  // 4. 释放资源
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 2-形式系统时间推进：给定整数步旧解 sol2_old 和半步旧解 sol1_old，求整数步新解
// sol2_new sol1_old : 半步旧解 (u1^{k-1/2}, ω2^{k-1/2}, P0^{k-1/2})，其中
// ω2^{k-1/2} 作为对流场 sol1_new : 半步新解 (u1^{k+1/2}, ω2^{k+1/2}, P0^{k}) ——
// 这里只作为接口占位，不直接使用 sol2_old : 整数步旧解 (u2^{k-1}, ω1^{k-1},
// P3^{k-3/2}) sol2_new : 整数步新解 (u2^{k},   ω1^{k},   P3^{k-1/2}) 求解得到
PetscErrorCode DUAL_MAC::time_evolve2(DM dmSol_1, DM dmSol_2, Vec sol1_old,
                                      Vec sol1_new, Vec sol2_old, Vec sol2_new,
                                      ExternalForce externalForce,
                                      PetscReal time) {
  PetscFunctionBeginUser;

  // 1. 创建矩阵和右端项
  Mat A;
  Vec rhs;
  PetscCall(DMCreateMatrix(dmSol_2, &A));
  PetscCall(DMCreateGlobalVector(dmSol_2, &rhs));

  // 2. 组装 2-形式系统矩阵与右端项
  //    u2_prev      = sol2_old (dmSol_2 中的 u2^{k-1})
  //    omega2_known = sol1_old (dmSol_1 中的 ω2^{k-1/2})
  //    omega1_prev  = sol2_old (dmSol_2 中的 ω1^{k-1})
  //    根据论文方程(5.1)，外力项是 I_{RT} f^{k-1/2}，即半整数时间步 k-1/2
  //    的外力 假设传入的 time 是当前整数时间步 k，则半整数时间步是 k - 0.5*dt
  PetscReal time_for_force = time - 0.5 * this->dt; // 半整数时间步 k-1/2
  DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 子步矩阵组装开始\n");
  PetscCall(assemble_2form_system_matrix(dmSol_1, dmSol_2, A, rhs, sol2_old,
                                         sol1_old, sol2_old, externalForce,
                                         time_for_force, this->dt));
  DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 子步矩阵组装完成\n");
#if DUAL_MAC_DEBUG
  {
    const PetscInt step_id = (PetscInt)(time / this->dt);
    PetscCall(dump_matrix_ascii_matlab_debug(A, "twoform_assembled", step_id));
    PetscCall(dump_vector_ascii_matlab_debug(rhs, "twoform_rhs", step_id));
    PetscCall(debug_check_vec_finite(rhs, "twoform_rhs_before_solve"));
    PetscCall(debug_check_vec_finite(sol2_new,
                                     "twoform_sol_initial_guess_before_zero"));
  }
#endif

  // 3. 求解线性系统，得到新的 2-形式解 sol2_new
  //    求解后会自动对 P3 压力进行均值归零处理
  PetscCall(VecZeroEntries(sol2_new));
#if DUAL_MAC_DEBUG
  PetscCall(debug_check_vec_finite(sol2_new, "twoform_sol_after_zero"));
#endif
  DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 子步线性求解开始\n");
  PetscCall(solve_linear_system_with_switch(
      A, rhs, sol2_new, dmSol_2, "two_", this->pinPressure, this->dt,
      this->stab_alpha, this->stab_gamma));
  DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 子步线性求解完成\n");

  // 4. 释放资源
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 完整的求解流程：初始条件设置 -> 1/2时刻解 -> Nt次时间推进
// 参数：
//   refSol: 参考解对象，用于设置初始条件
//   externalForce: 外力函数对象
PetscErrorCode DUAL_MAC::solve(RefSol refSol, ExternalForce externalForce) {
  PetscFunctionBeginUser;
  PetscCall(set_reference_solution(refSol));

  // ===== 1. 创建初始值向量 =====
  // u1_0, omega2_0 存储在 dmSol_1 中
  Vec u1_0, omega2_0;
  PetscCall(DMCreateGlobalVector(dmSol_1, &u1_0));
  PetscCall(DMCreateGlobalVector(dmSol_1, &omega2_0));

  // u2_0, omega1_0 存储在 dmSol_2 中
  Vec u2_0, omega1_0;
  PetscCall(DMCreateGlobalVector(dmSol_2, &u2_0));
  PetscCall(DMCreateGlobalVector(dmSol_2, &omega1_0));

  // ===== 2. 设置初始条件（t=0时刻）=====
  PetscCall(setup_initial_solution(refSol, u1_0, u2_0, omega1_0, omega2_0));

  // ===== 2b. 用离散旋度覆盖投影涡量，保证耦合方程一致性 =====
  // 连续层面 Π_face(∇×u) ≠ ∇_h × Π_edge(u)，直接投影参考解会导致
  // 初始涡量与耦合方程强制的离散旋度不一致，产生螺旋度跳变。
  PetscCall(
      compute_discrete_curl_edge_to_face(dmSol_1, u1_0, omega2_0, dx, dy, dz));
  PetscCall(
      compute_discrete_curl_face_to_edge(dmSol_2, u2_0, omega1_0, dx, dy, dz));

#if DUAL_MAC_DEBUG
  Evaluation debugEvaluator;
  const PetscReal debugCellVolume =
      (dx > 0.0 && dy > 0.0 && dz > 0.0) ? (dx * dy * dz) : -1.0;
  const PetscReal hx_dbg =
      (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  const PetscReal hy_dbg =
      (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  const PetscReal hz_dbg =
      (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 输出初值向量\n"));
  PetscCall(dump_vector_ascii_matlab_debug(u1_0, "initial_u1", 0));
  PetscCall(dump_vector_ascii_matlab_debug(u2_0, "initial_u2", 0));
  PetscCall(dump_vector_ascii_matlab_debug(omega1_0, "initial_omega1", 0));
  PetscCall(dump_vector_ascii_matlab_debug(omega2_0, "initial_omega2", 0));
#endif

  // ===== 3. 创建时间推进用的解向量 =====
  // sol1: (u1, omega2, P0) 在 dmSol_1 中
  // sol2: (u2, omega1, P3) 在 dmSol_2 中
  Vec sol1_old, sol1_new, sol2_old, sol2_new;
  PetscCall(DMCreateGlobalVector(dmSol_1, &sol1_old));
  PetscCall(DMCreateGlobalVector(dmSol_1, &sol1_new));
  PetscCall(DMCreateGlobalVector(dmSol_2, &sol2_old));
  PetscCall(DMCreateGlobalVector(dmSol_2, &sol2_new));

  // ===== 4. 初始化整数场 sol2_old (t=0) =====
  PetscCall(assemble_sol2_from_components(dmSol_2, u2_0, omega1_0, sol2_old));

  // ===== 5. 计算 1/2 时刻的解（隐式方法：用 Δt/2 求解标准 1-form 系统）=====
  // 用与正常半步方程 (5.4)-(5.6) 完全相同的结构，只是时间步长改为 Δt/2。
  // 两个场都从 t=0 的同一初始条件出发，第一个半步用 Δt/2 将半整数场
  // 从 t=0 推进到 t=Δt/2，建立交错时间网格。
  {
    Vec sol1_init;
    PetscCall(DMCreateGlobalVector(dmSol_1, &sol1_init));
    PetscCall(
        assemble_sol1_from_components(dmSol_1, u1_0, omega2_0, sol1_init));

    Mat A_startup;
    Vec rhs_startup;
    PetscCall(DMCreateMatrix(dmSol_1, &A_startup));
    PetscCall(DMCreateGlobalVector(dmSol_1, &rhs_startup));

    const PetscReal startup_dt = 0.5 * this->dt;
    PetscCall(assemble_1form_system_matrix(
        dmSol_1, dmSol_2, A_startup, rhs_startup, sol1_init, sol2_old,
        sol1_init, externalForce, 0.0, startup_dt));

    PetscCall(solve_linear_system_with_switch(
        A_startup, rhs_startup, sol1_old, dmSol_1, "one_", this->pinPressure,
        startup_dt, this->stab_alpha, this->stab_gamma));

    PetscCall(MatDestroy(&A_startup));
    PetscCall(VecDestroy(&rhs_startup));
    PetscCall(VecDestroy(&sol1_init));
  }
#if DUAL_MAC_DEBUG
  {
    Vec sol1_init_dbg = NULL;
    PetscCall(DMCreateGlobalVector(dmSol_1, &sol1_init_dbg));
    PetscCall(
        assemble_sol1_from_components(dmSol_1, u1_0, omega2_0, sol1_init_dbg));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[DEBUG] 输出初始整体状态向量(sol1_old/sol2_old)\n"));
    PetscCall(
        dump_vector_ascii_matlab_debug(sol1_old, "initial_sol1_half_state", 0));
    PetscCall(
        dump_vector_ascii_matlab_debug(sol2_old, "initial_sol2_state", 0));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 初值误差评估 (t=0)\n"));
    PetscCall(debugEvaluator.compute_error(dmSol_1, sol1_init_dbg, refSol, 0.0,
                                           0.0, debugCellVolume));
    PetscCall(debugEvaluator.compute_error(dmSol_2, sol2_old, refSol, 0.0, 0.0,
                                           debugCellVolume));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG] 1/2时刻误差评估 (t=%g)\n",
                          (double)(0.5 * this->dt)));
    PetscCall(debugEvaluator.compute_error(
        dmSol_1, sol1_old, refSol, 0.5 * this->dt, 0.0, debugCellVolume));
    PetscCall(debug_check_vec_finite(sol1_old, "sol1_old_after_half_assemble"));

    PetscCall(VecDestroy(&sol1_init_dbg));
  }
#endif

  // ===== 7. 时间推进循环 =====
  // 根据论文，时间推进顺序：
  //   1. 先推进 2-形式系统（整数时间步 k）
  //   2. 再推进 1-形式系统（半整数时间步 k+1/2）
  Evaluation invariantEvaluator;
  const PetscReal invariantCellVolume =
      (dx > 0.0 && dy > 0.0 && dz > 0.0) ? (dx * dy * dz) : -1.0;
  // 初始时刻不变量：使用 t=0 的 (u1,omega2) 与 (u2,omega1)
  Vec sol1_init_invariant = NULL;
  PetscCall(DMCreateGlobalVector(dmSol_1, &sol1_init_invariant));
  PetscCall(assemble_sol1_from_components(dmSol_1, u1_0, omega2_0,
                                          sol1_init_invariant));
  {
    InvariantResult inv0;
    PetscCall(invariantEvaluator.compute_invariants(
        dmSol_1, sol1_init_invariant, dmSol_2, sol2_old, invariantCellVolume,
        &inv0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[Invariant] initial, t=0: "
                          "H1=%.12e, H2=%.12e, K1=%.12e, K2=%.12e\n",
                          (double)inv0.H1, (double)inv0.H2, (double)inv0.K1,
                          (double)inv0.K2));
  }
  {
    InvariantResult inv_half;
    PetscCall(invariantEvaluator.compute_invariants(
        dmSol_1, sol1_old, dmSol_2, sol2_old, invariantCellVolume, &inv_half));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[Invariant] half-step, t2=0, t1=%.6f: "
                          "H1=%.12e, H2=%.12e, K1=%.12e, K2=%.12e\n",
                          (double)(0.5 * this->dt), (double)inv_half.H1,
                          (double)inv_half.H2, (double)inv_half.K1,
                          (double)inv_half.K2));
  }
  PetscReal current_time = 0.0;
  for (PetscInt k = 1; k <= this->Nt; ++k) {
    current_time = k * this->dt; // 当前整数时间步 k
    DUAL_MAC_DEBUG_LOG("[DEBUG] 时间步 %d 开始 (t=%g)\n", (int)k,
                       (double)current_time);

    // 7.1 推进 2-形式系统：从 k-1 到 k
    //     输入：sol2_old (k-1), sol1_old (k-1/2)
    //     输出：sol2_new (k)
    PetscCall(time_evolve2(dmSol_1, dmSol_2, sol1_old,
                           sol1_new,           // sol1_old 提供 omega2^{k-1/2}
                           sol2_old, sol2_new, // sol2_old -> sol2_new
                           externalForce, current_time));
    // 7.2 推进 1-形式系统：从 k-1/2 到 k+1/2
    //     输入：sol1_old (k-1/2), sol2_new (k) - 其中 omega1^k 用作对流场
    //     输出：sol1_new (k+1/2)
    //     外力时间按方程(5.4)使用整数时间步 k（即 current_time）
    PetscReal time_half = current_time + 0.5 * this->dt; // k+1/2 时刻
    PetscCall(time_evolve1(dmSol_1, dmSol_2, sol1_old,
                           sol1_new,           // sol1_old -> sol1_new
                           sol2_old, sol2_new, // sol2_new 提供 omega1^k
                           externalForce, current_time));

    // 7.3 计算并打印每一步不变量（螺旋度/动能）
    // H1 = <u1,omega1>*h^3, H2 = <u2,omega2>*h^3
    // K1 = 1/2<u1,u1>*h^3, K2 = 1/2<u2,u2>*h^3
    InvariantResult inv;
    PetscCall(invariantEvaluator.compute_invariants(
        dmSol_1, sol1_new, dmSol_2, sol2_new, invariantCellVolume, &inv,
        sol1_old));
    GradPressureOmegaInnerProductResult gpw;
    PetscCall(invariantEvaluator.compute_grad_pressure_omega_inner_products(
        dmSol_1, sol1_new, dmSol_2, sol2_new, invariantCellVolume, &gpw));
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "[Invariant] step=%" PetscInt_FMT ", t2=%.6f, t1=%.6f: "
        "H1=%.12e, H2=%.12e, K1=%.12e, K2=%.12e, "
        "<grad p3,omega2>=%.12e, <grad p0,omega1>=%.12e\n",
        k, (double)current_time, (double)time_half, (double)inv.H1,
        (double)inv.H2, (double)inv.K1, (double)inv.K2,
        (double)gpw.gradP3_dot_omega2, (double)gpw.gradP0_dot_omega1));
#if DUAL_MAC_DEBUG
    PetscCall(dump_vector_ascii_matlab_debug(sol2_new, "twoform_solution", k));
    PetscCall(dump_vector_ascii_matlab_debug(sol1_new, "oneform_solution", k));
    PetscCall(debug_report_curl_consistency(dmSol_1, sol1_new, dmSol_2,
                                            sol2_new, hx_dbg, hy_dbg, hz_dbg,
                                            debugCellVolume));
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD,
                    "[DEBUG] 时间步 %d 误差评估：2-form(t=%g), 1-form(t=%g)\n",
                    (int)k, (double)current_time, (double)time_half));
    PetscCall(debugEvaluator.compute_error(
        dmSol_2, sol2_new, refSol, current_time, current_time - 0.5 * this->dt,
        debugCellVolume));
    PetscCall(debugEvaluator.compute_error(dmSol_1, sol1_new, refSol, time_half,
                                           current_time, debugCellVolume));
#endif

    // 7.4 交换新旧解缓冲区，准备下一次迭代（避免整向量拷贝）
    PetscCall(VecSwap(sol1_old, sol1_new));
    PetscCall(VecSwap(sol2_old, sol2_new));
    DUAL_MAC_DEBUG_LOG("[DEBUG] 时间步 %d 完成\n", (int)k);

    // 可选：输出中间结果或计算误差
    // PetscCall(output_solution(...));
    // PetscCall(compute_error(current_time + 0.5 * this->dt));
  }

  // ===== 8. 缓存最终时刻解（用于误差计算）=====
  if (sol1_cached)
    PetscCall(VecDestroy(&sol1_cached));
  if (sol2_cached)
    PetscCall(VecDestroy(&sol2_cached));
  PetscCall(VecDuplicate(sol1_old, &sol1_cached));
  PetscCall(VecCopy(sol1_old, sol1_cached));
  PetscCall(VecDuplicate(sol2_old, &sol2_cached));
  PetscCall(VecCopy(sol2_old, sol2_cached));

  // ===== 9. 清理资源 =====
  PetscCall(VecDestroy(&u1_0));
  PetscCall(VecDestroy(&u2_0));
  PetscCall(VecDestroy(&omega1_0));
  PetscCall(VecDestroy(&omega2_0));
  PetscCall(VecDestroy(&sol1_old));
  PetscCall(VecDestroy(&sol1_new));
  PetscCall(VecDestroy(&sol2_old));
  PetscCall(VecDestroy(&sol2_new));
  PetscCall(VecDestroy(&sol1_init_invariant));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：从 u1 和 omega2 组装 sol1 向量（dmSol_1 中的完整解）
// sol1 包含：(u1, omega2, P0)，其中 P0 初始化为 0
PetscErrorCode DUAL_MAC::assemble_sol1_from_components(DM dmSol_1, Vec u1,
                                                       Vec omega2, Vec sol1) {
  PetscFunctionBeginUser;

  // 获取本地数组
  Vec localU1, localOmega2, localSol1;
  PetscScalar ****arrU1, ****arrOmega2, ****arrSol1;

  PetscCall(DMGetLocalVector(dmSol_1, &localU1));
  PetscCall(DMGlobalToLocal(dmSol_1, u1, INSERT_VALUES, localU1));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localU1, &arrU1));

  PetscCall(DMGetLocalVector(dmSol_1, &localOmega2));
  PetscCall(DMGlobalToLocal(dmSol_1, omega2, INSERT_VALUES, localOmega2));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localOmega2, &arrOmega2));

  PetscCall(DMGetLocalVector(dmSol_1, &localSol1));
  PetscCall(DMGlobalToLocal(dmSol_1, sol1, INSERT_VALUES, localSol1));
  PetscCall(DMStagVecGetArray(dmSol_1, localSol1, &arrSol1));

  // 获取 slot 索引
  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z));

  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));

  PetscInt slot_p0;
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK_DOWN_LEFT, 0, &slot_p0));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 复制 u1 和 omega2 到 sol1，P0 初始化为 0
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        // 复制 u1
        arrSol1[ez][ey][ex][slot_u1_x] = arrU1[ez][ey][ex][slot_u1_x];
        arrSol1[ez][ey][ex][slot_u1_y] = arrU1[ez][ey][ex][slot_u1_y];
        arrSol1[ez][ey][ex][slot_u1_z] = arrU1[ez][ey][ex][slot_u1_z];

        // 复制 omega2
        arrSol1[ez][ey][ex][slot_omega2_x] =
            arrOmega2[ez][ey][ex][slot_omega2_x];
        arrSol1[ez][ey][ex][slot_omega2_y] =
            arrOmega2[ez][ey][ex][slot_omega2_y];
        arrSol1[ez][ey][ex][slot_omega2_z] =
            arrOmega2[ez][ey][ex][slot_omega2_z];

        // P0 初始化为 0（压力在求解过程中会更新）
        arrSol1[ez][ey][ex][slot_p0] = 0.0;
      }
    }
  }

  // 写回全局向量
  PetscCall(DMLocalToGlobal(dmSol_1, localSol1, INSERT_VALUES, sol1));

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localU1, &arrU1));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localU1));
  PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localOmega2, &arrOmega2));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2));
  PetscCall(DMStagVecRestoreArray(dmSol_1, localSol1, &arrSol1));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localSol1));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// 辅助函数：从 u2 和 omega1 组装 sol2 向量（dmSol_2 中的完整解）
// sol2 包含：(u2, omega1, P3)，其中 P3 初始化为 0
PetscErrorCode DUAL_MAC::assemble_sol2_from_components(DM dmSol_2, Vec u2,
                                                       Vec omega1, Vec sol2) {
  PetscFunctionBeginUser;

  // 获取本地数组
  Vec localU2, localOmega1, localSol2;
  PetscScalar ****arrU2, ****arrOmega1, ****arrSol2;

  PetscCall(DMGetLocalVector(dmSol_2, &localU2));
  PetscCall(DMGlobalToLocal(dmSol_2, u2, INSERT_VALUES, localU2));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localU2, &arrU2));

  PetscCall(DMGetLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMGlobalToLocal(dmSol_2, omega1, INSERT_VALUES, localOmega1));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localOmega1, &arrOmega1));

  PetscCall(DMGetLocalVector(dmSol_2, &localSol2));
  PetscCall(DMGlobalToLocal(dmSol_2, sol2, INSERT_VALUES, localSol2));
  PetscCall(DMStagVecGetArray(dmSol_2, localSol2, &arrSol2));

  // 获取 slot 索引
  PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, LEFT, 0, &slot_u2_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN, 0, &slot_u2_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK, 0, &slot_u2_z));

  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z));

  PetscInt slot_p3;
  PetscCall(DMStagGetLocationSlot(dmSol_2, ELEMENT, 0, &slot_p3));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 复制 u2 和 omega1 到 sol2，P3 初始化为 0
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        // 复制 u2
        arrSol2[ez][ey][ex][slot_u2_x] = arrU2[ez][ey][ex][slot_u2_x];
        arrSol2[ez][ey][ex][slot_u2_y] = arrU2[ez][ey][ex][slot_u2_y];
        arrSol2[ez][ey][ex][slot_u2_z] = arrU2[ez][ey][ex][slot_u2_z];

        // 复制 omega1
        arrSol2[ez][ey][ex][slot_omega1_x] =
            arrOmega1[ez][ey][ex][slot_omega1_x];
        arrSol2[ez][ey][ex][slot_omega1_y] =
            arrOmega1[ez][ey][ex][slot_omega1_y];
        arrSol2[ez][ey][ex][slot_omega1_z] =
            arrOmega1[ez][ey][ex][slot_omega1_z];

        // P3 初始化为 0（压力在求解过程中会更新）
        arrSol2[ez][ey][ex][slot_p3] = 0.0;
      }
    }
  }

  // 写回全局向量
  PetscCall(DMLocalToGlobal(dmSol_2, localSol2, INSERT_VALUES, sol2));

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localU2, &arrU2));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localU2));
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localOmega1, &arrOmega1));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMStagVecRestoreArray(dmSol_2, localSol2, &arrSol2));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localSol2));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#include "../include/evaluation.h"
#include "../include/DUAL_MAC.h"

// 本文件实现基于 DMStag 交错网格的 L2 误差评估。
// 同一套接口可兼容两类子系统：
//   - 1-form 系统（dmSol_1）：u 在棱上，omega 在面上，p 为顶点 p0
//   - 2-form 系统（dmSol_2）：u 在面上，omega 在棱上，p 为单元中心 p3
// 实现中会根据 DOF 配置自动判断当前布局并选择相应的位置映射。

namespace {
/**
 * @brief 从 DM 网格信息中计算单元体积 dx*dy*dz。
 *
 * @param[in]  dmSol       当前解向量对应的 DMStag 网格对象。
 * @param[out] cellVolume  返回单元体积（L2 误差缩放因子）。
 *
 * @note 离散 L2 范数采用：sqrt( sum_i |e_i|^2 * cellVolume )。
 */
PetscErrorCode GetCellVolumeFromDM(DM dmSol, PetscReal *cellVolume) {
  PetscFunctionBeginUser;

  // 全局逻辑网格尺寸。对均匀网格可直接恢复步长。
  PetscInt Nx, Ny, Nz;
  PetscCall(DMStagGetGlobalSizes(dmSol, &Nx, &Ny, &Nz));

  // 坐标场包围盒，结合 (Nx,Ny,Nz) 计算 (dx,dy,dz)。
  PetscReal xyzMin[3], xyzMax[3];
  PetscCall(DMGetBoundingBox(dmSol, xyzMin, xyzMax));

  const PetscReal xmin = xyzMin[0];
  const PetscReal xmax = xyzMax[0];
  const PetscReal ymin = xyzMin[1];
  const PetscReal ymax = xyzMax[1];
  const PetscReal zmin = xyzMin[2];
  const PetscReal zmax = xyzMax[2];

  const PetscReal dx = (xmax - xmin) / static_cast<PetscReal>(Nx);
  const PetscReal dy = (ymax - ymin) / static_cast<PetscReal>(Ny);
  const PetscReal dz = (zmax - zmin) / static_cast<PetscReal>(Nz);

  // 离散 L2 的体积权重：
  // ||e||_L2 ~= sqrt( sum_i |e_i|^2 * (dx*dy*dz) )
  *cellVolume = dx * dy * dz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeInvariantSums(DM dmSol1, Vec sol1, DM dmSol2, Vec sol2,
                                    PetscReal *sumH1, PetscReal *sumH2,
                                    PetscReal *sumK1, PetscReal *sumK2,
                                    Vec sol1_prev = NULL) {
  PetscFunctionBeginUser;
  if (sumH1)
    *sumH1 = 0.0;
  if (sumH2)
    *sumH2 = 0.0;
  if (sumK1)
    *sumK1 = 0.0;
  if (sumK2)
    *sumK2 = 0.0;

  PetscInt dof0_1, dof1_1, dof2_1, dof3_1;
  PetscInt dof0_2, dof1_2, dof2_2, dof3_2;
  PetscCall(DMStagGetDOF(dmSol1, &dof0_1, &dof1_1, &dof2_1, &dof3_1));
  PetscCall(DMStagGetDOF(dmSol2, &dof0_2, &dof1_2, &dof2_2, &dof3_2));
  if (!(dof0_1 > 0 && dof0_2 == 0)) {
    SETERRQ(PetscObjectComm((PetscObject)dmSol1), PETSC_ERR_ARG_WRONG,
            "compute_invariants expects dmSol1=1-form and dmSol2=2-form");
  }

  PetscInt sx1, sy1, sz1, nx1, ny1, nz1;
  PetscInt sx2, sy2, sz2, nx2, ny2, nz2;
  PetscCall(DMStagGetCorners(dmSol1, &sx1, &sy1, &sz1, &nx1, &ny1, &nz1, NULL,
                             NULL, NULL));
  PetscCall(DMStagGetCorners(dmSol2, &sx2, &sy2, &sz2, &nx2, &ny2, &nz2, NULL,
                             NULL, NULL));
  if (sx1 != sx2 || sy1 != sy2 || sz1 != sz2 || nx1 != nx2 || ny1 != ny2 ||
      nz1 != nz2) {
    SETERRQ(PetscObjectComm((PetscObject)dmSol1), PETSC_ERR_ARG_WRONG,
            "dmSol1 and dmSol2 local decomposition mismatch");
  }

  Vec localSol1 = NULL, localSol2 = NULL;
  PetscScalar ****arrSol1 = NULL, ****arrSol2 = NULL;
  PetscCall(DMGetLocalVector(dmSol1, &localSol1));
  PetscCall(DMGlobalToLocal(dmSol1, sol1, INSERT_VALUES, localSol1));
  PetscCall(DMStagVecGetArrayRead(dmSol1, localSol1, &arrSol1));

  PetscCall(DMGetLocalVector(dmSol2, &localSol2));
  PetscCall(DMGlobalToLocal(dmSol2, sol2, INSERT_VALUES, localSol2));
  PetscCall(DMStagVecGetArrayRead(dmSol2, localSol2, &arrSol2));

  Vec localSol1Prev = NULL;
  PetscScalar ****arrSol1Prev = NULL;
  if (sol1_prev) {
    PetscCall(DMGetLocalVector(dmSol1, &localSol1Prev));
    PetscCall(DMGlobalToLocal(dmSol1, sol1_prev, INSERT_VALUES, localSol1Prev));
    PetscCall(DMStagVecGetArrayRead(dmSol1, localSol1Prev, &arrSol1Prev));
  }

  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;

  PetscCall(DMStagGetLocationSlot(dmSol1, BACK_DOWN, 0, &slot_u1_x));
  PetscCall(DMStagGetLocationSlot(dmSol1, BACK_LEFT, 0, &slot_u1_y));
  PetscCall(DMStagGetLocationSlot(dmSol1, DOWN_LEFT, 0, &slot_u1_z));

  PetscCall(DMStagGetLocationSlot(dmSol2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol2, DOWN_LEFT, 0, &slot_omega1_z));

  PetscCall(DMStagGetLocationSlot(dmSol2, LEFT, 0, &slot_u2_x));
  PetscCall(DMStagGetLocationSlot(dmSol2, DOWN, 0, &slot_u2_y));
  PetscCall(DMStagGetLocationSlot(dmSol2, BACK, 0, &slot_u2_z));

  PetscCall(DMStagGetLocationSlot(dmSol1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol1, BACK, 0, &slot_omega2_z));

  PetscReal localH1 = 0.0, localH2 = 0.0;
  PetscReal localK1 = 0.0, localK2 = 0.0;
  for (PetscInt ez = sz1; ez < sz1 + nz1; ++ez) {
    for (PetscInt ey = sy1; ey < sy1 + ny1; ++ey) {
      for (PetscInt ex = sx1; ex < sx1 + nx1; ++ex) {
        const PetscScalar u1x = arrSol1[ez][ey][ex][slot_u1_x];
        const PetscScalar u1y = arrSol1[ez][ey][ex][slot_u1_y];
        const PetscScalar u1z = arrSol1[ez][ey][ex][slot_u1_z];
        const PetscScalar o1x = arrSol2[ez][ey][ex][slot_omega1_x];
        const PetscScalar o1y = arrSol2[ez][ey][ex][slot_omega1_y];
        const PetscScalar o1z = arrSol2[ez][ey][ex][slot_omega1_z];

        const PetscScalar u2x = arrSol2[ez][ey][ex][slot_u2_x];
        const PetscScalar u2y = arrSol2[ez][ey][ex][slot_u2_y];
        const PetscScalar u2z = arrSol2[ez][ey][ex][slot_u2_z];
        const PetscScalar o2x = arrSol1[ez][ey][ex][slot_omega2_x];
        const PetscScalar o2y = arrSol1[ez][ey][ex][slot_omega2_y];
        const PetscScalar o2z = arrSol1[ez][ey][ex][slot_omega2_z];

        if (arrSol1Prev) {
          const PetscScalar u1x_p = arrSol1Prev[ez][ey][ex][slot_u1_x];
          const PetscScalar u1y_p = arrSol1Prev[ez][ey][ex][slot_u1_y];
          const PetscScalar u1z_p = arrSol1Prev[ez][ey][ex][slot_u1_z];
          const PetscScalar o2x_p = arrSol1Prev[ez][ey][ex][slot_omega2_x];
          const PetscScalar o2y_p = arrSol1Prev[ez][ey][ex][slot_omega2_y];
          const PetscScalar o2z_p = arrSol1Prev[ez][ey][ex][slot_omega2_z];
          localH1 += 0.5 * PetscRealPart((u1x + u1x_p) * PetscConj(o1x) +
                                         (u1y + u1y_p) * PetscConj(o1y) +
                                         (u1z + u1z_p) * PetscConj(o1z));
          localH2 += 0.5 * PetscRealPart(u2x * PetscConj(o2x + o2x_p) +
                                         u2y * PetscConj(o2y + o2y_p) +
                                         u2z * PetscConj(o2z + o2z_p));
        } else {
          localH1 += PetscRealPart(u1x * PetscConj(o1x) + u1y * PetscConj(o1y) +
                                   u1z * PetscConj(o1z));
          localH2 += PetscRealPart(u2x * PetscConj(o2x) + u2y * PetscConj(o2y) +
                                   u2z * PetscConj(o2z));
        }

        localK1 += PetscRealPart(u1x * PetscConj(u1x) + u1y * PetscConj(u1y) +
                                 u1z * PetscConj(u1z));
        localK2 += PetscRealPart(u2x * PetscConj(u2x) + u2y * PetscConj(u2y) +
                                 u2z * PetscConj(u2z));
      }
    }
  }

  PetscReal globalH1 = 0.0, globalH2 = 0.0, globalK1 = 0.0, globalK2 = 0.0;
  MPI_Comm comm = PetscObjectComm((PetscObject)dmSol1);
  PetscCall(PMPI_Allreduce(&localH1, &globalH1, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCall(PMPI_Allreduce(&localH2, &globalH2, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCall(PMPI_Allreduce(&localK1, &globalK1, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCall(PMPI_Allreduce(&localK2, &globalK2, 1, MPIU_REAL, MPIU_SUM, comm));

  if (sumH1)
    *sumH1 = globalH1;
  if (sumH2)
    *sumH2 = globalH2;
  if (sumK1)
    *sumK1 = globalK1;
  if (sumK2)
    *sumK2 = globalK2;

  PetscCall(DMStagVecRestoreArrayRead(dmSol1, localSol1, &arrSol1));
  PetscCall(DMRestoreLocalVector(dmSol1, &localSol1));
  PetscCall(DMStagVecRestoreArrayRead(dmSol2, localSol2, &arrSol2));
  PetscCall(DMRestoreLocalVector(dmSol2, &localSol2));
  if (arrSol1Prev) {
    PetscCall(DMStagVecRestoreArrayRead(dmSol1, localSol1Prev, &arrSol1Prev));
    PetscCall(DMRestoreLocalVector(dmSol1, &localSol1Prev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

/**
 * @brief 默认构造函数。
 */
Evaluation::Evaluation() = default;

/**
 * @brief 默认析构函数。
 */
Evaluation::~Evaluation() = default;

PetscErrorCode Evaluation::compute_helicity(DM dmSol1, Vec sol1, DM dmSol2,
                                            Vec sol2, PetscReal gridScale,
                                            PetscReal *H1Out,
                                            PetscReal *H2Out) {
  PetscFunctionBeginUser;
  PetscReal sumH1 = 0.0, sumH2 = 0.0;
  PetscCall(ComputeInvariantSums(dmSol1, sol1, dmSol2, sol2, &sumH1, &sumH2,
                                 NULL, NULL));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol1, &cellScale));
  if (H1Out)
    *H1Out = sumH1 * cellScale;
  if (H2Out)
    *H2Out = sumH2 * cellScale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluation::compute_kinetic_energy(DM dmSol1, Vec sol1,
                                                  DM dmSol2, Vec sol2,
                                                  PetscReal gridScale,
                                                  PetscReal *K1Out,
                                                  PetscReal *K2Out) {
  PetscFunctionBeginUser;
  PetscReal sumK1 = 0.0, sumK2 = 0.0;
  PetscCall(ComputeInvariantSums(dmSol1, sol1, dmSol2, sol2, NULL, NULL, &sumK1,
                                 &sumK2));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol1, &cellScale));
  if (K1Out)
    *K1Out = 0.5 * sumK1 * cellScale;
  if (K2Out)
    *K2Out = 0.5 * sumK2 * cellScale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluation::compute_invariants(DM dmSol1, Vec sol1, DM dmSol2,
                                              Vec sol2, PetscReal gridScale,
                                              InvariantResult *result,
                                              Vec sol1_prev) {
  PetscFunctionBeginUser;
  PetscReal sumH1 = 0.0, sumH2 = 0.0, sumK1 = 0.0, sumK2 = 0.0;
  PetscCall(ComputeInvariantSums(dmSol1, sol1, dmSol2, sol2, &sumH1, &sumH2,
                                 &sumK1, &sumK2, sol1_prev));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol1, &cellScale));

  const PetscReal H1 = sumH1 * cellScale;
  const PetscReal H2 = sumH2 * cellScale;
  const PetscReal K1 = 0.5 * sumK1 * cellScale;
  const PetscReal K2 = 0.5 * sumK2 * cellScale;

  if (result) {
    result->H1 = H1;
    result->H2 = H2;
    result->K1 = K1;
    result->K2 = K2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluation::compute_grad_pressure_omega_inner_products(
    DM dmSol1, Vec sol1, DM dmSol2, Vec sol2, PetscReal gridScale,
    GradPressureOmegaInnerProductResult *result) {
  PetscFunctionBeginUser;
  PetscReal gradP3DotOmega2 = 0.0;
  PetscReal gradP0DotOmega1 = 0.0;

  PetscInt dof0_1, dof1_1, dof2_1, dof3_1;
  PetscInt dof0_2, dof1_2, dof2_2, dof3_2;
  PetscCall(DMStagGetDOF(dmSol1, &dof0_1, &dof1_1, &dof2_1, &dof3_1));
  PetscCall(DMStagGetDOF(dmSol2, &dof0_2, &dof1_2, &dof2_2, &dof3_2));
  if (!(dof0_1 > 0 && dof0_2 == 0)) {
    SETERRQ(PetscObjectComm((PetscObject)dmSol1), PETSC_ERR_ARG_WRONG,
            "compute_grad_pressure_omega_inner_products expects dmSol1=1-form "
            "and dmSol2=2-form");
  }

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  Vec localSol1 = NULL, localSol2 = NULL;
  PetscScalar ****arrSol1 = NULL, ****arrSol2 = NULL;
  PetscCall(DMGetLocalVector(dmSol1, &localSol1));
  PetscCall(DMGlobalToLocal(dmSol1, sol1, INSERT_VALUES, localSol1));
  PetscCall(DMStagVecGetArrayRead(dmSol1, localSol1, &arrSol1));
  PetscCall(DMGetLocalVector(dmSol2, &localSol2));
  PetscCall(DMGlobalToLocal(dmSol2, sol2, INSERT_VALUES, localSol2));
  PetscCall(DMStagVecGetArrayRead(dmSol2, localSol2, &arrSol2));

  // 从 product 坐标读取均匀网格步长，避免 DMGetBoundingBox(部分 DMStag 不支持)
  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol1, &cArrX, &cArrY, &cArrZ));
  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol1, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol1, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol1, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol1, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol1, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol1, LEFT, &icz_prev));
  PetscReal hx = 2.0 * PetscAbsReal(PetscRealPart(cArrX[startx][icx_center] -
                                                  cArrX[startx][icx_prev]));
  PetscReal hy = 2.0 * PetscAbsReal(PetscRealPart(cArrY[starty][icy_center] -
                                                  cArrY[starty][icy_prev]));
  PetscReal hz = 2.0 * PetscAbsReal(PetscRealPart(cArrZ[startz][icz_center] -
                                                  cArrZ[startz][icz_prev]));
  if (hx <= 0.0 || hy <= 0.0 || hz <= 0.0) {
    SETERRQ(PetscObjectComm((PetscObject)dmSol1), PETSC_ERR_ARG_WRONG,
            "Invalid uniform spacing inferred from product coordinates");
  }

  PetscInt slot_p0, slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscInt slot_p3, slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(DMStagGetLocationSlot(dmSol1, BACK_DOWN_LEFT, 0, &slot_p0));
  PetscCall(DMStagGetLocationSlot(dmSol1, LEFT, 0, &slot_omega2_x));
  PetscCall(DMStagGetLocationSlot(dmSol1, DOWN, 0, &slot_omega2_y));
  PetscCall(DMStagGetLocationSlot(dmSol1, BACK, 0, &slot_omega2_z));
  PetscCall(DMStagGetLocationSlot(dmSol2, ELEMENT, 0, &slot_p3));
  PetscCall(DMStagGetLocationSlot(dmSol2, BACK_DOWN, 0, &slot_omega1_x));
  PetscCall(DMStagGetLocationSlot(dmSol2, BACK_LEFT, 0, &slot_omega1_y));
  PetscCall(DMStagGetLocationSlot(dmSol2, DOWN_LEFT, 0, &slot_omega1_z));

  PetscReal localSumP3O2 = 0.0, localSumP0O1 = 0.0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        // grad(p0) on 1-form edges (forward differences)
        const PetscScalar p0 = arrSol1[ez][ey][ex][slot_p0];
        const PetscScalar gradp0_x =
            (arrSol1[ez][ey][ex + 1][slot_p0] - p0) / hx;
        const PetscScalar gradp0_y =
            (arrSol1[ez][ey + 1][ex][slot_p0] - p0) / hy;
        const PetscScalar gradp0_z =
            (arrSol1[ez + 1][ey][ex][slot_p0] - p0) / hz;

        const PetscScalar o1x = arrSol2[ez][ey][ex][slot_omega1_x];
        const PetscScalar o1y = arrSol2[ez][ey][ex][slot_omega1_y];
        const PetscScalar o1z = arrSol2[ez][ey][ex][slot_omega1_z];
        localSumP0O1 += PetscRealPart(gradp0_x * PetscConj(o1x) +
                                      gradp0_y * PetscConj(o1y) +
                                      gradp0_z * PetscConj(o1z));

        // grad(p3) on 2-form faces (backward differences)
        const PetscScalar p3 = arrSol2[ez][ey][ex][slot_p3];
        const PetscScalar gradp3_x =
            (p3 - arrSol2[ez][ey][ex - 1][slot_p3]) / hx;
        const PetscScalar gradp3_y =
            (p3 - arrSol2[ez][ey - 1][ex][slot_p3]) / hy;
        const PetscScalar gradp3_z =
            (p3 - arrSol2[ez - 1][ey][ex][slot_p3]) / hz;

        const PetscScalar o2x = arrSol1[ez][ey][ex][slot_omega2_x];
        const PetscScalar o2y = arrSol1[ez][ey][ex][slot_omega2_y];
        const PetscScalar o2z = arrSol1[ez][ey][ex][slot_omega2_z];
        localSumP3O2 += PetscRealPart(gradp3_x * PetscConj(o2x) +
                                      gradp3_y * PetscConj(o2y) +
                                      gradp3_z * PetscConj(o2z));
      }
    }
  }

  MPI_Comm comm = PetscObjectComm((PetscObject)dmSol1);
  PetscReal globalSumP3O2 = 0.0, globalSumP0O1 = 0.0;
  PetscCall(PMPI_Allreduce(&localSumP3O2, &globalSumP3O2, 1, MPIU_REAL,
                           MPIU_SUM, comm));
  PetscCall(PMPI_Allreduce(&localSumP0O1, &globalSumP0O1, 1, MPIU_REAL,
                           MPIU_SUM, comm));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol1, &cellScale));
  gradP3DotOmega2 = globalSumP3O2 * cellScale;
  gradP0DotOmega1 = globalSumP0O1 * cellScale;

  if (result) {
    result->gradP3_dot_omega2 = gradP3DotOmega2;
    result->gradP0_dot_omega1 = gradP0DotOmega1;
  }

  PetscCall(DMStagVecRestoreArrayRead(dmSol1, localSol1, &arrSol1));
  PetscCall(DMRestoreLocalVector(dmSol1, &localSol1));
  PetscCall(DMStagVecRestoreArrayRead(dmSol2, localSol2, &arrSol2));
  PetscCall(DMRestoreLocalVector(dmSol2, &localSol2));
  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dmSol1, &cArrX, &cArrY, &cArrZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief 统一入口：依次计算速度、涡量和压力的 L2 误差。
 *
 * @param[in] dmSol      求解网格（dmSol_1 或 dmSol_2）。
 * @param[in] sol        数值解向量。
 * @param[in] refSol     参考解对象。
 * @param[in] time       误差评估时刻。
 * @param[in] gridScale  网格缩放因子；<=0 时自动使用 dx*dy*dz。
 */
PetscErrorCode Evaluation::compute_error(DM dmSol, Vec sol, RefSol refSol,
                                         PetscReal time_u_omega,
                                         PetscReal time_p, PetscReal gridScale,
                                         EvaluationResult *result) {
  PetscFunctionBeginUser;
  // 在同一交错网格上分别计算三类物理量的 L2 误差。
  PetscReal err_u = 0.0, div_l2 = 0.0, div_linf = 0.0;
  PetscReal err_omega = 0.0, err_p = 0.0;
  PetscReal err_grad_p = 0.0;
  PetscCall(compute_error_u(dmSol, sol, refSol, time_u_omega, gridScale, &err_u,
                            &div_l2, &div_linf));
  PetscCall(compute_error_omega(dmSol, sol, refSol, time_u_omega, gridScale,
                                &err_omega));
  PetscCall(compute_error_p(dmSol, sol, refSol, time_p, gridScale, &err_p));
  PetscCall(
      compute_error_grad_p(dmSol, sol, refSol, time_p, gridScale, &err_grad_p));
  if (result) {
    result->err_u = err_u;
    result->err_omega = err_omega;
    result->err_p = err_p;
    result->err_grad_p = err_grad_p;
    result->div_u_l2 = div_l2;
    result->div_u_linf = div_linf;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluation::diagnose_pressure_three_times(
    DM dmSol, Vec sol, RefSol refSol, PetscReal t_center, PetscReal dt,
    PetscReal gridScale, const char *label) {
  PetscFunctionBeginUser;
  const PetscReal t_minus = t_center - 0.5 * dt;
  const PetscReal t_zero = t_center;
  const PetscReal t_plus = t_center + 0.5 * dt;

  PetscReal err_minus = 0.0, err_zero = 0.0, err_plus = 0.0;
  PetscCall(
      compute_error_p(dmSol, sol, refSol, t_minus, gridScale, &err_minus));
  PetscCall(compute_error_p(dmSol, sol, refSol, t_zero, gridScale, &err_zero));
  PetscCall(compute_error_p(dmSol, sol, refSol, t_plus, gridScale, &err_plus));

  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol, &dof0, &dof1, &dof2, &dof3));
  const char *pname = (dof0 > 0) ? "p0" : "p3";
  const char *tag = label ? label : pname;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "[Pressure-3Time-Diag] %s (%s):\n"
                        "  t-dt/2 = %.6f -> L2 = %.12e\n"
                        "  t      = %.6f -> L2 = %.12e\n"
                        "  t+dt/2 = %.6f -> L2 = %.12e\n",
                        tag, pname, (double)t_minus, (double)err_minus,
                        (double)t_zero, (double)err_zero, (double)t_plus,
                        (double)err_plus));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief 计算速度向量 u 的 L2 误差。
 *
 * @param[in] dmSol      当前系统对应 DM（1-form 或 2-form）。
 * @param[in] sol        数值解向量。
 * @param[in] refSol     参考解对象。
 * @param[in] time       时间。
 * @param[in] gridScale  网格缩放因子；<=0 时自动计算单元体积。
 *
 * @note 误差定义：L2(u)=sqrt(sum(|u_h-u_ref|^2)*gridScale)。
 */
PetscErrorCode Evaluation::compute_error_u(DM dmSol, Vec sol, RefSol refSol,
                                           PetscReal time, PetscReal gridScale,
                                           PetscReal *l2ErrOut,
                                           PetscReal *divL2Out,
                                           PetscReal *divLinfOut) {
  PetscFunctionBeginUser;

  // 布局识别：
  // dmSol_1 通常 dof0>0（含顶点压力 p0）；
  // dmSol_2 通常 dof0==0（压力为单元中心 p3）。
  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol, &dof0, &dof1, &dof2, &dof3));
  const PetscBool is1FormSystem = (dof0 > 0) ? PETSC_TRUE : PETSC_FALSE;

  // product 坐标数组（x/y/z 各方向一维坐标）。
  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));

  Vec localSol;
  PetscScalar ****arrSol;
  PetscCall(DMGetLocalVector(dmSol, &localSol));
  PetscCall(DMGlobalToLocal(dmSol, sol, INSERT_VALUES, localSol));
  PetscCall(DMStagVecGetArrayRead(dmSol, localSol, &arrSol));

  // 本进程拥有的局部区域（误差累加时只遍历拥有区）。
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 坐标槽位：
  //   ELEMENT -> 各方向中心坐标
  //   LEFT/DOWN/BACK -> x/y/z 方向“前一个面”的坐标
  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icz_prev));

  // 速度分量槽位与系统类型相关：
  //   1-form：速度位于棱（BACK_DOWN, BACK_LEFT, DOWN_LEFT）
  //   2-form：速度位于面（LEFT, DOWN, BACK）
  PetscInt slot_x, slot_y, slot_z;
  if (is1FormSystem) {
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_DOWN, 0, &slot_x));
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_LEFT, 0, &slot_y));
    PetscCall(DMStagGetLocationSlot(dmSol, DOWN_LEFT, 0, &slot_z));
  } else {
    PetscCall(DMStagGetLocationSlot(dmSol, LEFT, 0, &slot_x));
    PetscCall(DMStagGetLocationSlot(dmSol, DOWN, 0, &slot_y));
    PetscCall(DMStagGetLocationSlot(dmSol, BACK, 0, &slot_z));
  }

  PetscReal localSqErr = 0.0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        PetscScalar x, y, z;
        PetscScalar dux, duy, duz;

        // 在每个离散自由度的真实几何位置取参考解，并与数值解做差。
        if (is1FormSystem) {
          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_prev];
          dux = arrSol[ez][ey][ex][slot_x] - refSol.uxRef(x, y, z, time);

          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_prev];
          duy = arrSol[ez][ey][ex][slot_y] - refSol.uyRef(x, y, z, time);

          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_center];
          duz = arrSol[ez][ey][ex][slot_z] - refSol.uzRef(x, y, z, time);
        } else {
          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_center];
          dux = arrSol[ez][ey][ex][slot_x] - refSol.uxRef(x, y, z, time);

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_center];
          duy = arrSol[ez][ey][ex][slot_y] - refSol.uyRef(x, y, z, time);

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_prev];
          duz = arrSol[ez][ey][ex][slot_z] - refSol.uzRef(x, y, z, time);
        }

        // 累加 |du|^2 = |dux|^2 + |duy|^2 + |duz|^2。
        // 使用 PetscRealPart/PetscConj 保证对实数/复数 PetscScalar 都安全。
        localSqErr += PetscRealPart(
            dux * PetscConj(dux) + duy * PetscConj(duy) + duz * PetscConj(duz));
      }
    }
  }

  // 同步输出速度场散度诊断，便于检查离散不可压缩约束是否满足：
  // - 1-form(u1): 使用与 assemble_u1_divergence_matrix 一致的后向差分写法
  // - 2-form(u2): 使用与 assemble_u2_divergence_matrix 一致的前向差分写法
  //
  // 这里假设网格均匀：hx = 2*(x_center - x_prev), hy/hz 同理。
  // 直接从 product 坐标读取，避免调用 DMGetBoundingBox(某些 DMSTAG
  // 构建不支持)。
  PetscReal dx = 2.0 * PetscAbsReal(PetscRealPart(cArrX[startx][icx_center] -
                                                  cArrX[startx][icx_prev]));
  PetscReal dy = 2.0 * PetscAbsReal(PetscRealPart(cArrY[starty][icy_center] -
                                                  cArrY[starty][icy_prev]));
  PetscReal dz = 2.0 * PetscAbsReal(PetscRealPart(cArrZ[startz][icz_center] -
                                                  cArrZ[startz][icz_prev]));
  if (dx <= 0.0 || dy <= 0.0 || dz <= 0.0) {
    SETERRQ(PetscObjectComm((PetscObject)dmSol), PETSC_ERR_ARG_WRONG,
            "Invalid uniform spacing inferred from product coordinates");
  }

  PetscReal localSqDiv = 0.0;
  PetscReal localMaxAbsDiv = 0.0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        PetscReal divVal = 0.0;

        if (is1FormSystem) {
          // div(u1) at BACK_DOWN_LEFT vertex:
          // (u1_x(ex)-u1_x(ex-1))/dx + (u1_y(ey)-u1_y(ey-1))/dy +
          // (u1_z(ez)-u1_z(ez-1))/dz
          divVal = (PetscRealPart(arrSol[ez][ey][ex][slot_x]) -
                    PetscRealPart(arrSol[ez][ey][ex - 1][slot_x])) /
                       dx +
                   (PetscRealPart(arrSol[ez][ey][ex][slot_y]) -
                    PetscRealPart(arrSol[ez][ey - 1][ex][slot_y])) /
                       dy +
                   (PetscRealPart(arrSol[ez][ey][ex][slot_z]) -
                    PetscRealPart(arrSol[ez - 1][ey][ex][slot_z])) /
                       dz;
        } else {
          // div(u2) at ELEMENT center:
          // (u2_x(ex+1)-u2_x(ex))/dx + (u2_y(ey+1)-u2_y(ey))/dy +
          // (u2_z(ez+1)-u2_z(ez))/dz
          divVal = (PetscRealPart(arrSol[ez][ey][ex + 1][slot_x]) -
                    PetscRealPart(arrSol[ez][ey][ex][slot_x])) /
                       dx +
                   (PetscRealPart(arrSol[ez][ey + 1][ex][slot_y]) -
                    PetscRealPart(arrSol[ez][ey][ex][slot_y])) /
                       dy +
                   (PetscRealPart(arrSol[ez + 1][ey][ex][slot_z]) -
                    PetscRealPart(arrSol[ez][ey][ex][slot_z])) /
                       dz;
        }

        localSqDiv += divVal * divVal;
        const PetscReal absDiv = PetscAbsReal(divVal);
        if (absDiv > localMaxAbsDiv)
          localMaxAbsDiv = absDiv;
      }
    }
  }

  // MPI 全局归约，得到所有进程上的误差平方和。
  PetscReal globalSqErr = 0.0;
  PetscCall(PMPI_Allreduce(&localSqErr, &globalSqErr, 1, MPIU_REAL, MPIU_SUM,
                           PetscObjectComm((PetscObject)dmSol)));
  PetscReal globalSqDiv = 0.0;
  PetscReal globalMaxAbsDiv = 0.0;
  PetscCall(PMPI_Allreduce(&localSqDiv, &globalSqDiv, 1, MPIU_REAL, MPIU_SUM,
                           PetscObjectComm((PetscObject)dmSol)));
  PetscCall(PMPI_Allreduce(&localMaxAbsDiv, &globalMaxAbsDiv, 1, MPIU_REAL,
                           MPIU_MAX, PetscObjectComm((PetscObject)dmSol)));

  // 若未显式提供缩放因子，则自动使用单元体积。
  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol, &cellScale));
  // 按要求乘网格大小后开方，得到离散 L2 误差。
  const PetscReal l2Err = PetscSqrtReal(globalSqErr * cellScale);
  const PetscReal l2Div = PetscSqrtReal(globalSqDiv * cellScale);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error of %s = %.12e\n",
                        is1FormSystem ? "u1" : "u2", (double)l2Err));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "Divergence diagnostics of %s: L2 = %.12e, Linf = %.12e\n",
      is1FormSystem ? "u1" : "u2", (double)l2Div, (double)globalMaxAbsDiv));
  if (l2ErrOut)
    *l2ErrOut = l2Err;
  if (divL2Out)
    *divL2Out = l2Div;
  if (divLinfOut)
    *divLinfOut = globalMaxAbsDiv;

  PetscCall(DMStagVecRestoreArrayRead(dmSol, localSol, &arrSol));
  PetscCall(DMRestoreLocalVector(dmSol, &localSol));
  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief 计算涡量 omega 的 L2 误差。
 *
 * @param[in] dmSol      当前系统对应 DM（1-form 或 2-form）。
 * @param[in] sol        数值解向量。
 * @param[in] refSol     参考解对象。
 * @param[in] time       时间。
 * @param[in] gridScale  网格缩放因子；<=0 时自动计算单元体积。
 *
 * @note 误差定义：L2(omega)=sqrt(sum(|omega_h-omega_ref|^2)*gridScale)。
 */
PetscErrorCode Evaluation::compute_error_omega(DM dmSol, Vec sol, RefSol refSol,
                                               PetscReal time,
                                               PetscReal gridScale,
                                               PetscReal *l2ErrOut) {
  PetscFunctionBeginUser;

  // 与速度误差相同的系统布局判断方式。
  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol, &dof0, &dof1, &dof2, &dof3));
  const PetscBool is1FormSystem = (dof0 > 0) ? PETSC_TRUE : PETSC_FALSE;

  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));

  Vec localSol;
  PetscScalar ****arrSol;
  PetscCall(DMGetLocalVector(dmSol, &localSol));
  PetscCall(DMGlobalToLocal(dmSol, sol, INSERT_VALUES, localSol));
  PetscCall(DMStagVecGetArrayRead(dmSol, localSol, &arrSol));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icz_prev));

  // 涡量在两类系统中的位置与速度相反：
  //   1-form：omega 在面上
  //   2-form：omega 在棱上
  PetscInt slot_x, slot_y, slot_z;
  if (is1FormSystem) {
    PetscCall(DMStagGetLocationSlot(dmSol, LEFT, 0, &slot_x));
    PetscCall(DMStagGetLocationSlot(dmSol, DOWN, 0, &slot_y));
    PetscCall(DMStagGetLocationSlot(dmSol, BACK, 0, &slot_z));
  } else {
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_DOWN, 0, &slot_x));
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_LEFT, 0, &slot_y));
    PetscCall(DMStagGetLocationSlot(dmSol, DOWN_LEFT, 0, &slot_z));
  }

  PetscReal localSqErr = 0.0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        PetscScalar x, y, z;
        PetscScalar dox, doy, doz;

        // 在对应 DoF 几何位置评估参考涡量并求点值误差。
        if (is1FormSystem) {
          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_center];
          dox = arrSol[ez][ey][ex][slot_x] - refSol.omegaxRef(x, y, z, time);

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_center];
          doy = arrSol[ez][ey][ex][slot_y] - refSol.omegayRef(x, y, z, time);

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_prev];
          doz = arrSol[ez][ey][ex][slot_z] - refSol.omegazRef(x, y, z, time);
        } else {
          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_prev];
          dox = arrSol[ez][ey][ex][slot_x] - refSol.omegaxRef(x, y, z, time);

          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_prev];
          doy = arrSol[ez][ey][ex][slot_y] - refSol.omegayRef(x, y, z, time);

          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_center];
          doz = arrSol[ez][ey][ex][slot_z] - refSol.omegazRef(x, y, z, time);
        }

        localSqErr += PetscRealPart(
            dox * PetscConj(dox) + doy * PetscConj(doy) + doz * PetscConj(doz));
      }
    }
  }

  // MPI 全局归约，汇总所有进程上的误差平方和。
  PetscReal globalSqErr = 0.0;
  PetscCall(PMPI_Allreduce(&localSqErr, &globalSqErr, 1, MPIU_REAL, MPIU_SUM,
                           PetscObjectComm((PetscObject)dmSol)));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol, &cellScale));
  const PetscReal l2Err = PetscSqrtReal(globalSqErr * cellScale);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error of %s = %.12e\n",
                        is1FormSystem ? "omega2" : "omega1", (double)l2Err));
  if (l2ErrOut)
    *l2ErrOut = l2Err;

  PetscCall(DMStagVecRestoreArrayRead(dmSol, localSol, &arrSol));
  PetscCall(DMRestoreLocalVector(dmSol, &localSol));
  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief 计算压力 p（p0 或 p3）的 L2 误差。
 *
 * @param[in] dmSol      当前系统对应 DM（1-form 或 2-form）。
 * @param[in] sol        数值解向量。
 * @param[in] refSol     参考解对象。
 * @param[in] time       时间。
 * @param[in] gridScale  网格缩放因子；<=0 时自动计算单元体积。
 *
 * @note
 * - 1-form 系统：压力取顶点 p0（BACK_DOWN_LEFT）。
 * - 2-form 系统：压力取单元中心 p3（ELEMENT）。
 */
PetscErrorCode Evaluation::compute_error_p(DM dmSol, Vec sol, RefSol refSol,
                                           PetscReal time, PetscReal gridScale,
                                           PetscReal *l2ErrOut) {
  PetscFunctionBeginUser;

  // 压力所在位置取决于系统类型：
  //   1-form：p0 在顶点
  //   2-form：p3 在单元中心
  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol, &dof0, &dof1, &dof2, &dof3));
  const PetscBool hasP0 = (dof0 > 0) ? PETSC_TRUE : PETSC_FALSE;

  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));

  Vec localSol;
  PetscScalar ****arrSol;
  PetscCall(DMGetLocalVector(dmSol, &localSol));
  PetscCall(DMGlobalToLocal(dmSol, sol, INSERT_VALUES, localSol));
  PetscCall(DMStagVecGetArrayRead(dmSol, localSol, &arrSol));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icz_prev));

  PetscInt slot_p;
  if (hasP0) {
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_DOWN_LEFT, 0, &slot_p));
  } else {
    PetscCall(DMStagGetLocationSlot(dmSol, ELEMENT, 0, &slot_p));
  }

  // 先计算数值压力与参考压力在离散点集上的全局均值，再做去均值后的误差：
  // e_p = (p_h - mean(p_h)) - (p_ref - mean(p_ref))
  PetscScalar localPSum = 0.0;
  PetscScalar localPRefSum = 0.0;
  PetscInt localPCount = 0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        localPSum += arrSol[ez][ey][ex][slot_p];
        PetscScalar x, y, z;
        if (hasP0) {
          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_prev];
        } else {
          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_center];
        }
        localPRefSum += refSol.pRef(x, y, z, time);
        localPCount++;
      }
    }
  }

  PetscScalar globalPSum = 0.0, meanP = 0.0;
  PetscScalar globalPRefSum = 0.0, meanPRef = 0.0;
  PetscInt globalPCount = 0;
  PetscCall(PMPI_Allreduce(&localPSum, &globalPSum, 1, MPIU_SCALAR, MPI_SUM,
                           PetscObjectComm((PetscObject)dmSol)));
  PetscCall(PMPI_Allreduce(&localPRefSum, &globalPRefSum, 1, MPIU_SCALAR,
                           MPI_SUM, PetscObjectComm((PetscObject)dmSol)));
  PetscCall(PMPI_Allreduce(&localPCount, &globalPCount, 1, MPIU_INT, MPI_SUM,
                           PetscObjectComm((PetscObject)dmSol)));
  if (globalPCount > 0) {
    meanP = globalPSum / static_cast<PetscScalar>(globalPCount);
    meanPRef = globalPRefSum / static_cast<PetscScalar>(globalPCount);
  }

  PetscReal localSqErr = 0.0;
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        PetscScalar x, y, z;
        // 压力参考值采样点必须与对应交错存储位置一致。
        if (hasP0) {
          x = cArrX[ex][icx_prev];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_prev];
        } else {
          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_center];
        }
        const PetscScalar pNumZeroMean = arrSol[ez][ey][ex][slot_p] - meanP;
        const PetscScalar pRefZeroMean = refSol.pRef(x, y, z, time) - meanPRef;
        const PetscScalar dp = pNumZeroMean - pRefZeroMean;
        localSqErr += PetscRealPart(dp * PetscConj(dp));
      }
    }
  }

  // MPI 归约后按网格尺度缩放并开方，得到最终 L2 误差。
  PetscReal globalSqErr = 0.0;
  PetscCall(PMPI_Allreduce(&localSqErr, &globalSqErr, 1, MPIU_REAL, MPIU_SUM,
                           PetscObjectComm((PetscObject)dmSol)));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol, &cellScale));
  const PetscReal l2Err = PetscSqrtReal(globalSqErr * cellScale);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error of %s = %.12e\n",
                        hasP0 ? "p0" : "p3", (double)l2Err));
  if (l2ErrOut)
    *l2ErrOut = l2Err;

  PetscCall(DMStagVecRestoreArrayRead(dmSol, localSol, &arrSol));
  PetscCall(DMRestoreLocalVector(dmSol, &localSol));
  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Evaluation::compute_error_grad_p(DM dmSol, Vec sol,
                                                RefSol refSol, PetscReal time,
                                                PetscReal gridScale,
                                                PetscReal *l2ErrOut) {
  PetscFunctionBeginUser;

  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dmSol, &dof0, &dof1, &dof2, &dof3));
  const PetscBool hasP0 = (dof0 > 0) ? PETSC_TRUE : PETSC_FALSE;

  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));

  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icx_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icy_prev));
  PetscCall(
      DMStagGetProductCoordinateLocationSlot(dmSol, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol, LEFT, &icz_prev));

  Vec localSol;
  PetscScalar ****arrSol;
  PetscCall(DMGetLocalVector(dmSol, &localSol));
  PetscCall(DMGlobalToLocal(dmSol, sol, INSERT_VALUES, localSol));
  PetscCall(DMStagVecGetArrayRead(dmSol, localSol, &arrSol));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  const PetscReal hx =
      PetscRealPart(cArrX[startx + 1][icx_prev] - cArrX[startx][icx_prev]);
  const PetscReal hy =
      PetscRealPart(cArrY[starty + 1][icy_prev] - cArrY[starty][icy_prev]);
  const PetscReal hz =
      PetscRealPart(cArrZ[startz + 1][icz_prev] - cArrZ[startz][icz_prev]);

  PetscReal localSqErr = 0.0;

  if (hasP0) {
    PetscInt slot_p;
    PetscCall(DMStagGetLocationSlot(dmSol, BACK_DOWN_LEFT, 0, &slot_p));

    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {
          const PetscScalar pHere = arrSol[ez][ey][ex][slot_p];
          const PetscScalar xH = cArrX[ex][icx_prev];
          const PetscScalar yH = cArrY[ey][icy_prev];
          const PetscScalar zH = cArrZ[ez][icz_prev];
          const PetscScalar pRefHere = refSol.pRef(xH, yH, zH, time);

          // dp/dx on BACK_DOWN edge: (p[ex+1]-p[ex])/hx
          {
            const PetscScalar xR = cArrX[ex + 1][icx_prev];
            const PetscScalar dpNum =
                (arrSol[ez][ey][ex + 1][slot_p] - pHere) / hx;
            const PetscScalar dpRef =
                (refSol.pRef(xR, yH, zH, time) - pRefHere) / hx;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
          // dp/dy on BACK_LEFT edge: (p[ey+1]-p[ey])/hy
          {
            const PetscScalar yU = cArrY[ey + 1][icy_prev];
            const PetscScalar dpNum =
                (arrSol[ez][ey + 1][ex][slot_p] - pHere) / hy;
            const PetscScalar dpRef =
                (refSol.pRef(xH, yU, zH, time) - pRefHere) / hy;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
          // dp/dz on DOWN_LEFT edge: (p[ez+1]-p[ez])/hz
          {
            const PetscScalar zF = cArrZ[ez + 1][icz_prev];
            const PetscScalar dpNum =
                (arrSol[ez + 1][ey][ex][slot_p] - pHere) / hz;
            const PetscScalar dpRef =
                (refSol.pRef(xH, yH, zF, time) - pRefHere) / hz;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
        }
      }
    }
  } else {
    PetscInt slot_p;
    PetscCall(DMStagGetLocationSlot(dmSol, ELEMENT, 0, &slot_p));

    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {
          const PetscScalar pHere = arrSol[ez][ey][ex][slot_p];
          const PetscScalar xH = cArrX[ex][icx_center];
          const PetscScalar yH = cArrY[ey][icy_center];
          const PetscScalar zH = cArrZ[ez][icz_center];
          const PetscScalar pRefHere = refSol.pRef(xH, yH, zH, time);

          // dp/dx on LEFT face: (p[ex]-p[ex-1])/hx
          {
            const PetscScalar xL = cArrX[ex - 1][icx_center];
            const PetscScalar dpNum =
                (pHere - arrSol[ez][ey][ex - 1][slot_p]) / hx;
            const PetscScalar dpRef =
                (pRefHere - refSol.pRef(xL, yH, zH, time)) / hx;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
          // dp/dy on DOWN face: (p[ey]-p[ey-1])/hy
          {
            const PetscScalar yD = cArrY[ey - 1][icy_center];
            const PetscScalar dpNum =
                (pHere - arrSol[ez][ey - 1][ex][slot_p]) / hy;
            const PetscScalar dpRef =
                (pRefHere - refSol.pRef(xH, yD, zH, time)) / hy;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
          // dp/dz on BACK face: (p[ez]-p[ez-1])/hz
          {
            const PetscScalar zB = cArrZ[ez - 1][icz_center];
            const PetscScalar dpNum =
                (pHere - arrSol[ez - 1][ey][ex][slot_p]) / hz;
            const PetscScalar dpRef =
                (pRefHere - refSol.pRef(xH, yH, zB, time)) / hz;
            const PetscScalar d = dpNum - dpRef;
            localSqErr += PetscRealPart(d * PetscConj(d));
          }
        }
      }
    }
  }

  PetscReal globalSqErr = 0.0;
  PetscCall(PMPI_Allreduce(&localSqErr, &globalSqErr, 1, MPIU_REAL, MPIU_SUM,
                           PetscObjectComm((PetscObject)dmSol)));

  PetscReal cellScale = gridScale;
  if (cellScale <= 0.0)
    PetscCall(GetCellVolumeFromDM(dmSol, &cellScale));
  const PetscReal l2Err = PetscSqrtReal(globalSqErr * cellScale);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 error of grad(%s) = %.12e\n",
                        hasP0 ? "p0" : "p3", (double)l2Err));
  if (l2ErrOut)
    *l2ErrOut = l2Err;

  PetscCall(DMStagVecRestoreArrayRead(dmSol, localSol, &arrSol));
  PetscCall(DMRestoreLocalVector(dmSol, &localSol));
  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dmSol, &cArrX, &cArrY, &cArrZ));
  PetscFunctionReturn(PETSC_SUCCESS);
}
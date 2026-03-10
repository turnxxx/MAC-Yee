// #pragma once
#include "../include/DUAL_MAC.h"
#include "../include/ref_sol.h"
#include "petscdmstag.h"
#include "petscsystypes.h"
#include <petsc.h>
// 1形式系统矩阵组装
// 根据论文第5章方程(5.4)-(5.6)，组装半整数时间步的系统矩阵
// 系统包含三个方程：
// 1. 动量方程(5.4)：时间导数 + 对流项 + 旋度项 + 压力梯度 = 右端项
// 2. 耦合方程(5.5)：ω₂ - ∇×u₁ = 0
// 3. 散度方程(5.6)：∇·u₁ = 0
PetscErrorCode DUAL_MAC::assemble_1form_system_matrix(
    DM dmSol_1, DM dmSol_2, Mat A, Vec rhs, Vec u1_prev, Vec omega1_known,
    Vec omega2_prev, ExternalForce externalForce, PetscReal time,
    PetscReal dt) {
  PetscFunctionBeginUser;
  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 系统矩阵组装开始\n");

  // 计算雷诺数（假设 nu 是动力粘度，Re = 1/nu）
  // 如果 nu 已经是雷诺数的倒数，则 Re = 1/nu
  PetscReal Re = 1.0 / this->nu;

  // ===== 1. 初始化矩阵和右端项 =====
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(rhs));

  // ===== 2. 组装时间导数矩阵：1/dt * I（对 u₁）=====
  // 对应方程(5.4)中的 (u₁^{k+1/2} - u₁^{k-1/2})/dt
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 时间导数矩阵组装开始\n");
  PetscCall(assemble_u1_dt_matrix(dmSol_1, A, dt));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 时间导数矩阵组装完成\n");

  // ===== 3. 组装对流项矩阵：0.5 * ω₁^{h,k} ×（对 u₁）=====
  // 对应方程(5.4)中的 ω₁^{h,k} × (u₁^{k+1/2} + u₁^{k-1/2})/2
  // 注意：assemble_u1_conv_matrix 内部已经包含了 0.5 的系数
  /*   DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 对流项矩阵组装开始\n");
    PetscCall(assemble_u1_conv_matrix(dmSol_1, A, dmSol_2, omega1_known));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 对流项矩阵组装完成\n"); */

  // ===== 4. 组装旋度项矩阵：0.5/Re * ∇×（对 ω₂）=====
  // 对应方程(5.4)中的 (1/Re) ∇_h × (ω₂^{k+1/2} + ω₂^{k-1/2})/2
  // 由于 assemble_omega2_curl_matrix 使用 coeff = 1.0/Re
  // 我们需要传入 2*Re 来得到 coeff = 1.0/(2*Re) = 0.5/Re 的效果
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 旋度项矩阵组装开始\n");
  PetscCall(assemble_omega2_curl_matrix(2.0 * Re, dmSol_1, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 旋度项矩阵组装完成\n");

  // ===== 5. 组装压力梯度矩阵：∇（对 P₀）=====
  // 对应方程(5.4)中的 ∇_h P₀^{h,k}
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 压力梯度矩阵组装开始\n");
  PetscCall(assemble_p0_gradient_matrix(dmSol_1, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 压力梯度矩阵组装完成\n");

  // ===== 6. 组装耦合矩阵：-∇×（对 u₁）和 +I（对 ω₂）=====
  // 对应方程(5.5)中的 ω₂^{h,k+1/2} - ∇_h × u₁^{h,k+1/2} = 0
  // 注意：assemble_u1_omega2_coupling_matrix 组装的是 ω₂ - ∇×u₁ = 0
  // 所以它已经包含了 +I 项（对 ω₂）
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 耦合矩阵组装开始\n");
  PetscCall(assemble_u1_omega2_coupling_matrix(dmSol_1, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 耦合矩阵组装完成\n");

  // ===== 7. 组装散度矩阵：∇·（对 u₁）=====
  // 对应方程(5.6)中的 ∇_h · u₁^{h,k+1/2} = 0
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 散度矩阵组装开始\n");
  PetscCall(assemble_u1_divergence_matrix(dmSol_1, A));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 散度矩阵组装完成\n");

  // ===== 8. 组装右端项（不包括外力）=====
  // 包含时间导数项、对流项和旋度项：
  // - +u₁^{k-1/2}/dt（时间导数项）
  // - -0.5 * ω₁^{h,k} × u₁^{k-1/2}（对流项，移项后变负）
  // - -0.5/Re * ∇_h × ω₂^{h,k-1/2}（旋度项，移项后变负）
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 右端项组装开始\n");
  PetscCall(assemble_rhs1_vector(dmSol_1, rhs, u1_prev, dmSol_2, omega1_known,
                                 omega2_prev, dt));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 右端项组装完成\n");

  // ===== 9. 组装外力项 =====
  // 对应方程(5.4)中的 R₁ f^k
  // 外力项直接添加到右端项向量中
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 外力项组装开始\n");
  PetscCall(assemble_force1_vector(dmSol_1, rhs, externalForce, time));
  DUAL_MAC_DEBUG_LOG("[DEBUG] [1-form] 外力项组装完成\n");

  // ===== 10. 矩阵和向量最终组装 =====
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(rhs));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(rhs));

  // ===== 11. 1-form 压力钉住（仅当启用 pinPressure）=====
  // 将 (0,0,0) 处的 p0 方程固定为 p0 =
  // 0，缓解周期边界下压力零空间导致的慢收敛。
  if (this->pinPressure) {
    PetscInt startx_pin, starty_pin, startz_pin, nx_pin, ny_pin, nz_pin;
    PetscCall(DMStagGetCorners(dmSol_1, &startx_pin, &starty_pin, &startz_pin,
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
      PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &prow, &pval,
                                          INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyEnd(rhs));
  }

  DUAL_MAC_DEBUG_LOG("[DEBUG] 1-form 系统矩阵组装完成\n");
  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装右端项(不包括外力)
// 根据Crank-Nicolson时间步进，右端项包含：
// 1. 时间导数项：+u₁^{k-1/2} / dt（上一步的速度）
// 2. 对流项：-0.5 * ω₁^{h,k} × u₁^{h,k-1/2}（移项后变负）
// 3. 旋度项：-0.5/Re * ∇_h × ω₂^{h,k-1/2}（移项后变负）
PetscErrorCode DUAL_MAC::assemble_rhs1_vector(DM dmSol_1, Vec rhs, Vec u1_prev,
                                              DM dmSol_2, Vec omega1_known,
                                              Vec omega2_prev, PetscReal dt) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 计算雷诺数和旋度项系数
  PetscReal Re = 1.0 / this->nu;
  PetscScalar curl_coeff = -0.5 / Re; // -0.5/Re 系数（来自方程5.4）
  PetscPrintf(PETSC_COMM_WORLD, "assemble_rhs1_vector: curl_coeff: %g\n",
              curl_coeff);
  // 获取上一步速度场的本地数组
  Vec localU1Prev;
  PetscScalar ****arrU1Prev;
  PetscCall(DMGetLocalVector(dmSol_1, &localU1Prev));
  PetscCall(DMGlobalToLocal(dmSol_1, u1_prev, INSERT_VALUES, localU1Prev));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localU1Prev, &arrU1Prev));

  // 获取已知涡度场的本地数组（使用 dmSol_2，因为 ω₁ 存储在 dmSol_2 中）
  Vec localOmega1;
  PetscScalar ****arrOmega1;
  PetscCall(DMGetLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMGlobalToLocal(dmSol_2, omega1_known, INSERT_VALUES, localOmega1));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localOmega1, &arrOmega1));

  // 获取上一步2形式涡度场的本地数组（使用 dmSol_1，因为 ω₂ 存储在 dmSol_1 中）
  Vec localOmega2Prev;
  PetscScalar ****arrOmega2Prev;
  PetscCall(DMGetLocalVector(dmSol_1, &localOmega2Prev));
  PetscCall(
      DMGlobalToLocal(dmSol_1, omega2_prev, INSERT_VALUES, localOmega2Prev));
  PetscCall(DMStagVecGetArrayRead(dmSol_1, localOmega2Prev, &arrOmega2Prev));

  // 获取 slot 索引
  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x)); // x方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y)); // y方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z)); // z方向棱

  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x)); // x方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y)); // y方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z)); // z方向棱

  // 获取 ω₂ 的 slot 索引（ω₂ 存储在 dmSol_1 的面上）
  PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
  PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0,
                                  &slot_omega2_x)); // x方向面（LEFT/RIGHT）
  PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0,
                                  &slot_omega2_y)); // y方向面（DOWN/UP）
  PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0,
                                  &slot_omega2_z)); // z方向面（BACK/FRONT）

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的右端项 =====
        // 1. 时间导数项：u₁ˣ^{k-1/2} / dt
        // 2. 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 x 分量
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN; // x方向棱
        row.c = 0;

        PetscScalar rhs_val = 0.0;

        // 时间导数项：u₁ˣ^{k-1/2} / dt
        rhs_val += arrU1Prev[ez][ey][ex][slot_u1_x] / dt;

        // 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 x 分量
        // (ω₁ × u₁)_x = ω₁ʸ u₁ᶻ - ω₁ᶻ u₁ʸ
        // 需要插值 ω₁ 和 u₁ 到 x 方向棱的位置

        // ω₁ʸ：在 y 方向棱上，需要插值到 x 方向棱的位置
        // x 方向棱（BACK_DOWN）位于 y=prev, z=prev
        // 使用四个位置的双线性插值（在 x 和 y 两个方向上）
        PetscScalar omega1_y =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_y] +
                    arrOmega1[ez][ey][ex + 1][slot_omega1_y] +
                    arrOmega1[ez][ey - 1][ex][slot_omega1_y] +
                    arrOmega1[ez][ey - 1][ex + 1][slot_omega1_y]);

        // ω₁ᶻ：在 z 方向棱上，需要插值到 x 方向棱的位置
        // x 方向棱（BACK_DOWN）位于 y=prev, z=prev
        // 使用四个位置的双线性插值（在 x 和 z 两个方向上）
        PetscScalar omega1_z =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_z] +
                    arrOmega1[ez][ey][ex + 1][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex + 1][slot_omega1_z]);

        // u₁ᶻ^{k-1/2}：需要插值到 x 方向棱的位置，使用四个 z 方向棱
        PetscScalar u1_z_prev =
            0.25 * (arrU1Prev[ez][ey][ex][slot_u1_z] +
                    arrU1Prev[ez][ey][ex + 1][slot_u1_z] +
                    arrU1Prev[ez - 1][ey][ex][slot_u1_z] +
                    arrU1Prev[ez - 1][ey][ex + 1][slot_u1_z]);

        // u₁ʸ^{k-1/2}：需要插值到 x 方向棱的位置，使用四个 y 方向棱
        PetscScalar u1_y_prev =
            0.25 * (arrU1Prev[ez][ey][ex][slot_u1_y] +
                    arrU1Prev[ez][ey][ex + 1][slot_u1_y] +
                    arrU1Prev[ez][ey - 1][ex][slot_u1_y] +
                    arrU1Prev[ez][ey - 1][ex + 1][slot_u1_y]);

        // 对流项：-0.5 * (ω₁ʸ u₁ᶻ - ω₁ᶻ u₁ʸ)（0.5 来自时间平均，负号来自移项）
        // 右端对流项
        // rhs_val -= 0.5 * (omega1_y * u1_z_prev - omega1_z * u1_y_prev);

        // 旋度项：-0.5/Re * (∇ × ω₂^{k-1/2})_x
        // (∇ × ω₂)_x = ∂ω₂ᶻ/∂y - ∂ω₂ʸ/∂z
        // 第一项：∂ω₂ᶻ/∂y ≈ (ω₂ᶻ|_{ey} - ω₂ᶻ|_{ey-1}) / hy
        // ω₂ᶻ 在 xy 平面上（法向 z），即 BACK 和 FRONT 面
        PetscScalar curl_x = 0.0;
        curl_x += (arrOmega2Prev[ez][ey][ex][slot_omega2_z] -
                   arrOmega2Prev[ez][ey - 1][ex][slot_omega2_z]) /
                  hy;
        // 第二项：-∂ω₂ʸ/∂z ≈ -(ω₂ʸ|_{ez} - ω₂ʸ|_{ez-1}) / hz
        // ω₂ʸ 在 xz 平面上（法向 y），即 DOWN 和 UP 面
        curl_x -= (arrOmega2Prev[ez][ey][ex][slot_omega2_y] -
                   arrOmega2Prev[ez - 1][ey][ex][slot_omega2_y]) /
                  hz;
        rhs_val += curl_coeff * curl_x;

        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的右端项 =====
        // 1. 时间导数项：u₁ʸ^{k-1/2} / dt
        // 2. 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 y 分量
        row.loc = BACK_LEFT; // y方向棱
        rhs_val = 0.0;

        // 时间导数项：u₁ʸ^{k-1/2} / dt
        rhs_val += arrU1Prev[ez][ey][ex][slot_u1_y] / dt;

        // 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 y 分量
        // (ω₁ × u₁)_y = ω₁ᶻ u₁ˣ - ω₁ˣ u₁ᶻ

        // ω₁ᶻ：在 z 方向棱上，需要插值到 y 方向棱的位置
        // y 方向棱（BACK_LEFT）位于 x=prev, z=prev
        // 使用四个位置的双线性插值（在 y 和 z 两个方向上）
        PetscScalar omega1_z_y =
            0.25 * (arrOmega1[ez][ey][ex][slot_omega1_z] +
                    arrOmega1[ez][ey + 1][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey][ex][slot_omega1_z] +
                    arrOmega1[ez - 1][ey + 1][ex][slot_omega1_z]);

        // ω₁ˣ：在 x 方向棱上，需要插值到 y 方向棱的位置
        // y 方向棱（BACK_LEFT）位于 x=prev, z=prev
        // 使用四个位置的双线性插值（在 x 和 y 两个方向上）
        PetscScalar omega1_x =
            0.25 *
            (arrOmega1[ez][ey][ex][slot_omega1_x] + // BACK_DOWN (当前单元)
             arrOmega1[ez][ey][ex - 1][slot_omega1_x] + // 左邻居的 BACK_DOWN
             arrOmega1[ez][ey + 1][ex]
                      [slot_omega1_x] + // BACK_UP (上邻居的BACK_DOWN)
             arrOmega1[ez][ey + 1][ex - 1]
                      [slot_omega1_x]); // 左上邻居的 BACK_DOWN

        // u₁ˣ^{k-1/2}：需要插值到 y 方向棱的位置，使用四个 x 方向棱（在 x 和 y
        // 两个方向上）
        PetscScalar u1_x_prev =
            0.25 *
            (arrU1Prev[ez][ey][ex][slot_u1_x] +     // BACK_DOWN (当前单元)
             arrU1Prev[ez][ey][ex - 1][slot_u1_x] + // 左邻居的 BACK_DOWN
             arrU1Prev[ez][ey + 1][ex]
                      [slot_u1_x] + // BACK_UP (上邻居的BACK_DOWN)
             arrU1Prev[ez][ey + 1][ex - 1][slot_u1_x]); // 左上邻居的 BACK_DOWN

        // u₁ᶻ^{k-1/2}：需要插值到 y 方向棱的位置，使用四个 z 方向棱
        PetscScalar u1_z_prev_y =
            0.25 * (arrU1Prev[ez][ey][ex][slot_u1_z] +
                    arrU1Prev[ez][ey + 1][ex][slot_u1_z] +
                    arrU1Prev[ez - 1][ey][ex][slot_u1_z] +
                    arrU1Prev[ez - 1][ey + 1][ex][slot_u1_z]);

        // 对流项：-0.5 * (ω₁ᶻ u₁ˣ - ω₁ˣ u₁ᶻ)（0.5 来自时间平均，负号来自移项）
        // 右端对流项
        // rhs_val -= 0.5 * (omega1_z_y * u1_x_prev - omega1_x * u1_z_prev_y);

        // 旋度项：-0.5/Re * (∇ × ω₂^{k-1/2})_y
        // (∇ × ω₂)_y = ∂ω₂ˣ/∂z - ∂ω₂ᶻ/∂x
        // 第一项：∂ω₂ˣ/∂z ≈ (ω₂ˣ|_{ez} - ω₂ˣ|_{ez-1}) / hz
        // ω₂ˣ 在 yz 平面上（法向 x），即 LEFT 和 RIGHT 面
        PetscScalar curl_y = 0.0;
        curl_y += (arrOmega2Prev[ez][ey][ex][slot_omega2_x] -
                   arrOmega2Prev[ez - 1][ey][ex][slot_omega2_x]) /
                  hz;
        // 第二项：-∂ω₂ᶻ/∂x ≈ -(ω₂ᶻ|_{ex} - ω₂ᶻ|_{ex-1}) / hx
        // ω₂ᶻ 在 xy 平面上（法向 z），即 BACK 和 FRONT 面
        curl_y -= (arrOmega2Prev[ez][ey][ex][slot_omega2_z] -
                   arrOmega2Prev[ez][ey][ex - 1][slot_omega2_z]) /
                  hx;
        rhs_val += curl_coeff * curl_y;

        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的右端项 =====
        // 1. 时间导数项：u₁ᶻ^{k-1/2} / dt
        // 2. 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 z 分量
        row.loc = DOWN_LEFT; // z方向棱
        rhs_val = 0.0;

        // 时间导数项：u₁ᶻ^{k-1/2} / dt
        rhs_val += arrU1Prev[ez][ey][ex][slot_u1_z] / dt;

        // 对流项：ω₁^{h,k} × u₁^{h,k-1/2} 的 z 分量
        // (ω₁ × u₁)_z = ω₁ˣ u₁ʸ - ω₁ʸ u₁ˣ

        // ω₁ˣ：在 x 方向棱上，需要插值到 z 方向棱的位置
        // z 方向棱（DOWN_LEFT）位于 x=prev, y=prev
        // 使用四个位置的双线性插值（在 y 和 z 两个方向上）
        // ω₁ˣ：在 x 方向棱上，需要插值到 z 方向棱的位置
        // z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，z 坐标是单元中心 ez
        // 使用四个位置的双线性插值（在 x 和 z 两个方向上）
        PetscScalar omega1_x_z =
            0.25 * (arrOmega1[ez][ey][ex]
                             [slot_omega1_x] + // BACK_DOWN (当前单元, 本地)
                    arrOmega1[ez + 1][ey][ex]
                             [slot_omega1_x] + // FRONT_DOWN (前邻居, 前方)
                    arrOmega1[ez][ey][ex - 1]
                             [slot_omega1_x] + // 左邻居的 BACK_DOWN (左边)
                    arrOmega1[ez + 1][ey][ex - 1]
                             [slot_omega1_x]); // 左前邻居的 BACK_DOWN (左前)

        // ω₁ʸ：在 y 方向棱上，需要插值到 z 方向棱的位置
        // z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，z 坐标是单元中心 ez
        // 使用四个位置的双线性插值（在 y 和 z 两个方向上）
        PetscScalar omega1_y_z =
            0.25 *
            (arrOmega1[ez][ey][ex]
                      [slot_omega1_y] + // BACK_LEFT (当前单元, 本地)
             arrOmega1[ez + 1][ey][ex]
                      [slot_omega1_y] + // FRONT_LEFT (前邻居, 前方)
             arrOmega1[ez][ey - 1][ex]
                      [slot_omega1_y] + // 下邻居的 BACK_LEFT (下邻居)
             arrOmega1[ez + 1][ey - 1][ex]
                      [slot_omega1_y]); // 前下邻居的 BACK_LEFT (前下邻居)

        // u₁ʸ^{k-1/2}：需要插值到 z 方向棱的位置，使用四个 y 方向棱（在 y 和 z
        // 两个方向上）
        PetscScalar u1_y_prev_z =
            0.25 *
            (arrU1Prev[ez][ey][ex][slot_u1_y] + // BACK_LEFT (当前单元, 本地)
             arrU1Prev[ez + 1][ey][ex][slot_u1_y] + // FRONT_LEFT (前邻居, 前方)
             arrU1Prev[ez][ey - 1][ex]
                      [slot_u1_y] + // 下邻居的 BACK_LEFT (下邻居)
             arrU1Prev[ez + 1][ey - 1][ex]
                      [slot_u1_y]); // 前下邻居的 BACK_LEFT (前下邻居)

        // u₁ˣ^{k-1/2}：需要插值到 z 方向棱的位置，使用四个 x 方向棱（在 x 和 z
        // 两个方向上）
        PetscScalar u1_x_prev_z =
            0.25 *
            (arrU1Prev[ez][ey][ex][slot_u1_x] + // BACK_DOWN (当前单元, 本地)
             arrU1Prev[ez + 1][ey][ex][slot_u1_x] + // FRONT_DOWN (前邻居, 前方)
             arrU1Prev[ez][ey][ex - 1][slot_u1_x] + // 左邻居的 BACK_DOWN (左边)
             arrU1Prev[ez + 1][ey][ex - 1]
                      [slot_u1_x]); // 左前邻居的 BACK_DOWN (左前)

        // 对流项：-0.5 * (ω₁ˣ u₁ʸ - ω₁ʸ u₁ˣ)（0.5 来自时间平均，负号来自移项）
        // 右端对流项
        // rhs_val -= 0.5 * (omega1_x_z * u1_y_prev_z - omega1_y_z *
        // u1_x_prev_z);

        // 旋度项：-0.5/Re * (∇ × ω₂^{k-1/2})_z
        // (∇ × ω₂)_z = ∂ω₂ʸ/∂x - ∂ω₂ˣ/∂y
        // 第一项：∂ω₂ʸ/∂x ≈ (ω₂ʸ|_{ex} - ω₂ʸ|_{ex-1}) / hx
        // ω₂ʸ 在 xz 平面上（法向 y），即 DOWN 和 UP 面
        PetscScalar curl_z = 0.0;
        curl_z += (arrOmega2Prev[ez][ey][ex][slot_omega2_y] -
                   arrOmega2Prev[ez][ey][ex - 1][slot_omega2_y]) /
                  hx;
        // 第二项：-∂ω₂ˣ/∂y ≈ -(ω₂ˣ|_{ey} - ω₂ˣ|_{ey-1}) / hy
        // ω₂ˣ 在 yz 平面上（法向 x），即 LEFT 和 RIGHT 面
        curl_z -= (arrOmega2Prev[ez][ey][ex][slot_omega2_x] -
                   arrOmega2Prev[ez][ey - 1][ex][slot_omega2_x]) /
                  hy;
        rhs_val += curl_coeff * curl_z;

        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &rhs_val,
                                            ADD_VALUES));
      }
    }
  }

  // 释放数组
  PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localU1Prev, &arrU1Prev));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localU1Prev));
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localOmega1, &arrOmega1));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1));
  PetscCall(
      DMStagVecRestoreArrayRead(dmSol_1, localOmega2Prev, &arrOmega2Prev));
  PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2Prev));

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装2形式robust外力项
PetscErrorCode DUAL_MAC::assemble_robust_force1_vector() {
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装2形式外力项
// 在每一个棱上取对应点和方向的外力值
PetscErrorCode DUAL_MAC::assemble_force1_vector(DM dmSol_1, Vec rhs,
                                                ExternalForce externalForce,
                                                PetscReal time) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 获取 product 坐标数组
  PetscScalar **cArrX, **cArrY, **cArrZ;
  PetscCall(
      DMStagGetProductCoordinateArraysRead(dmSol_1, &cArrX, &cArrY, &cArrZ));

  // 获取坐标的 slot 索引
  // 对于每个方向（x, y, z），需要获取不同位置的坐标slot
  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(DMStagGetProductCoordinateLocationSlot(
      dmSol_1, ELEMENT, &icx_center)); // x坐标在单元中心
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT,
                                                   &icx_prev)); // x坐标在左面
  PetscCall(DMStagGetProductCoordinateLocationSlot(
      dmSol_1, ELEMENT, &icy_center)); // y坐标在单元中心
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT,
                                                   &icy_prev)); // y坐标在下面
  PetscCall(DMStagGetProductCoordinateLocationSlot(
      dmSol_1, ELEMENT, &icz_center)); // z坐标在单元中心
  PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_1, LEFT,
                                                   &icz_prev)); // z坐标在后面

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的外力项 =====
        // x 方向棱沿 x 方向延伸，位于后面(z=prev)和下面(y=prev)的交线
        // x 坐标在单元中心，y 坐标在单元底部，z 坐标在单元后面
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN; // x方向棱
        row.c = 0;

        // 获取坐标：x 在单元中心，y 和 z 在 prev 位置
        PetscScalar x = cArrX[ex][icx_center];
        PetscScalar y = cArrY[ey][icy_prev];
        PetscScalar z = cArrZ[ez][icz_prev];

        // 调用 x 方向的外力函数
        PetscScalar force_val = externalForce.fx(x, y, z, time);
        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &force_val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的外力项 =====
        // y 方向棱沿 y 方向延伸，位于后面(z=prev)和左面(x=prev)的交线
        // x 坐标在单元左面，y 坐标在单元中心，z 坐标在单元后面
        row.loc = BACK_LEFT; // y方向棱

        // 获取坐标：y 在单元中心，x 和 z 在 prev 位置
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_center];
        z = cArrZ[ez][icz_prev];

        // 调用 y 方向的外力函数
        force_val = externalForce.fy(x, y, z, time);
        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &force_val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的外力项 =====
        // z 方向棱沿 z 方向延伸，位于下面(y=prev)和左面(x=prev)的交线
        // x 坐标在单元左面，y 坐标在单元底部，z 坐标在单元中心
        row.loc = DOWN_LEFT; // z方向棱

        // 获取坐标：z 在单元中心，x 和 y 在 prev 位置
        x = cArrX[ex][icx_prev];
        y = cArrY[ey][icy_prev];
        z = cArrZ[ez][icz_center];

        // 调用 z 方向的外力函数
        force_val = externalForce.fz(x, y, z, time);
        PetscCall(DMStagVecSetValuesStencil(dmSol_1, rhs, 1, &row, &force_val,
                                            ADD_VALUES));
      }
    }
  }

  // 释放坐标数组
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dmSol_1, &cArrX, &cArrY,
                                                     &cArrZ));

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装u1时间矩阵
PetscErrorCode DUAL_MAC::assemble_u1_dt_matrix(DM dmSol_1, Mat A,
                                               PetscReal dt) {
  PetscFunctionBeginUser;

  // ===== 1. 获取本进程的局部单元范围 =====
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  // 时间导数系数：1/dt
  // 如果用弱形式，还需乘以对偶体积 hx*hy*hz
  PetscScalar coeff = 1.0 / dt;

  // ===== 2. 遍历本进程的所有单元 =====
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // 每个单元拥有 3 条棱的 DOF
        // 它们是该单元"左下后"角的 3 条棱
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.c = 0; // dof1=1，只有一个分量

        // ---- x 方向棱 (BACK_DOWN) ----
        // 沿 x 方向延伸，位于后面(z=prev)和下面(y=prev)的交线
        row.loc = BACK_DOWN;
        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, // 1 行
                                            1,
                                            &row, // 1 列（行列相同 → 对角元素）
                                            &coeff, ADD_VALUES));

        // ---- y 方向棱 (BACK_LEFT) ----
        // 沿 y 方向延伸，位于后面(z=prev)和左面(x=prev)的交线
        row.loc = BACK_LEFT;
        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 1, &row,
                                            &coeff, ADD_VALUES));

        // ---- z 方向棱 (DOWN_LEFT) ----
        // 沿 z 方向延伸，位于下面(y=prev)和左面(x=prev)的交线
        row.loc = DOWN_LEFT;
        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 1, &row,
                                            &coeff, ADD_VALUES));
      }
    }
  }

  // ===== 3. 注意：这里不调用 MatAssemblyBegin/End =====
  // 因为整个系统矩阵可能还有其他项（对流、扩散、压力梯度等）
  // 统一在所有子矩阵组装完毕后再 Assembly

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装 u₁ 对流项矩阵：ω₁ × u₁
// 这是1-形式（棱）到1-形式（棱）的耦合项
// 对应方程5.4中的 ω₁^{h,k} × (u₁^{h,k+1/2} + u₁^{h,k-1/2})/2
// 注意：这里使用已知的 ω₁^{h,k}（从整数步系统得到）来线性化对流项
// 重要：omega1_known 存储在 dmSol_2 中，必须使用 dmSol_2 来读取
PetscErrorCode DUAL_MAC::assemble_u1_conv_matrix(DM dmSol_1, Mat A, DM dmSol_2,
                                                 Vec omega1_known) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  // 使用 dmSol_1 获取单元范围（两个 DM 的网格分区相同）
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 获取已知涡度场的本地数组（使用 dmSol_2，因为 ω₁ 存储在 dmSol_2 中）
  Vec localOmega1;
  PetscScalar ****arrOmega1;
  PetscCall(DMGetLocalVector(dmSol_2, &localOmega1));
  PetscCall(DMGlobalToLocal(dmSol_2, omega1_known, INSERT_VALUES, localOmega1));
  PetscCall(DMStagVecGetArrayRead(dmSol_2, localOmega1, &arrOmega1));

  // 获取 slot 索引
  // ω₁ 在 dmSol_2 的 dof1 上（slot=0，因为 dof1=1）
  PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x)); // x方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y)); // y方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z)); // z方向棱

  // u₁ 在 dmSol_1 的 dof1 上（slot=0，因为 dof1=1）
  PetscInt slot_u1_x, slot_u1_y, slot_u1_z;
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, BACK_DOWN, 0, &slot_u1_x)); // x方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, BACK_LEFT, 0, &slot_u1_y)); // y方向棱
  PetscCall(
      DMStagGetLocationSlot(dmSol_1, DOWN_LEFT, 0, &slot_u1_z)); // z方向棱

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的对流项 =====
        // (ω₁ × u₁)_x = ω₁ʸ u₁ᶻ - ω₁ᶻ u₁ʸ
        // 行：x方向棱（u₁ˣ 的方程）
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN; // x方向棱
        row.c = 0;

        // 读取已知的涡度值（在棱上，需要插值到当前棱的位置）
        // 注意：在MAC网格上，不同方向的棱不在同一位置，需要插值
        // 使用相邻棱的平均值提高插值精度
        PetscScalar omega1_y, omega1_z;

        // ω₁ʸ：在 y 方向棱上，需要插值到 x 方向棱的位置
        // x 方向棱（BACK_DOWN）位于 y=prev, z=prev
        // 使用四个位置的双线性插值（在 x 和 y 两个方向上）
        // 1. 当前单元的 BACK_LEFT (x=prev, y=prev, z=prev)
        // 2. 当前单元的 BACK_RIGHT = 右邻居的 BACK_LEFT (x=next, y=prev,
        // z=prev)
        // 3. 下方单元的 BACK_LEFT (x=prev, y=prev-1, z=prev)
        // 4. 下方单元的 BACK_RIGHT = 右下邻居的 BACK_LEFT (x=next, y=prev-1,
        // z=prev)
        omega1_y =
            0.25 *
            (arrOmega1[ez][ey][ex][slot_omega1_y] + // BACK_LEFT (当前单元)
             arrOmega1[ez][ey][ex + 1]
                      [slot_omega1_y] + // BACK_RIGHT (右邻居的BACK_LEFT)
             arrOmega1[ez][ey - 1][ex][slot_omega1_y] + // 下邻居的 BACK_LEFT
             arrOmega1[ez][ey - 1][ex + 1]
                      [slot_omega1_y]); // 右下邻居的 BACK_LEFT

        // ω₁ᶻ：在 z 方向棱上，需要插值到 x 方向棱的位置
        // x 方向棱（BACK_DOWN）位于 y=prev, z=prev（后面）
        // 使用四个位置的双线性插值（在 x 和 z 两个方向上）
        // 注意：ez 是当前单元的 z 索引，ez-1 是后邻居（更靠后），ez+1
        // 是前邻居（更靠前） BACK_DOWN 位于
        // z=prev（后面），所以应该使用当前层（ez）和后一层（ez-1）
        // 1. DOWN_LEFT (x=prev, y=prev, z=当前层) - 当前单元
        // 2. DOWN_RIGHT (x=next, y=prev, z=当前层) - 右邻居的 DOWN_LEFT
        // 3. 后邻居的 DOWN_LEFT (x=prev, y=prev, z=后一层) - 后邻居
        // 4. 后邻居的 DOWN_RIGHT (x=next, y=prev, z=后一层) - 后右邻居的
        // DOWN_LEFT
        omega1_z =
            0.25 *
            (arrOmega1[ez][ey][ex][slot_omega1_z] + // DOWN_LEFT (当前单元)
             arrOmega1[ez][ey][ex + 1]
                      [slot_omega1_z] + // DOWN_RIGHT (右邻居的DOWN_LEFT)
             arrOmega1[ez - 1][ey][ex]
                      [slot_omega1_z] + // 后邻居的 DOWN_LEFT (更靠后)
             arrOmega1[ez - 1][ey][ex + 1]
                      [slot_omega1_z]); // 后右邻居的 DOWN_LEFT (更靠后)

        // 列：涉及8个未知量（u₁ᶻ 和 u₁ʸ 各4个插值点）
        // 使用四个位置的双线性插值来插值 u₁ 到 x 方向棱的位置
        DMStagStencil col[8];
        PetscScalar val[8];
        PetscInt nCol = 0;
        PetscScalar coeff = 0.5 * 0.25; // 0.5 来自时间平均，0.25 来自四点平均

        // 第一项：+ω₁ʸ u₁ᶻ（插值）
        // u₁ᶻ 需要插值到 x 方向棱的位置，使用四个 z 方向棱（在 x 和 z
        // 两个方向上）
        // 1. DOWN_LEFT (当前单元)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_y;
        nCol++;

        // 2. DOWN_RIGHT (右邻居的 DOWN_LEFT)
        col[nCol].i = ex + 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_y;
        nCol++;

        // 3. 后邻居的 DOWN_LEFT
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_y;
        nCol++;

        // 4. 后右邻居的 DOWN_LEFT
        col[nCol].i = ex + 1;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_y;
        nCol++;

        // 第二项：-ω₁ᶻ u₁ʸ（插值）
        // u₁ʸ 需要插值到 x 方向棱的位置，使用四个 y 方向棱（在 x 和 y
        // 两个方向上）
        // 1. BACK_LEFT (当前单元)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_z; // 负号来自叉乘公式
        nCol++;

        // 2. BACK_RIGHT (右邻居的 BACK_LEFT)
        col[nCol].i = ex + 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_z;
        nCol++;

        // 3. 下邻居的 BACK_LEFT
        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_z;
        nCol++;

        // 4. 下右邻居的 BACK_LEFT
        col[nCol].i = ex + 1;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_z;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的对流项 =====
        // (ω₁ × u₁)_y = ω₁ᶻ u₁ˣ - ω₁ˣ u₁ᶻ
        row.loc = BACK_LEFT; // y方向棱
        nCol = 0;

        // 读取已知的涡度值（使用四个位置的双线性插值）
        PetscScalar omega1_x, omega1_z_y;

        // ω₁ᶻ：在 z 方向棱上，需要插值到 y 方向棱的位置
        // y 方向棱（BACK_LEFT）位于 x=prev, z=prev（后面）
        // 使用四个位置的双线性插值（在 y 和 z 两个方向上）
        // 注意：BACK_LEFT 位于
        // z=prev（后面），所以应该使用当前层（ez）和后一层（ez-1）
        // 1. DOWN_LEFT (x=prev, y=prev, z=当前层) - 当前单元
        // 2. UP_LEFT (x=prev, y=next, z=当前层) - 上邻居的 DOWN_LEFT
        // 3. 后邻居的 DOWN_LEFT (x=prev, y=prev, z=后一层) - 后邻居
        // 4. 后上邻居的 DOWN_LEFT (x=prev, y=next, z=后一层) - 后上邻居
        omega1_z_y =
            0.25 *
            (arrOmega1[ez][ey][ex][slot_omega1_z] + // DOWN_LEFT (当前单元)
             arrOmega1[ez][ey + 1][ex]
                      [slot_omega1_z] + // UP_LEFT (上邻居的DOWN_LEFT)
             arrOmega1[ez - 1][ey][ex]
                      [slot_omega1_z] + // 后邻居的 DOWN_LEFT (更靠后)
             arrOmega1[ez - 1][ey + 1][ex]
                      [slot_omega1_z]); // 后上邻居的 DOWN_LEFT (更靠后)

        // ω₁ˣ：在 x 方向棱上，需要插值到 y 方向棱的位置
        // y 方向棱（BACK_LEFT）位于 x=prev, z=prev
        // 使用四个位置的双线性插值（在 x 和 y 两个方向上）
        // 1. 当前单元的 x 方向棱 (BACK_DOWN, x=prev, y=prev, z=prev)
        // 2. 左邻居的 x 方向棱 (x=prev-1, y=prev, z=prev) - 更靠左
        // 3. 上邻居的 x 方向棱 (x=prev, y=next, z=prev)
        // 4. 左上邻居的 x 方向棱 (x=prev-1, y=next, z=prev)
        omega1_x =
            0.25 *
            (arrOmega1[ez][ey][ex][slot_omega1_x] + // BACK_DOWN (当前单元)
             arrOmega1[ez][ey][ex - 1]
                      [slot_omega1_x] + // 左邻居的 BACK_DOWN (更靠左)
             arrOmega1[ez][ey + 1][ex]
                      [slot_omega1_x] + // BACK_UP (上邻居的BACK_DOWN)
             arrOmega1[ez][ey + 1][ex - 1]
                      [slot_omega1_x]); // 左上邻居的 BACK_DOWN

        // 列：涉及8个未知量（u₁ˣ 和 u₁ᶻ 各4个插值点）
        // 注意：col 和 val 数组已在 x 方向棱部分声明，这里只重置 nCol
        nCol = 0;
        coeff = 0.5 * 0.25; // 0.5 来自时间平均，0.25 来自四点平均

        // 第一项：+ω₁ᶻ u₁ˣ（插值）
        // u₁ˣ 需要插值到 y 方向棱的位置，使用四个 x 方向棱（在 x 和 y
        // 两个方向上） y 方向棱（BACK_LEFT）位于 x=prev, z=prev，所以应该在 x
        // 和 y 方向上插值
        // 1. BACK_DOWN (当前单元, x=prev, y=prev, z=prev)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_z_y;
        nCol++;

        // 2. 左邻居的 BACK_DOWN (x=prev-1, y=prev, z=prev) - 更靠左
        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_z_y;
        nCol++;

        // 3. BACK_UP (上邻居的 BACK_DOWN, x=prev, y=next, z=prev)
        col[nCol].i = ex;
        col[nCol].j = ey + 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_z_y;
        nCol++;

        // 4. 左上邻居的 BACK_DOWN (x=prev-1, y=next, z=prev)
        col[nCol].i = ex - 1;
        col[nCol].j = ey + 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_z_y;
        nCol++;

        // 第二项：-ω₁ˣ u₁ᶻ（插值）
        // u₁ᶻ 需要插值到 y 方向棱的位置，使用四个 z 方向棱（在 y 和 z
        // 两个方向上） y 方向棱（BACK_LEFT）位于 x=prev, z=prev，所以应该在 y
        // 和 z 方向上插值
        // 1. DOWN_LEFT (当前单元, x=prev, y=prev, z=当前层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_x; // 负号来自叉乘公式
        nCol++;

        // 2. UP_LEFT (上邻居的 DOWN_LEFT, x=prev, y=next, z=当前层)
        col[nCol].i = ex;
        col[nCol].j = ey + 1;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_x;
        nCol++;

        // 3. 后邻居的 DOWN_LEFT (x=prev, y=prev, z=后一层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_x;
        nCol++;

        // 4. 后上邻居的 DOWN_LEFT (x=prev, y=next, z=后一层)
        col[nCol].i = ex;
        col[nCol].j = ey + 1;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_x;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的对流项 =====
        // (ω₁ × u₁)_z = ω₁ˣ u₁ʸ - ω₁ʸ u₁ˣ
        row.loc = DOWN_LEFT; // z方向棱
        nCol = 0;

        // 读取已知的涡度值（使用四个位置的双线性插值）
        PetscScalar omega1_x_z, omega1_y_z;

        // ω₁ˣ：在 x 方向棱上，需要插值到 z 方向棱的位置
        // z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，z 坐标是单元中心 ez
        // 使用四个位置的双线性插值（在 x 和 z 两个方向上）
        // 1. 本地 x 方向：BACK_DOWN (当前单元, x=prev, y=prev, z=当前层)
        // 2. 前方 x 方向：FRONT_DOWN (前邻居, x=prev, y=prev, z=前一层) - ez+1
        // 3. 左边 x 方向：左邻居的 BACK_DOWN (x=prev-1, y=prev, z=当前层)
        // 4. 左前 x 方向：左前邻居的 BACK_DOWN (x=prev-1, y=prev, z=前一层)
        omega1_x_z =
            0.25 * (arrOmega1[ez][ey][ex]
                             [slot_omega1_x] + // BACK_DOWN (当前单元, 本地)
                    arrOmega1[ez + 1][ey][ex]
                             [slot_omega1_x] + // FRONT_DOWN (前邻居, 前方)
                    arrOmega1[ez][ey][ex - 1]
                             [slot_omega1_x] + // 左邻居的 BACK_DOWN (左边)
                    arrOmega1[ez + 1][ey][ex - 1]
                             [slot_omega1_x]); // 左前邻居的 BACK_DOWN (左前)

        // ω₁ʸ：在 y 方向棱上，需要插值到 z 方向棱的位置
        // z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，z 坐标是单元中心 ez
        // 使用四个位置的双线性插值（在 y 和 z 两个方向上）
        // 1. 本地 y 方向：BACK_LEFT (当前单元, x=prev, y=prev, z=当前层)
        // 2. 前邻居：FRONT_LEFT (前邻居, x=prev, y=prev, z=前一层) - ez+1
        // 3. 下邻居：下邻居的 BACK_LEFT (x=prev, y=prev-1, z=当前层) - ey-1
        // 4. 前下邻居：前下邻居的 BACK_LEFT (x=prev, y=prev-1, z=前一层)
        omega1_y_z =
            0.25 *
            (arrOmega1[ez][ey][ex]
                      [slot_omega1_y] + // BACK_LEFT (当前单元, 本地)
             arrOmega1[ez + 1][ey][ex]
                      [slot_omega1_y] + // FRONT_LEFT (前邻居, 前方)
             arrOmega1[ez][ey - 1][ex]
                      [slot_omega1_y] + // 下邻居的 BACK_LEFT (下邻居)
             arrOmega1[ez + 1][ey - 1][ex]
                      [slot_omega1_y]); // 前下邻居的 BACK_LEFT (前下邻居)

        // 列：涉及8个未知量（u₁ʸ 和 u₁ˣ 各4个插值点）
        // 注意：col 和 val 数组已在 x 方向棱部分声明，这里只重置 nCol
        nCol = 0;
        coeff = 0.5 * 0.25; // 0.5 来自时间平均，0.25 来自四点平均

        // 第一项：+ω₁ˣ u₁ʸ（插值）
        // u₁ʸ 需要插值到 z 方向棱的位置，使用四个 y 方向棱（在 y 和 z
        // 两个方向上） z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，所以应该在 y
        // 和 z 方向上插值
        // 1. 本地 y 方向：BACK_LEFT (当前单元, x=prev, y=prev, z=当前层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_x_z;
        nCol++;

        // 2. 前邻居：FRONT_LEFT (前邻居, x=prev, y=prev, z=前一层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_x_z;
        nCol++;

        // 3. 下邻居：下邻居的 BACK_LEFT (x=prev, y=prev-1, z=当前层)
        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_x_z;
        nCol++;

        // 4. 前下邻居：前下邻居的 BACK_LEFT (x=prev, y=prev-1, z=前一层)
        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = coeff * omega1_x_z;
        nCol++;

        // 第二项：-ω₁ʸ u₁ˣ（插值）
        // u₁ˣ 需要插值到 z 方向棱的位置，使用四个 x 方向棱（在 x 和 z
        // 两个方向上） z 方向棱（DOWN_LEFT）位于 x=prev, y=prev，所以应该在 x
        // 和 z 方向上插值
        // 1. 本地 x 方向：BACK_DOWN (当前单元, x=prev, y=prev, z=当前层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_y_z; // 负号来自叉乘公式
        nCol++;

        // 2. 前方 x 方向：FRONT_DOWN (前邻居, x=prev, y=prev, z=前一层)
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_y_z;
        nCol++;

        // 3. 左边 x 方向：左邻居的 BACK_DOWN (x=prev-1, y=prev, z=当前层)
        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_y_z;
        nCol++;

        // 4. 左前 x 方向：左前邻居的 BACK_DOWN (x=prev-1, y=prev, z=前一层)
        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -coeff * omega1_y_z;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));
      }
    }
  }

  // 释放数组（使用 dmSol_2，因为是从 dmSol_2 获取的）
  PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localOmega1, &arrOmega1));
  PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1));

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装 omega2 的旋度项矩阵：∇ × ω₂
// 这是从2-形式（面）到1-形式（棱）的耦合项
// 对应方程5.4中的 (1/Re) ∇_h × ω_2^{k+1/2}
PetscErrorCode DUAL_MAC::assemble_omega2_curl_matrix(PetscReal Re, DM dmSol_1,
                                                     Mat A) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (xmax - xmin) / static_cast<PetscReal>(this->Nx);
  hy = (ymax - ymin) / static_cast<PetscReal>(this->Ny);
  hz = (zmax - zmin) / static_cast<PetscReal>(this->Nz);

  PetscScalar coeff = 1.0 / Re; // 1/Re 系数
  PetscPrintf(PETSC_COMM_WORLD, "assemble_omega2_curl_matrix: coeff: %g\n",
              coeff);
  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的旋度项 =====
        // (∇ × ω₂)_x = ∂ω₂ᶻ/∂y - ∂ω₂ʸ/∂z
        // 行：x方向棱（u₁ˣ 的方程）
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN; // x方向棱
        row.c = 0;

        // 列：涉及4个面上的 ω₂（2项，每项2个面）
        DMStagStencil col[4];
        PetscScalar val[4];
        PetscInt nCol = 0;

        // 第一项：∂ω₂ᶻ/∂y ≈ (ω₂ᶻ|_{ey} - ω₂ᶻ|_{ey-1}) / hy
        // ω₂ᶻ 在 xy 平面上（法向 z），即 BACK 和 FRONT 面
        // 使用本地的 BACK 面和下方单元的 BACK 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK;   // 本地的后面（z=prev，xy平面）
        col[nCol].c = 0;        // ω₂ 在 dof2 的第0个分量
        val[nCol] = coeff / hy; // 正号因为 ∂/∂y 的向后差分
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK; // 下方单元的后面
        col[nCol].c = 0;
        val[nCol] = -coeff / hy; // 负号因为 ∂/∂y 的向后差分
        nCol++;

        // 第二项：-∂ω₂ʸ/∂z ≈ -(ω₂ʸ|_{ez} - ω₂ʸ|_{ez-1}) / hz
        // ω₂ʸ 在 xz 平面上（法向 y），即 DOWN 和 UP 面
        // BACK_DOWN 棱位于 z=prev，所以用向后差分
        // 使用本地的 DOWN 面和后方单元的 DOWN 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN;    // 本地的下面（y=prev，xz平面）
        col[nCol].c = 0;         // ω₂ʸ 在 dof2 的第0个分量
        val[nCol] = -coeff / hz; // 负号来自 -(ω₂ʸ|_{ez} - ω₂ʸ|_{ez-1})
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN; // 后方单元的下面
        col[nCol].c = 0;
        val[nCol] = coeff / hz; // 正号来自 -(ω₂ʸ|_{ez} - ω₂ʸ|_{ez-1})
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的旋度项 =====
        // (∇ × ω₂)_y = ∂ω₂ˣ/∂z - ∂ω₂ᶻ/∂x
        row.loc = BACK_LEFT; // y方向棱
        nCol = 0;

        // 第一项：∂ω₂ˣ/∂z ≈ (ω₂ˣ|_{ez} - ω₂ˣ|_{ez-1}) / hz
        // ω₂ˣ 在 yz 平面上（法向 x），即 LEFT 和 RIGHT 面
        // BACK_LEFT 棱位于 z=prev，所以用向后差分
        // 使用本地的 LEFT 面和后方单元的 LEFT 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = LEFT; // 本地的左面（x=prev，yz平面）
        col[nCol].c = 0;
        val[nCol] = coeff / hz; // 正号来自 (ω₂ˣ|_{ez} - ω₂ˣ|_{ez-1})
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = LEFT; // 后方单元的左面
        col[nCol].c = 0;
        val[nCol] = -coeff / hz; // 负号来自 (ω₂ˣ|_{ez} - ω₂ˣ|_{ez-1})
        nCol++;

        // 第二项：-∂ω₂ᶻ/∂x ≈ -(ω₂ᶻ|_{ex} - ω₂ᶻ|_{ex-1}) / hx
        // BACK_LEFT 棱位于 x=prev，所以用向后差分
        // 使用本地的 BACK 面和左邻居的 BACK 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK; // 本地的后面（z=prev，xy平面）
        col[nCol].c = 0;
        val[nCol] = -coeff / hx; // 负号来自 -(ω₂ᶻ|_{ex} - ω₂ᶻ|_{ex-1})
        nCol++;

        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK; // 左邻居的后面
        col[nCol].c = 0;
        val[nCol] = coeff / hx; // 正号来自 -(ω₂ᶻ|_{ex} - ω₂ᶻ|_{ex-1})
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的旋度项 =====
        // (∇ × ω₂)_z = ∂ω₂ʸ/∂x - ∂ω₂ˣ/∂y
        row.loc = DOWN_LEFT; // z方向棱
        nCol = 0;

        // 第一项：∂ω₂ʸ/∂x ≈ (ω₂ʸ|_{ex} - ω₂ʸ|_{ex-1}) / hx
        // DOWN_LEFT 棱位于 x=prev，所以用向后差分
        // 使用本地的 DOWN 面和左邻居的 DOWN 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN; // 本地的下面（y=prev，xz平面）
        col[nCol].c = 0;
        val[nCol] = coeff / hx; // 正号来自 (ω₂ʸ|_{ex} - ω₂ʸ|_{ex-1})
        nCol++;

        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN; // 左邻居的下面
        col[nCol].c = 0;
        val[nCol] = -coeff / hx; // 负号来自 (ω₂ʸ|_{ex} - ω₂ʸ|_{ex-1})
        nCol++;

        // 第二项：-∂ω₂ˣ/∂y ≈ -(ω₂ˣ|_{ey} - ω₂ˣ|_{ey-1}) / hy
        // 使用本地的 LEFT 面和下方单元的 LEFT 面
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = LEFT; // 本地的左面（x=prev，yz平面）
        col[nCol].c = 0;
        val[nCol] = -coeff / hy; // 负号来自 -(ω₂ˣ|_{ey} - ω₂ˣ|_{ey-1})
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = LEFT; // 下方单元的左面
        col[nCol].c = 0;
        val[nCol] = coeff / hy; // 正号来自 -(ω₂ˣ|_{ey} - ω₂ˣ|_{ey-1})
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装 P₀ 压力梯度矩阵：∇P₀
// 这是从0-形式（顶点）到1-形式（棱）的耦合项
// 对应方程5.4中的 ∇_h P₀^{h,k}
// 在1-形式的动量方程中，压力梯度项是 ⟨∇P₀, ε₁⟩
PetscErrorCode DUAL_MAC::assemble_p0_gradient_matrix(DM dmSol_1, Mat A) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向棱 (BACK_DOWN) 的压力梯度项 =====
        // ∂P₀/∂x ≈ (P₀|_{RIGHT} - P₀|_{LEFT}) / hx
        // 行：x方向棱（u₁ˣ 的方程）
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN; // x方向棱
        row.c = 0;

        // 列：涉及2个顶点的 P₀
        DMStagStencil col[2];
        PetscScalar val[2];

        // 左顶点（x=prev）
        col[0].i = ex;
        col[0].j = ey;
        col[0].k = ez;
        col[0].loc = BACK_DOWN_LEFT; // 左下后顶点（x=prev, y=prev, z=prev）
        col[0].c = 0;                // P₀ 在 dof0 的第0个分量
        val[0] = -1.0 / hx;          // 负号因为向后差分

        // 右顶点（x=next）
        // 注意：右顶点是 (ex+1, ey, ez) 的左顶点
        col[1].i = ex + 1;
        col[1].j = ey;
        col[1].k = ez;
        col[1].loc = BACK_DOWN_LEFT; // 右邻居的左下后顶点 = 本单元的右下后顶点
        col[1].c = 0;
        val[1] = 1.0 / hx; // 正号因为向前差分

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 2, col, val,
                                            ADD_VALUES));

        // ===== y 方向棱 (BACK_LEFT) 的压力梯度项 =====
        // ∂P₀/∂y ≈ (P₀|_{UP} - P₀|_{DOWN}) / hy
        row.loc = BACK_LEFT; // y方向棱

        // 下顶点（y=prev）
        col[0].i = ex;
        col[0].j = ey;
        col[0].k = ez;
        col[0].loc = BACK_DOWN_LEFT; // 左下后顶点（y=prev）
        col[0].c = 0;
        val[0] = -1.0 / hy;

        // 上顶点（y=next）
        // 上顶点是 (ex, ey+1, ez) 的下顶点
        col[1].i = ex;
        col[1].j = ey + 1;
        col[1].k = ez;
        col[1].loc =
            BACK_DOWN_LEFT; // 上邻居的左下后顶点 = 本单元的左下后顶点（y方向）
        col[1].c = 0;
        val[1] = 1.0 / hy;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 2, col, val,
                                            ADD_VALUES));

        // ===== z 方向棱 (DOWN_LEFT) 的压力梯度项 =====
        // ∂P₀/∂z ≈ (P₀|_{FRONT} - P₀|_{BACK}) / hz
        row.loc = DOWN_LEFT; // z方向棱

        // 后顶点（z=prev）
        col[0].i = ex;
        col[0].j = ey;
        col[0].k = ez;
        col[0].loc = BACK_DOWN_LEFT; // 左下后顶点（z=prev）
        col[0].c = 0;
        val[0] = -1.0 / hz;

        // 前顶点（z=next）
        // 前顶点是 (ex, ey, ez+1) 的后顶点
        col[1].i = ex;
        col[1].j = ey;
        col[1].k = ez + 1;
        col[1].loc =
            BACK_DOWN_LEFT; // 前邻居的左下后顶点 = 本单元的左下后顶点（z方向）
        col[1].c = 0;
        val[1] = 1.0 / hz;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 2, col, val,
                                            ADD_VALUES));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装 u1-omega2耦合矩阵：ω₂ - ∇×u₁ = 0
// 这是从1-形式（棱）到2-形式（面）的耦合项
// 对应方程 ω₂ = ∇×u₁
// 行：面（2-形式，ω₂ 的位置）
// 列：棱（1-形式，u₁ 的位置）
PetscErrorCode DUAL_MAC::assemble_u1_omega2_coupling_matrix(DM dmSol_1, Mat A) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 遍历所有单元
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== x 方向面 (LEFT) 的耦合方程 =====
        // ω₂ₓ - (∇×u₁)ₓ = ω₂ₓ - (∂u₁ᶻ/∂y - ∂u₁ʸ/∂z) = 0
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = LEFT;
        row.c = 0;

        DMStagStencil col[5];
        PetscScalar val[5];
        PetscInt nCol = 0;

        // -∂u₁ᶻ/∂y
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hy;
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey + 1;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hy;
        nCol++;

        // +∂u₁ʸ/∂z
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hz;
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hz;
        nCol++;

        // +ω₂ₓ 对角项
        col[nCol] = row;
        val[nCol] = 1.0;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== y 方向面 (DOWN) 的耦合方程 =====
        // ω₂ᵧ - (∂u₁ˣ/∂z - ∂u₁ᶻ/∂x) = 0
        row.loc = DOWN;
        nCol = 0;

        // -∂u₁ˣ/∂z
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hz;
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez + 1;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hz;
        nCol++;

        // +∂u₁ᶻ/∂x
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hx;
        nCol++;

        col[nCol].i = ex + 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hx;
        nCol++;

        // +ω₂ᵧ 对角项
        col[nCol] = row;
        val[nCol] = 1.0;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));

        // ===== z 方向面 (BACK) 的耦合方程 =====
        // ω₂ᵤ - (∂u₁ʸ/∂x - ∂u₁ˣ/∂y) = 0
        row.loc = BACK;
        nCol = 0;

        // -∂u₁ʸ/∂x
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hx;
        nCol++;

        col[nCol].i = ex + 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hx;
        nCol++;

        // +∂u₁ˣ/∂y
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = -1.0 / hy;
        nCol++;

        col[nCol].i = ex;
        col[nCol].j = ey + 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN;
        col[nCol].c = 0;
        val[nCol] = 1.0 / hy;
        nCol++;

        // +ω₂ᶻ 对角项
        col[nCol] = row;
        val[nCol] = 1.0;
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
// 组装u1散度为0矩阵：∇·u₁ = 0
// 这是从1-形式（棱）到0-形式（顶点）的耦合项
// 对应连续性方程 ∇·u₁ = 0
// 这是压力梯度矩阵的转置（或负转置，取决于符号约定）
PetscErrorCode DUAL_MAC::assemble_u1_divergence_matrix(DM dmSol_1, Mat A) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmSol_1, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));
  PetscReal hx, hy, hz;
  hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
  hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
  hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

  // 遍历所有单元
  // 每个单元的左下后顶点（BACK_DOWN_LEFT）对应一个散度方程
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {

        // ===== 顶点 (BACK_DOWN_LEFT) 的散度方程 =====
        // ∇·u₁ = (∂u₁ˣ/∂x + ∂u₁ʸ/∂y + ∂u₁ᶻ/∂z) = 0
        // 行：顶点（P₀ 的方程）
        DMStagStencil row;
        row.i = ex;
        row.j = ey;
        row.k = ez;
        row.loc = BACK_DOWN_LEFT; // 左下后顶点（x=prev, y=prev, z=prev）
        row.c = 0;                // P₀ 在 dof0 的第0个分量

        // 启用 pinPressure 时，用单位行替换一个压力方程：p0(0,0,0)=0
        if (this->pinPressure && ex == 0 && ey == 0 && ez == 0) {
          DMStagStencil pcol = row;
          PetscScalar one = 1.0;
          PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, 1, &pcol,
                                              &one, ADD_VALUES));
          continue;
        }

        // 列：涉及6条棱的 u₁（3个方向，每个方向2条棱）
        DMStagStencil col[6];
        PetscScalar val[6];
        PetscInt nCol = 0;

        // ===== x 方向分量：∂u₁ˣ/∂x ≈ (u₁ˣ|_{x=next} - u₁ˣ|_{x=prev}) / hx
        // ===== u₁ˣ|_{x=next}：当前单元的 x 方向棱（BACK_DOWN，位于 x=next）
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN; // x方向棱
        col[nCol].c = 0;           // u₁ˣ 在 dof1 的第0个分量
        val[nCol] = 1.0 / hx; // 正号因为 (u₁ˣ|_{x=next} - u₁ˣ|_{x=prev})
        nCol++;

        // u₁ˣ|_{x=prev}：左邻居的 x 方向棱（BACK_DOWN，位于 x=prev）
        col[nCol].i = ex - 1;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_DOWN; // 左邻居的 x 方向棱
        col[nCol].c = 0;
        val[nCol] = -1.0 / hx; // 负号因为 (u₁ˣ|_{x=next} - u₁ˣ|_{x=prev})
        nCol++;

        // ===== y 方向分量：∂u₁ʸ/∂y ≈ (u₁ʸ|_{y=next} - u₁ʸ|_{y=prev}) / hy
        // ===== u₁ʸ|_{y=next}：当前单元的 y 方向棱（BACK_LEFT，位于 y=next）
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT; // y方向棱
        col[nCol].c = 0;           // u₁ʸ 在 dof1 的第0个分量
        val[nCol] = 1.0 / hy; // 正号因为 (u₁ʸ|_{y=next} - u₁ʸ|_{y=prev})
        nCol++;

        // u₁ʸ|_{y=prev}：下邻居的 y 方向棱（BACK_LEFT，位于 y=prev）
        col[nCol].i = ex;
        col[nCol].j = ey - 1;
        col[nCol].k = ez;
        col[nCol].loc = BACK_LEFT; // 下邻居的 y 方向棱
        col[nCol].c = 0;
        val[nCol] = -1.0 / hy; // 负号因为 (u₁ʸ|_{y=next} - u₁ʸ|_{y=prev})
        nCol++;

        // ===== z 方向分量：∂u₁ᶻ/∂z ≈ (u₁ᶻ|_{z=next} - u₁ᶻ|_{z=prev}) / hz
        // ===== u₁ᶻ|_{z=next}：当前单元的 z 方向棱（DOWN_LEFT，位于 z=next）
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez;
        col[nCol].loc = DOWN_LEFT; // z方向棱
        col[nCol].c = 0;           // u₁ᶻ 在 dof1 的第0个分量
        val[nCol] = 1.0 / hz; // 正号因为 (u₁ᶻ|_{z=next} - u₁ᶻ|_{z=prev})
        nCol++;

        // u₁ᶻ|_{z=prev}：后邻居的 z 方向棱（DOWN_LEFT，位于 z=prev）
        col[nCol].i = ex;
        col[nCol].j = ey;
        col[nCol].k = ez - 1;
        col[nCol].loc = DOWN_LEFT; // 后邻居的 z 方向棱
        col[nCol].c = 0;
        val[nCol] = -1.0 / hz; // 负号因为 (u₁ᶻ|_{z=next} - u₁ᶻ|_{z=prev})
        nCol++;

        PetscCall(DMStagMatSetValuesStencil(dmSol_1, A, 1, &row, nCol, col, val,
                                            ADD_VALUES));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// #pragma once
#include <petsc.h>
#include "../include/DUAL_MAC.h"
#include "petscdmstag.h"
#include "petscsystypes.h"
#include "../include/ref_sol.h"
// 2形式系统矩阵组装
// 根据论文第5章方程(5.1)-(5.3)，组装整数时间步的系统矩阵
// 系统包含三个方程：
// 1. 动量方程(5.1)：时间导数 + 对流项 + 旋度项 + 压力梯度 = 右端项
// 2. 耦合方程(5.2)：∇×u₂ - ω₁ = 0
// 3. 散度方程(5.3)：∇·u₂ = 0
PetscErrorCode DUAL_MAC::assemble_2form_system_matrix(DM dmSol_1, DM dmSol_2, Mat A, Vec rhs, Vec u2_prev, Vec omega2_known, Vec omega1_prev, ExternalForce externalForce, PetscReal time, PetscReal dt)
{
    PetscFunctionBeginUser;
    DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 系统矩阵组装开始\n");

    // 计算雷诺数（假设 nu 是动力粘度，Re = 1/nu）
    PetscReal Re = 1.0 / this->nu;

    // ===== 1. 初始化矩阵和右端项 =====
    PetscCall(MatZeroEntries(A));
    PetscCall(VecZeroEntries(rhs));

    // ===== 2. 组装时间导数矩阵：1/dt * I（对 u₂）=====
    // 对应方程(5.1)中的 (u₂^{k} - u₂^{k-1})/dt
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 时间导数矩阵组装开始\n");
    PetscCall(assemble_u2_dt_matrix(dmSol_2, A, dt));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 时间导数矩阵组装完成\n");

    // ===== 3. 组装对流项矩阵：0.5 * ω₂^{h,k-1/2} ×（对 u₂）=====
    // 对应方程(5.1)中的 ω₂^{h,k-1/2} × (u₂^{k} + u₂^{k-1})/2
    // 注意：assemble_u2_conv_matrix 内部已经包含了 0.5 的系数
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 对流项矩阵组装开始\n");
    PetscCall(assemble_u2_conv_matrix(dmSol_2, A, dmSol_1, omega2_known));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 对流项矩阵组装完成\n");

    // ===== 4. 组装旋度项矩阵：0.5/Re * ∇×（对 ω₁）=====
    // 对应方程(5.1)中的 (1/Re) ∇_h × (ω₁^{k} + ω₁^{k-1})/2
    // 由于 assemble_omega1_curl_matrix 使用 coeff = 1.0/Re
    // 我们需要传入 2*Re 来得到 coeff = 1.0/(2*Re) = 0.5/Re 的效果
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 旋度项矩阵组装开始\n");
    PetscCall(assemble_omega1_curl_matrix(2.0 * Re, dmSol_2, A));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 旋度项矩阵组装完成\n");

    // ===== 5. 组装压力梯度矩阵：∇（对 P₃）=====
    // 对应方程(5.1)中的 ∇_h P₃^{h,k-1/2}
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 压力梯度矩阵组装开始\n");
    PetscCall(assemble_p3_gradient_matrix(dmSol_2, A));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 压力梯度矩阵组装完成\n");

    // ===== 6. 组装耦合矩阵：∇×（对 u₂）和 -I（对 ω₁）=====
    // 对应方程(5.2)中的 ∇_h × u₂^{h,k} - ω₁^{h,k} = 0
    // 注意：assemble_u2_omega1_coupling_matrix 组装的是 ∇×u₂ - ω₁ = 0
    // 所以它已经包含了 -I 项（对 ω₁）
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 耦合矩阵组装开始\n");
    PetscCall(assemble_u2_omega1_coupling_matrix(dmSol_2, A));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 耦合矩阵组装完成\n");

    // ===== 7. 组装散度矩阵：∇·（对 u₂）=====
    // 对应方程(5.3)中的 ∇_h · u₂^{h,k} = 0
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 散度矩阵组装开始\n");
    PetscCall(assemble_u2_divergence_matrix(dmSol_2, A));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 散度矩阵组装完成\n");

    // ===== 8. 组装右端项（不包括外力）=====
    // 包含时间导数项、对流项和旋度项：
    // - u₂^{k-1}/dt（时间导数项）
    // - 0.5 * ω₂^{h,k-1/2} × u₂^{k-1}（对流项）
    // - -0.5/Re * ∇_h × ω₁^{h,k-1}（旋度项）
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 右端项组装开始\n");
    PetscCall(assemble_rhs2_vector(dmSol_2, rhs, u2_prev, dmSol_1, omega2_known, omega1_prev, dt));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 右端项组装完成\n");

    // ===== 9. 组装外力项 =====
    // 对应方程(5.1)中的 I_{RT} f^{k-1/2}
    // 外力项直接添加到右端项向量中
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 外力项组装开始\n");
    PetscCall(assemble_force2_vector(dmSol_2, rhs, externalForce, time));
    DUAL_MAC_DEBUG_LOG("[DEBUG] [2-form] 外力项组装完成\n");

    // ===== 10. 矩阵和向量最终组装 =====
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyEnd(rhs));

    DUAL_MAC_DEBUG_LOG("[DEBUG] 2-form 系统矩阵组装完成\n");
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装右端项(不包括外力)
// 根据leapfrog时间步进，右端项包含：
// 1. 时间导数项：u₂^{k-1} / dt（上一步的速度）
// 2. 对流项：ω₂^{h,k-1/2} × u₂^{k-1}（上一步的对流项）
// 3. 旋度项：-0.5/Re * ∇_h × ω₁^{h,k-1}（上一步的旋度项）
PetscErrorCode DUAL_MAC::assemble_rhs2_vector(DM dmSol_2, Vec rhs, Vec u2_prev, DM dmSol_1, Vec omega2_known, Vec omega1_prev, PetscReal dt)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

    // 计算雷诺数和旋度项系数
    PetscReal Re = 1.0 / this->nu;
    PetscScalar curl_coeff = -0.5 / Re;  // -0.5/Re 系数（来自方程5.1）

    // 获取上一步速度场的本地数组（u₂ 存储在 dmSol_2 的 dof2，slot=1）
    Vec localU2Prev;
    PetscScalar ****arrU2Prev;
    PetscCall(DMGetLocalVector(dmSol_2, &localU2Prev));
    PetscCall(DMGlobalToLocal(dmSol_2, u2_prev, INSERT_VALUES, localU2Prev));
    PetscCall(DMStagVecGetArrayRead(dmSol_2, localU2Prev, &arrU2Prev));

    // 获取已知涡度场的本地数组（ω₂ 存储在 dmSol_1 的 dof2，slot=0）
    Vec localOmega2;
    PetscScalar ****arrOmega2;
    PetscCall(DMGetLocalVector(dmSol_1, &localOmega2));
    PetscCall(DMGlobalToLocal(dmSol_1, omega2_known, INSERT_VALUES, localOmega2));
    PetscCall(DMStagVecGetArrayRead(dmSol_1, localOmega2, &arrOmega2));

    // 获取上一步1形式涡度场的本地数组（ω₁ 存储在 dmSol_2 的 dof1，slot=0）
    Vec localOmega1Prev;
    PetscScalar ****arrOmega1Prev;
    PetscCall(DMGetLocalVector(dmSol_2, &localOmega1Prev));
    PetscCall(DMGlobalToLocal(dmSol_2, omega1_prev, INSERT_VALUES, localOmega1Prev));
    PetscCall(DMStagVecGetArrayRead(dmSol_2, localOmega1Prev, &arrOmega1Prev));

    // 获取 slot 索引
    // u₂ 在 dmSol_2 的 dof2 上（slot=0，因为 dof2 只有1个分量：u2在slot=0）
    PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
    PetscCall(DMStagGetLocationSlot(dmSol_2, LEFT, 0, &slot_u2_x));   // x方向面
    PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN, 0, &slot_u2_y));   // y方向面
    PetscCall(DMStagGetLocationSlot(dmSol_2, BACK, 0, &slot_u2_z));   // z方向面

    // ω₂ 在 dmSol_1 的 dof2 上（slot=0）
    PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
    PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));   // x方向面（LEFT/RIGHT）
    PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));   // y方向面（DOWN/UP）
    PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));   // z方向面（BACK/FRONT）

    // ω₁ 在 dmSol_2 的 dof1 上（slot=0）
    PetscInt slot_omega1_x, slot_omega1_y, slot_omega1_z;
    PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_DOWN, 0, &slot_omega1_x));  // x方向棱
    PetscCall(DMStagGetLocationSlot(dmSol_2, BACK_LEFT, 0, &slot_omega1_y));  // y方向棱
    PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN_LEFT, 0, &slot_omega1_z));  // z方向棱

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向面 (LEFT) 的右端项 =====
          // 1. 时间导数项：u₂ˣ^{k-1} / dt
          // 2. 对流项：ω₂^{h,k-1/2} × u₂^{h,k-1} 的 x 分量
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = LEFT;  // x方向面
          row.c = 0;        // u₂ 在 dof2 的第0个分量

          PetscScalar rhs_val = 0.0;

          // 时间导数项：u₂ˣ^{k-1} / dt
          rhs_val += arrU2Prev[ez][ey][ex][slot_u2_x] / dt;

          // 对流项：ω₂^{h,k-1/2} × u₂^{h,k-1} 的 x 分量
          // (ω₂ × u₂)_x = ω₂ʸ u₂ᶻ - ω₂ᶻ u₂ʸ
          // 需要插值 ω₂ 和 u₂ 到 x 方向面的位置
          
          // ω₂ʸ：在 y 方向面上，需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // y 方向面（DOWN）位于 y=prev，需要在 x 和 y 两个方向上插值
          // 应该是：本地DOWN面，左边DOWN面，上边单元的DOWN面，左上单元的DOWN面
          PetscScalar omega2_y = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_y] +      // 本地DOWN面
                                         arrOmega2[ez][ey][ex-1][slot_omega2_y] +    // 左边DOWN面
                                         arrOmega2[ez][ey+1][ex][slot_omega2_y] +    // 上边单元的DOWN面
                                         arrOmega2[ez][ey+1][ex-1][slot_omega2_y]);  // 左上单元的DOWN面

          // ω₂ᶻ：在 z 方向面上，需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 x 和 z 两个方向上插值
          // 应该是：本地BACK面，左边BACK面，前方单元的BACK面，左前单元的BACK面
          PetscScalar omega2_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_z] +      // 本地BACK面
                                         arrOmega2[ez][ey][ex-1][slot_omega2_z] +    // 左边BACK面
                                         arrOmega2[ez+1][ey][ex][slot_omega2_z] +      // 前方单元的BACK面
                                         arrOmega2[ez+1][ey][ex-1][slot_omega2_z]);   // 左前单元的BACK面

          // u₂ᶻ^{k-1}：需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 x 和 z 两个方向上插值
          PetscScalar u2_z_prev = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_z] +      // 本地BACK面
                                         arrU2Prev[ez][ey][ex-1][slot_u2_z] +    // 左边BACK面
                                         arrU2Prev[ez+1][ey][ex][slot_u2_z] +      // 前方单元的BACK面
                                         arrU2Prev[ez+1][ey][ex-1][slot_u2_z]);   // 左前单元的BACK面

          // u₂ʸ^{k-1}：需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // y 方向面（DOWN）位于 y=prev，需要在 x 和 y 两个方向上插值
          PetscScalar u2_y_prev = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_y] +      // 本地DOWN面
                                         arrU2Prev[ez][ey][ex-1][slot_u2_y] +    // 左边DOWN面
                                         arrU2Prev[ez][ey+1][ex][slot_u2_y] +    // 上边单元的DOWN面
                                         arrU2Prev[ez][ey+1][ex-1][slot_u2_y]);  // 左上单元的DOWN面

          // 对流项：0.5 * (ω₂ʸ u₂ᶻ - ω₂ᶻ u₂ʸ)（0.5 来自时间平均）
          rhs_val += 0.5 * (omega2_y * u2_z_prev - omega2_z * u2_y_prev);

          // 旋度项：-0.5/Re * (∇ × ω₁^{k-1})_x
          // (∇ × ω₁)_x = ∂ω₁ᶻ/∂y - ∂ω₁ʸ/∂z
          // 第一项：∂ω₁ᶻ/∂y ≈ (ω₁ᶻ|_{ey} - ω₁ᶻ|_{ey-1}) / hy
          // ω₁ᶻ 在 z 方向棱上（DOWN_LEFT）
          PetscScalar curl_x = 0.0;
          curl_x += (arrOmega1Prev[ez][ey][ex][slot_omega1_z] - arrOmega1Prev[ez][ey-1][ex][slot_omega1_z]) / hy;
          // 第二项：-∂ω₁ʸ/∂z ≈ -(ω₁ʸ|_{ez} - ω₁ʸ|_{ez-1}) / hz
          // ω₁ʸ 在 y 方向棱上（BACK_LEFT）
          curl_x -= (arrOmega1Prev[ez][ey][ex][slot_omega1_y] - arrOmega1Prev[ez-1][ey][ex][slot_omega1_y]) / hz;
          rhs_val += curl_coeff * curl_x;

          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &rhs_val, ADD_VALUES));

          // ===== y 方向面 (DOWN) 的右端项 =====
          row.loc = DOWN;  // y方向面
          row.c = 0;
          rhs_val = 0.0;

          // 时间导数项：u₂ʸ^{k-1} / dt
          rhs_val += arrU2Prev[ez][ey][ex][slot_u2_y] / dt;

          // 对流项：(ω₂ × u₂)_y = ω₂ᶻ u₂ˣ - ω₂ˣ u₂ᶻ
          // ω₂ᶻ：在 z 方向面上，需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地BACK面，前方BACK面，下方BACK面，前下BACK面
          PetscScalar omega2_z_y = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_z] +      // 本地BACK面
                                            arrOmega2[ez+1][ey][ex][slot_omega2_z] +      // 前方BACK面
                                            arrOmega2[ez][ey-1][ex][slot_omega2_z] +    // 下方BACK面
                                            arrOmega2[ez+1][ey-1][ex][slot_omega2_z]);   // 前下BACK面

          // ω₂ˣ：在 x 方向面上，需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 y 两个方向上插值
          // 应该是：本地LEFT面，下边LEFT面，右边LEFT面，右下LEFT面
          PetscScalar omega2_x = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_x] +      // 本地LEFT面
                                         arrOmega2[ez][ey-1][ex][slot_omega2_x] +    // 下边LEFT面
                                         arrOmega2[ez][ey][ex+1][slot_omega2_x] +    // 右边LEFT面
                                         arrOmega2[ez][ey-1][ex+1][slot_omega2_x]);   // 右下LEFT面

          // u₂ˣ^{k-1}：需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 y 两个方向上插值
          PetscScalar u2_x_prev = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_x] +      // 本地LEFT面
                                         arrU2Prev[ez][ey-1][ex][slot_u2_x] +    // 下边LEFT面
                                         arrU2Prev[ez][ey][ex+1][slot_u2_x] +    // 右边LEFT面
                                         arrU2Prev[ez][ey-1][ex+1][slot_u2_x]);   // 右下LEFT面

          // u₂ᶻ^{k-1}：需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 y 和 z 两个方向上插值
          PetscScalar u2_z_prev_y = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_z] +      // 本地BACK面
                                            arrU2Prev[ez+1][ey][ex][slot_u2_z] +      // 前方BACK面
                                            arrU2Prev[ez][ey-1][ex][slot_u2_z] +    // 下方BACK面
                                            arrU2Prev[ez+1][ey-1][ex][slot_u2_z]);   // 前下BACK面

          rhs_val += 0.5 * (omega2_z_y * u2_x_prev - omega2_x * u2_z_prev_y);

          // 旋度项：-0.5/Re * (∇ × ω₁^{k-1})_y
          // (∇ × ω₁)_y = ∂ω₁ˣ/∂z - ∂ω₁ᶻ/∂x
          PetscScalar curl_y = 0.0;
          curl_y += (arrOmega1Prev[ez][ey][ex][slot_omega1_x] - arrOmega1Prev[ez-1][ey][ex][slot_omega1_x]) / hz;
          curl_y -= (arrOmega1Prev[ez][ey][ex][slot_omega1_z] - arrOmega1Prev[ez][ey][ex-1][slot_omega1_z]) / hx;
          rhs_val += curl_coeff * curl_y;

          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &rhs_val, ADD_VALUES));

          // ===== z 方向面 (BACK) 的右端项 =====
          row.loc = BACK;  // z方向面
          row.c = 0;
          rhs_val = 0.0;

          // 时间导数项：u₂ᶻ^{k-1} / dt
          rhs_val += arrU2Prev[ez][ey][ex][slot_u2_z] / dt;

          // 对流项：(ω₂ × u₂)_z = ω₂ˣ u₂ʸ - ω₂ʸ u₂ˣ
          // ω₂ˣ：在 x 方向面上，需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 z 两个方向上插值
          // 应该是：本地LEFT面，后方LEFT面，右边LEFT面，右后LEFT面
          PetscScalar omega2_x_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_x] +      // 本地LEFT面
                                            arrOmega2[ez-1][ey][ex][slot_omega2_x] +    // 后方LEFT面
                                            arrOmega2[ez][ey][ex+1][slot_omega2_x] +      // 右边LEFT面
                                            arrOmega2[ez-1][ey][ex+1][slot_omega2_x]);   // 右后LEFT面

          // ω₂ʸ：在 y 方向面上，需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // y 方向面（DOWN）位于 y=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地DOWN面，上方DOWN面，后方DOWN面，后上DOWN面
          PetscScalar omega2_y_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_y] +      // 本地DOWN面
                                            arrOmega2[ez][ey+1][ex][slot_omega2_y] +    // 上方DOWN面
                                            arrOmega2[ez-1][ey][ex][slot_omega2_y] +      // 后方DOWN面
                                            arrOmega2[ez-1][ey+1][ex][slot_omega2_y]);   // 后上DOWN面

          // u₂ʸ^{k-1}：需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // y 方向面（DOWN）位于 y=prev，需要在 y 和 z 两个方向上插值
          PetscScalar u2_y_prev_z = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_y] +      // 本地DOWN面
                                            arrU2Prev[ez][ey+1][ex][slot_u2_y] +    // 上方DOWN面
                                            arrU2Prev[ez-1][ey][ex][slot_u2_y] +      // 后方DOWN面
                                            arrU2Prev[ez-1][ey+1][ex][slot_u2_y]);   // 后上DOWN面

          // u₂ˣ^{k-1}：需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 z 两个方向上插值
          PetscScalar u2_x_prev_z = 0.25 * (arrU2Prev[ez][ey][ex][slot_u2_x] +      // 本地LEFT面
                                            arrU2Prev[ez-1][ey][ex][slot_u2_x] +    // 后方LEFT面
                                            arrU2Prev[ez][ey][ex+1][slot_u2_x] +      // 右边LEFT面
                                            arrU2Prev[ez-1][ey][ex+1][slot_u2_x]);   // 右后LEFT面

          rhs_val += 0.5 * (omega2_x_z * u2_y_prev_z - omega2_y_z * u2_x_prev_z);

          // 旋度项：-0.5/Re * (∇ × ω₁^{k-1})_z
          // (∇ × ω₁)_z = ∂ω₁ʸ/∂x - ∂ω₁ˣ/∂y
          PetscScalar curl_z = 0.0;
          curl_z += (arrOmega1Prev[ez][ey][ex][slot_omega1_y] - arrOmega1Prev[ez][ey][ex-1][slot_omega1_y]) / hx;
          curl_z -= (arrOmega1Prev[ez][ey][ex][slot_omega1_x] - arrOmega1Prev[ez][ey-1][ex][slot_omega1_x]) / hy;
          rhs_val += curl_coeff * curl_z;

          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &rhs_val, ADD_VALUES));
        }
      }
    }

    // 释放数组
    PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localU2Prev, &arrU2Prev));
    PetscCall(DMRestoreLocalVector(dmSol_2, &localU2Prev));
    PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localOmega2, &arrOmega2));
    PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2));
    PetscCall(DMStagVecRestoreArrayRead(dmSol_2, localOmega1Prev, &arrOmega1Prev));
    PetscCall(DMRestoreLocalVector(dmSol_2, &localOmega1Prev));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装u2时间矩阵
PetscErrorCode DUAL_MAC::assemble_u2_dt_matrix(DM dmSol_2, Mat A, PetscReal dt)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));

    PetscScalar coeff = 1.0 / dt;

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // u₂ 存储在面上（2形式）
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.c = 0;   // u₂ 在 dof2 的第0个分量

          // ---- x 方向面 (LEFT) ----
          row.loc = LEFT;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row,
                    &coeff, ADD_VALUES));

          // ---- y 方向面 (DOWN) ----
          row.loc = DOWN;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row,
                    &coeff, ADD_VALUES));

          // ---- z 方向面 (BACK) ----
          row.loc = BACK;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row,
                    &coeff, ADD_VALUES));
        }
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装 u₂ 对流项矩阵：ω₂ × u₂
// 这是2-形式（面）到2-形式（面）的耦合项
// 对应方程5.1中的 ω₂^{h,k-1/2} × (u₂^{h,k} + u₂^{h,k-1})/2
// 注意：这里使用已知的 ω₂^{h,k-1/2}（从半整数步系统得到）来线性化对流项
// 重要：omega2_known 存储在 dmSol_1 中，必须使用 dmSol_1 来读取
PetscErrorCode DUAL_MAC::assemble_u2_conv_matrix(DM dmSol_2, Mat A, DM dmSol_1, Vec omega2_known)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

    // 获取已知涡度场的本地数组（使用 dmSol_1，因为 ω₂ 存储在 dmSol_1 中）
    Vec localOmega2;
    PetscScalar ****arrOmega2;
    PetscCall(DMGetLocalVector(dmSol_1, &localOmega2));
    PetscCall(DMGlobalToLocal(dmSol_1, omega2_known, INSERT_VALUES, localOmega2));
    PetscCall(DMStagVecGetArrayRead(dmSol_1, localOmega2, &arrOmega2));

    // 获取 slot 索引
    // ω₂ 在 dmSol_1 的 dof2 上（slot=0）
    PetscInt slot_omega2_x, slot_omega2_y, slot_omega2_z;
    PetscCall(DMStagGetLocationSlot(dmSol_1, LEFT, 0, &slot_omega2_x));   // x方向面
    PetscCall(DMStagGetLocationSlot(dmSol_1, DOWN, 0, &slot_omega2_y));   // y方向面
    PetscCall(DMStagGetLocationSlot(dmSol_1, BACK, 0, &slot_omega2_z));   // z方向面

    // u₂ 在 dmSol_2 的 dof2 上（slot=0）
    PetscInt slot_u2_x, slot_u2_y, slot_u2_z;
    PetscCall(DMStagGetLocationSlot(dmSol_2, LEFT, 0, &slot_u2_x));   // x方向面
    PetscCall(DMStagGetLocationSlot(dmSol_2, DOWN, 0, &slot_u2_y));   // y方向面
    PetscCall(DMStagGetLocationSlot(dmSol_2, BACK, 0, &slot_u2_z));   // z方向面

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向面 (LEFT) 的对流项 =====
          // (ω₂ × u₂)_x = ω₂ʸ u₂ᶻ - ω₂ᶻ u₂ʸ
          // 行：x方向面（u₂ˣ 的方程）
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = LEFT;  // x方向面
          row.c = 0;        // u₂ 在 dof2 的第0个分量

          // 读取已知的涡度值（在面上，需要插值到当前面的位置）
          // 使用四个位置的双线性插值
          PetscScalar omega2_y, omega2_z;
          
          // ω₂ʸ：在 y 方向面上，需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // y 方向面（DOWN）位于 y=prev，需要在 x 和 y 两个方向上插值
          omega2_y = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_y] +      // 本地DOWN面
                             arrOmega2[ez][ey][ex-1][slot_omega2_y] +    // 左边DOWN面
                             arrOmega2[ez][ey+1][ex][slot_omega2_y] +    // 上边单元的DOWN面
                             arrOmega2[ez][ey+1][ex-1][slot_omega2_y]);  // 左上单元的DOWN面
          
          // ω₂ᶻ：在 z 方向面上，需要插值到 x 方向面的位置
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 x 和 z 两个方向上插值
          omega2_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_z] +      // 本地BACK面
                             arrOmega2[ez][ey][ex-1][slot_omega2_z] +    // 左边BACK面
                             arrOmega2[ez+1][ey][ex][slot_omega2_z] +      // 前方单元的BACK面
                             arrOmega2[ez+1][ey][ex-1][slot_omega2_z]);   // 左前单元的BACK面

          // 列：涉及8个未知量（u₂ᶻ 和 u₂ʸ 各4个插值点）
          DMStagStencil col[8];
          PetscScalar val[8];
          PetscInt nCol = 0;
          PetscScalar coeff = 0.5 * 0.25;  // 0.5 来自时间平均，0.25 来自四点平均

          // 第一项：+ω₂ʸ u₂ᶻ（插值）
          // u₂ᶻ 需要插值到 x 方向面的位置，使用四个 z 方向面
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 x 和 z 两个方向上插值
          // 应该是：本地BACK面，左边BACK面，前方单元的BACK面，左前单元的BACK面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 本地BACK面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_y;
          nCol++;

          col[nCol].i = ex - 1;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 左边BACK面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_y;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK;  // 前方单元的BACK面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_y;
          nCol++;

          col[nCol].i = ex - 1;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK;  // 左前单元的BACK面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_y;
          nCol++;

          // 第二项：-ω₂ᶻ u₂ʸ（插值）
          // u₂ʸ 需要插值到 x 方向面的位置，使用四个 y 方向面
          // x 方向面（LEFT）位于 x=prev, y=center, z=center
          // y 方向面（DOWN）位于 y=prev，需要在 x 和 y 两个方向上插值
          // 应该是：本地DOWN面，左边DOWN面，上边单元的DOWN面，左上单元的DOWN面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 本地DOWN面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_z;
          nCol++;

          col[nCol].i = ex - 1;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 左边DOWN面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_z;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 上边单元的DOWN面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_z;
          nCol++;

          col[nCol].i = ex - 1;  col[nCol].j = ey + 1;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 左上单元的DOWN面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_z;
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== y 方向面 (DOWN) 的对流项 =====
          // (ω₂ × u₂)_y = ω₂ᶻ u₂ˣ - ω₂ˣ u₂ᶻ
          row.loc = DOWN;  // y方向面
          nCol = 0;

          // ω₂ᶻ：在 z 方向面上，需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地BACK面，前方BACK面，下方BACK面，前下BACK面
          PetscScalar omega2_z_y = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_z] +      // 本地BACK面
                                            arrOmega2[ez+1][ey][ex][slot_omega2_z] +      // 前方BACK面
                                            arrOmega2[ez][ey-1][ex][slot_omega2_z] +    // 下方BACK面
                                            arrOmega2[ez+1][ey-1][ex][slot_omega2_z]);   // 前下BACK面

          // ω₂ˣ：在 x 方向面上，需要插值到 y 方向面的位置
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 y 两个方向上插值
          // 应该是：本地LEFT面，下边LEFT面，右边LEFT面，右下LEFT面
          PetscScalar omega2_x = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_x] +      // 本地LEFT面
                                         arrOmega2[ez][ey-1][ex][slot_omega2_x] +    // 下边LEFT面
                                         arrOmega2[ez][ey][ex+1][slot_omega2_x] +    // 右边LEFT面
                                         arrOmega2[ez][ey-1][ex+1][slot_omega2_x]);   // 右下LEFT面

          // 第一项：+ω₂ᶻ u₂ˣ
          // u₂ˣ 需要插值到 y 方向面的位置，使用四个 x 方向面
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 y 两个方向上插值
          // 应该是：本地LEFT面，下边LEFT面，右边LEFT面，右下LEFT面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 本地LEFT面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_z_y;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey - 1;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 下边LEFT面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_z_y;
          nCol++;

          col[nCol].i = ex + 1;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 右边LEFT面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_z_y;
          nCol++;

          col[nCol].i = ex + 1;  col[nCol].j = ey - 1;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 右下LEFT面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_z_y;
          nCol++;

          // 第二项：-ω₂ˣ u₂ᶻ
          // u₂ᶻ 需要插值到 y 方向面的位置，使用四个 z 方向面
          // y 方向面（DOWN）位于 y=prev, x=center, z=center
          // z 方向面（BACK）位于 z=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地BACK面，前方BACK面，下方BACK面，前下BACK面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 本地BACK面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_x;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK;  // 前方BACK面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_x;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey - 1;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 下方BACK面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_x;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey - 1;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK;  // 前下BACK面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_x;
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== z 方向面 (BACK) 的对流项 =====
          // (ω₂ × u₂)_z = ω₂ˣ u₂ʸ - ω₂ʸ u₂ˣ
          row.loc = BACK;  // z方向面
          nCol = 0;

          // ω₂ˣ：在 x 方向面上，需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 z 两个方向上插值
          // 应该是：本地LEFT面，后方LEFT面，右边LEFT面，右后LEFT面
          PetscScalar omega2_x_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_x] +      // 本地LEFT面
                                            arrOmega2[ez-1][ey][ex][slot_omega2_x] +    // 后方LEFT面
                                            arrOmega2[ez][ey][ex+1][slot_omega2_x] +      // 右边LEFT面
                                            arrOmega2[ez-1][ey][ex+1][slot_omega2_x]);   // 右后LEFT面

          // ω₂ʸ：在 y 方向面上，需要插值到 z 方向面的位置
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // y 方向面（DOWN）位于 y=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地DOWN面，上方DOWN面，后方DOWN面，后上DOWN面
          PetscScalar omega2_y_z = 0.25 * (arrOmega2[ez][ey][ex][slot_omega2_y] +      // 本地DOWN面
                                            arrOmega2[ez][ey+1][ex][slot_omega2_y] +    // 上方DOWN面
                                            arrOmega2[ez-1][ey][ex][slot_omega2_y] +      // 后方DOWN面
                                            arrOmega2[ez-1][ey+1][ex][slot_omega2_y]);   // 后上DOWN面

          // 第一项：+ω₂ˣ u₂ʸ
          // u₂ʸ 需要插值到 z 方向面的位置，使用四个 y 方向面
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // y 方向面（DOWN）位于 y=prev，需要在 y 和 z 两个方向上插值
          // 应该是：本地DOWN面，上方DOWN面，后方DOWN面，后上DOWN面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 本地DOWN面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_x_z;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 上方DOWN面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_x_z;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez - 1;
          col[nCol].loc = DOWN;  // 后方DOWN面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_x_z;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1;  col[nCol].k = ez - 1;
          col[nCol].loc = DOWN;  // 后上DOWN面
          col[nCol].c = 0;
          val[nCol] = coeff * omega2_x_z;
          nCol++;

          // 第二项：-ω₂ʸ u₂ˣ
          // u₂ˣ 需要插值到 z 方向面的位置，使用四个 x 方向面
          // z 方向面（BACK）位于 z=prev, x=center, y=center
          // x 方向面（LEFT）位于 x=prev，需要在 x 和 z 两个方向上插值
          // 应该是：本地LEFT面，后方LEFT面，右边LEFT面，右后LEFT面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 本地LEFT面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_y_z;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez - 1;
          col[nCol].loc = LEFT;  // 后方LEFT面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_y_z;
          nCol++;

          col[nCol].i = ex + 1;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 右边LEFT面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_y_z;
          nCol++;

          col[nCol].i = ex + 1;  col[nCol].j = ey;  col[nCol].k = ez - 1;
          col[nCol].loc = LEFT;  // 右后LEFT面
          col[nCol].c = 0;
          val[nCol] = -coeff * omega2_y_z;
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));
        }
      }
    }

    PetscCall(DMStagVecRestoreArrayRead(dmSol_1, localOmega2, &arrOmega2));
    PetscCall(DMRestoreLocalVector(dmSol_1, &localOmega2));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装 omega1 的旋度项矩阵：∇ × ω₁
// 这是从1-形式（棱）到2-形式（面）的耦合项
// 对应方程5.1中的 (1/Re) ∇_h × ω_1^{k}
// 注意：这是从棱到面，与 omega2_curl_matrix 相反（从面到棱）
PetscErrorCode DUAL_MAC::assemble_omega1_curl_matrix(PetscReal Re, DM dmSol_2, Mat A)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);
    
    PetscScalar coeff = 1.0 / Re;  // 1/Re 系数

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向面 (LEFT) 的旋度项 =====
          // (∇ × ω₁)_x = ∂ω₁ᶻ/∂y - ∂ω₁ʸ/∂z
          // 行：x方向面（u₂ˣ 的方程）
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = LEFT;  // x方向面
          row.c = 0;        // u₂ 在 dof2 的第0个分量

          // 列：涉及4条棱的 ω₁（2项，每项2条棱）
          DMStagStencil col[4];
          PetscScalar val[4];
          PetscInt nCol = 0;

          // 第一项：∂ω₁ᶻ/∂y ≈ (ω₁ᶻ|_{y=next} - ω₁ᶻ|_{y=prev}) / hy
          // ω₁ᶻ 在 z 方向棱上（DOWN_LEFT）
          // LEFT 面位于 y=center，使用向前差分
          // 使用本地的 z 方向棱和上方单元的 z 方向棱
          col[nCol].i = ex;  col[nCol].j = ey;     col[nCol].k = ez;
          col[nCol].loc = DOWN_LEFT;  // 本地的 z 方向棱（y=prev）
          col[nCol].c = 0;       // ω₁ 在 dof1 的第0个分量
          val[nCol] = -coeff / hy;       // 负号因为 (ω₁ᶻ|_{y=next} - ω₁ᶻ|_{y=prev})
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1; col[nCol].k = ez;
          col[nCol].loc = DOWN_LEFT;  // 上方单元的 z 方向棱（y=next）
          col[nCol].c = 0;
          val[nCol] = coeff / hy;        // 正号因为 (ω₁ᶻ|_{y=next} - ω₁ᶻ|_{y=prev})
          nCol++;

          // 第二项：-∂ω₁ʸ/∂z ≈ -(ω₁ʸ|_{z=next} - ω₁ʸ|_{z=prev}) / hz
          // ω₁ʸ 在 y 方向棱上（BACK_LEFT）
          // LEFT 面位于 z=center，使用向前差分
          // 使用本地的 y 方向棱和前方单元的 y 方向棱
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK_LEFT;  // 本地的 y 方向棱（z=prev）
          col[nCol].c = 0;
          val[nCol] = coeff / hz;        // 正号来自 -(ω₁ʸ|_{z=next} - ω₁ʸ|_{z=prev}) = (ω₁ʸ|_{z=prev} - ω₁ʸ|_{z=next}) / hz
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK_LEFT;  // 前方单元的 y 方向棱（z=next）
          col[nCol].c = 0;
          val[nCol] = -coeff / hz;       // 负号来自 -(ω₁ʸ|_{z=next} - ω₁ʸ|_{z=prev}) = (ω₁ʸ|_{z=prev} - ω₁ʸ|_{z=next}) / hz
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== y 方向面 (DOWN) 的旋度项 =====
          // (∇ × ω₁)_y = ∂ω₁ˣ/∂z - ∂ω₁ᶻ/∂x
          row.loc = DOWN;  // y方向面
          nCol = 0;

          // 第一项：∂ω₁ˣ/∂z ≈ (ω₁ˣ|_{z=next} - ω₁ˣ|_{z=prev}) / hz
          // ω₁ˣ 在 x 方向棱上（BACK_DOWN）
          // DOWN 面位于 z=prev，使用向前差分
          // 使用本地的 x 方向棱和前方单元的 x 方向棱
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK_DOWN;  // 本地的 x 方向棱（z=prev）
          col[nCol].c = 0;
          val[nCol] = -coeff / hz;       // 负号因为 (ω₁ˣ|_{z=next} - ω₁ˣ|_{z=prev})
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK_DOWN;  // 前方单元的 x 方向棱（z=next）
          col[nCol].c = 0;
          val[nCol] = coeff / hz;        // 正号因为 (ω₁ˣ|_{z=next} - ω₁ˣ|_{z=prev})
          nCol++;

          // 第二项：-∂ω₁ᶻ/∂x ≈ -(ω₁ᶻ|_{x=next} - ω₁ᶻ|_{x=prev}) / hx
          // ω₁ᶻ 在 z 方向棱上（DOWN_LEFT）
          // DOWN 面位于 y=prev，在 x 方向上跨越整个单元，使用向前差分
          // 使用本地的 z 方向棱和右边单元的 z 方向棱
          col[nCol].i = ex;     col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN_LEFT;  // 本地的 z 方向棱（x=prev）
          col[nCol].c = 0;
          val[nCol] = coeff / hx;        // 正号来自 -(ω₁ᶻ|_{x=next} - ω₁ᶻ|_{x=prev}) = (ω₁ᶻ|_{x=prev} - ω₁ᶻ|_{x=next}) / hx
          nCol++;

          col[nCol].i = ex + 1; col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN_LEFT;  // 右边单元的 z 方向棱（x=next）
          col[nCol].c = 0;
          val[nCol] = -coeff / hx;       // 负号来自 -(ω₁ᶻ|_{x=next} - ω₁ᶻ|_{x=prev}) = (ω₁ᶻ|_{x=prev} - ω₁ᶻ|_{x=next}) / hx
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== z 方向面 (BACK) 的旋度项 =====
          // (∇ × ω₁)_z = ∂ω₁ʸ/∂x - ∂ω₁ˣ/∂y
          row.loc = BACK;  // z方向面
          nCol = 0;

          // 第一项：∂ω₁ʸ/∂x ≈ (ω₁ʸ|_{x=next} - ω₁ʸ|_{x=prev}) / hx
          // ω₁ʸ 在 y 方向棱上（BACK_LEFT）
          // BACK 面位于 z=prev，在 x 方向上跨越整个单元，使用向前差分
          // 使用本地的 y 方向棱和右边单元的 y 方向棱
          col[nCol].i = ex;     col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK_LEFT;  // 本地的 y 方向棱（x=prev）
          col[nCol].c = 0;
          val[nCol] = -coeff / hx;       // 负号因为 (ω₁ʸ|_{x=next} - ω₁ʸ|_{x=prev})
          nCol++;

          col[nCol].i = ex + 1; col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK_LEFT;  // 右边单元的 y 方向棱（x=next）
          col[nCol].c = 0;
          val[nCol] = coeff / hx;        // 正号因为 (ω₁ʸ|_{x=next} - ω₁ʸ|_{x=prev})
          nCol++;

          // 第二项：-∂ω₁ˣ/∂y ≈ -(ω₁ˣ|_{y=next} - ω₁ˣ|_{y=prev}) / hy
          // ω₁ˣ 在 x 方向棱上（BACK_DOWN）
          // BACK 面位于 y=center，使用向前差分
          // 使用本地的 x 方向棱和上方单元的 x 方向棱
          col[nCol].i = ex;  col[nCol].j = ey;     col[nCol].k = ez;
          col[nCol].loc = BACK_DOWN;  // 本地的 x 方向棱（y=prev）
          col[nCol].c = 0;
          val[nCol] = -coeff / hy;       // 负号来自 -(ω₁ˣ|_{y=next} - ω₁ˣ|_{y=prev})
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1; col[nCol].k = ez;
          col[nCol].loc = BACK_DOWN;  // 上方单元的 x 方向棱（y=next）
          col[nCol].c = 0;
          val[nCol] = coeff / hy;        // 正号来自 -(ω₁ˣ|_{y=next} - ω₁ˣ|_{y=prev})
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));
        }
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装 P₃ 压力梯度矩阵：∇P₃
// 这是从3-形式（单元中心）到2-形式（面）的耦合项
// 对应方程5.1中的 ∇_h P₃^{h,k-1/2}
PetscErrorCode DUAL_MAC::assemble_p3_gradient_matrix(DM dmSol_2, Mat A)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向面 (LEFT) 的压力梯度项 =====
          // ∂P₃/∂x ≈ (P₃|_{RIGHT} - P₃|_{LEFT}) / hx
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = LEFT;  // x方向面
          row.c = 0;        // u₂ 在 dof2 的第0个分量

          DMStagStencil col[2];
          PetscScalar val[2];

          // 左单元中心（x=prev）
          // 本地LEFT面位于 x=prev，应使用本地单元中心(ex,...)和左边单元中心(ex-1,...)
          col[0].i = ex - 1;  col[0].j = ey;  col[0].k = ez;
          col[0].loc = ELEMENT;
          col[0].c = 0;                  // P₃ 在 dof3 的第0个分量
          val[0] = -1.0 / hx;

          // 右单元中心（x=next）
          col[1].i = ex;  col[1].j = ey;  col[1].k = ez;
          col[1].loc = ELEMENT;
          col[1].c = 0;
          val[1] = 1.0 / hx;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 2, col, val, ADD_VALUES));

          // ===== y 方向面 (DOWN) 的压力梯度项 =====
          // ∂P₃/∂y ≈ (P₃|_{y=center} - P₃|_{y=prev}) / hy
          // DOWN 面位于 y=prev，应使用本地单元中心(ex,ey,ez)和下方单元中心(ex,ey-1,ez)
          row.loc = DOWN;  // y方向面
          
          col[0].i = ex;  col[0].j = ey;  col[0].k = ez;
          col[0].loc = ELEMENT;  // 本地ELEMENT（y=center）
          col[0].c = 0;
          val[0] = 1.0 / hy;  // 正号因为 (P₃|_{y=center} - P₃|_{y=prev})

          col[1].i = ex;  col[1].j = ey - 1;  col[1].k = ez;
          col[1].loc = ELEMENT;  // 下方单元的ELEMENT（y=prev）
          col[1].c = 0;
          val[1] = -1.0 / hy;  // 负号因为 (P₃|_{y=center} - P₃|_{y=prev})

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 2, col, val, ADD_VALUES));

          // ===== z 方向面 (BACK) 的压力梯度项 =====
          // ∂P₃/∂z ≈ (P₃|_{z=center} - P₃|_{z=prev}) / hz
          // BACK 面位于 z=prev，应使用本地单元中心(ex,ey,ez)和后方单元中心(ex,ey,ez-1)
          row.loc = BACK;  // z方向面
          
          col[0].i = ex;  col[0].j = ey;  col[0].k = ez;
          col[0].loc = ELEMENT;  // 本地ELEMENT（z=center）
          col[0].c = 0;
          val[0] = 1.0 / hz;  // 正号因为 (P₃|_{z=center} - P₃|_{z=prev})

          col[1].i = ex;  col[1].j = ey;  col[1].k = ez - 1;
          col[1].loc = ELEMENT;  // 后方单元的ELEMENT（z=prev）
          col[1].c = 0;
          val[1] = -1.0 / hz;  // 负号因为 (P₃|_{z=center} - P₃|_{z=prev})

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 2, col, val, ADD_VALUES));
        }
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装 u2-omega1耦合矩阵：∇×u₂ - ω₁ = 0
// 这是从2-形式（面）到1-形式（棱）的耦合项
// 对应方程 ∇×u₂ = ω₁
// 行：棱（1-形式，ω₁ 的位置）
// 列：面（2-形式，u₂ 的位置）
PetscErrorCode DUAL_MAC::assemble_u2_omega1_coupling_matrix(DM dmSol_2, Mat A)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向棱 (BACK_DOWN) 的旋度方程 =====
          // (∇×u₂)ₓ = ∂u₂ᶻ/∂y - ∂u₂ʸ/∂z = ω₁ₓ
          // 行：x方向棱（ω₁ˣ 的方程）
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = BACK_DOWN;  // x方向棱
          row.c = 0;             // ω₁ 在 dof1 的第0个分量

          // 列：涉及4个面的 u₂（2项，每项2个面）
          DMStagStencil col[4];
          PetscScalar val[4];
          PetscInt nCol = 0;

          // 第一项：∂u₂ᶻ/∂y ≈ (u₂ᶻ|_{y=center} - u₂ᶻ|_{y=prev}) / hy
          // 对本地BACK_DOWN棱来说，应使用本地BACK面和下方单元的BACK面
          // u₂ᶻ 在 z 方向面上（BACK）
          col[nCol].i = ex;  col[nCol].j = ey;     col[nCol].k = ez;
          col[nCol].loc = BACK;  // 本地的 z 方向面（y=center）
          col[nCol].c = 0;        // u₂ 在 dof2 的第0个分量
          val[nCol] = 1.0 / hy;   // 正号 (u₂ᶻ|_{y=center} - u₂ᶻ|_{y=prev}) / hy
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey - 1; col[nCol].k = ez;
          col[nCol].loc = BACK;  // 下方单元的 z 方向面（y=prev）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hy;
          nCol++;

          // 第二项：-∂u₂ʸ/∂z ≈ -(u₂ʸ|_{z=center} - u₂ʸ|_{z=prev}) / hz
          // u₂ʸ 在 y 方向面上（DOWN）
          // 对本地BACK_DOWN棱的这一项，应使用本地DOWN面和后方单元的DOWN面：
          // 即 -(u₂ʸ|_{z=center} - u₂ʸ|_{z=prev}) / hz = (u₂ʸ|_{z=prev} - u₂ʸ|_{z=center}) / hz
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 本地的 y 方向面（z=center）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hz;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez - 1;
          col[nCol].loc = DOWN;  // 后方单元的 y 方向面（z=prev）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hz;
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== y 方向棱 (BACK_LEFT) 的旋度方程 =====
          // (∇×u₂)ᵧ = ∂u₂ˣ/∂z - ∂u₂ᶻ/∂x = ω₁ᵧ
          row.loc = BACK_LEFT;  // y方向棱
          nCol = 0;

          // 第一项：∂u₂ˣ/∂z ≈ (u₂ˣ|_{z=center} - u₂ˣ|_{z=prev}) / hz
          // 对本地BACK_LEFT棱来说，应使用本地LEFT面和后方单元的LEFT面
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 本地的 x 方向面（z=center）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hz;   // +u₂ˣ|_{z=center}/hz
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez - 1;
          col[nCol].loc = LEFT;  // 后方单元的 x 方向面（z=prev）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hz;  // -u₂ˣ|_{z=prev}/hz
          nCol++;

          // 第二项：-∂u₂ᶻ/∂x ≈ -(u₂ᶻ|_{x=center} - u₂ᶻ|_{x=prev}) / hx
          // 对本地BACK_LEFT棱来说，应使用本地BACK面和左边单元的BACK面
          col[nCol].i = ex;     col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 本地 z 方向面（x=center）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hx;   // -u₂ᶻ|_{x=center}/hx
          nCol++;

          col[nCol].i = ex - 1; col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // 左边单元的 z 方向面（x=prev）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hx;  // +u₂ᶻ|_{x=prev}/hx
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== z 方向棱 (DOWN_LEFT) 的旋度方程 =====
          // (∇×u₂)ᵧ = ∂u₂ʸ/∂x - ∂u₂ˣ/∂y = ω₁ᵧ
          row.loc = DOWN_LEFT;  // z方向棱
          nCol = 0;

          // 第一项：∂u₂ʸ/∂x ≈ (u₂ʸ|_{x=center} - u₂ʸ|_{x=prev}) / hx
          // 对本地DOWN_LEFT棱来说，应使用本地DOWN面和左边单元的DOWN面
          col[nCol].i = ex;     col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 本地的 y 方向面（x=center）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hx;   // +u₂ʸ|_{x=center}/hx
          nCol++;

          col[nCol].i = ex - 1; col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 左边单元的 y 方向面（x=prev）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hx;  // -u₂ʸ|_{x=prev}/hx
          nCol++;

          // 第二项：-∂u₂ˣ/∂y ≈ -(u₂ˣ|_{y=center} - u₂ˣ|_{y=prev}) / hy
          // 对本地DOWN_LEFT棱来说，应使用本地LEFT面和下方单元的LEFT面
          col[nCol].i = ex;  col[nCol].j = ey;     col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 本地 x 方向面（y=center）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hy;   // -u₂ˣ|_{y=center}/hy
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey - 1; col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 下方单元的 x 方向面（y=prev）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hy;  // +u₂ˣ|_{y=prev}/hy
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));

          // ===== 添加 -I 项（对 ω₁）=====
          // 对于每个棱，添加 -1 的对角项
          row.loc = BACK_DOWN;
          row.c = 0;
          PetscScalar minus_one = -1.0;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row, &minus_one, ADD_VALUES));

          row.loc = BACK_LEFT;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row, &minus_one, ADD_VALUES));

          row.loc = DOWN_LEFT;
          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, 1, &row, &minus_one, ADD_VALUES));
        }
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装u2散度为0矩阵：∇·u₂ = 0
// 这是从2-形式（面）到3-形式（单元中心）的耦合项
// 对应连续性方程 ∇·u₂ = 0
PetscErrorCode DUAL_MAC::assemble_u2_divergence_matrix(DM dmSol_2, Mat A)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));
    PetscReal hx, hy, hz;
    hx = (this->xmax - this->xmin) / static_cast<PetscReal>(this->Nx);
    hy = (this->ymax - this->ymin) / static_cast<PetscReal>(this->Ny);
    hz = (this->zmax - this->zmin) / static_cast<PetscReal>(this->Nz);

    // 遍历所有单元
    // 每个单元的单元中心对应一个散度方程
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== 单元中心 (ELEMENT) 的散度方程 =====
          // ∇·u₂ = (∂u₂ˣ/∂x + ∂u₂ʸ/∂y + ∂u₂ᶻ/∂z) = 0
          // 行：单元中心（P₃ 的方程）
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = ELEMENT;  // 单元中心
          row.c = 0;           // P₃ 在 dof3 的第0个分量

          // 启用 pinPressure 时，用单位行替换一个压力方程：p3(0,0,0)=0
          // 这样导出的 2-form 矩阵本身就是满秩，不再依赖线性求解阶段再做约束。
          if (this->pinPressure && ex == 0 && ey == 0 && ez == 0) {
              DMStagStencil pcol = row;
              PetscScalar one = 1.0;
              PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                        1, &row, 1, &pcol, &one, ADD_VALUES));
              continue;
          }

          // 列：涉及6个面的 u₂（3个方向，每个方向2个面）
          DMStagStencil col[6];
          PetscScalar val[6];
          PetscInt nCol = 0;

          // ===== x 方向分量：∂u₂ˣ/∂x ≈ (u₂ˣ|_{x=next} - u₂ˣ|_{x=prev}) / hx =====
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // x方向面（x=prev）
          col[nCol].c = 0;        // u₂ 在 dof2 的第0个分量
          val[nCol] = -1.0 / hx;
          nCol++;

          col[nCol].i = ex + 1;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = LEFT;  // 右邻居的 x 方向面（x=next）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hx;
          nCol++;

          // ===== y 方向分量：∂u₂ʸ/∂y ≈ (u₂ʸ|_{y=next} - u₂ʸ|_{y=prev}) / hy =====
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // y方向面（y=prev）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hy;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey + 1;  col[nCol].k = ez;
          col[nCol].loc = DOWN;  // 上邻居的 y 方向面（y=next）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hy;
          nCol++;

          // ===== z 方向分量：∂u₂ᶻ/∂z ≈ (u₂ᶻ|_{z=next} - u₂ᶻ|_{z=prev}) / hz =====
          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez;
          col[nCol].loc = BACK;  // z方向面（z=prev）
          col[nCol].c = 0;
          val[nCol] = -1.0 / hz;
          nCol++;

          col[nCol].i = ex;  col[nCol].j = ey;  col[nCol].k = ez + 1;
          col[nCol].loc = BACK;  // 前邻居的 z 方向面（z=next）
          col[nCol].c = 0;
          val[nCol] = 1.0 / hz;
          nCol++;

          PetscCall(DMStagMatSetValuesStencil(dmSol_2, A,
                    1, &row, nCol, col, val, ADD_VALUES));
        }
      }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 组装外力项向量
// 在每一个面上取对应点和方向的外力值
PetscErrorCode DUAL_MAC::assemble_force2_vector(DM dmSol_2, Vec rhs, ExternalForce externalForce, PetscReal time)
{
    PetscFunctionBeginUser;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dmSol_2, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));

    // 获取 product 坐标数组
    PetscScalar **cArrX, **cArrY, **cArrZ;
    PetscCall(DMStagGetProductCoordinateArraysRead(dmSol_2, &cArrX, &cArrY, &cArrZ));
    
    // 获取坐标的 slot 索引
    PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, ELEMENT, &icx_center));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, LEFT, &icx_prev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, ELEMENT, &icy_center));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, LEFT, &icy_prev));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, ELEMENT, &icz_center));
    PetscCall(DMStagGetProductCoordinateLocationSlot(dmSol_2, LEFT, &icz_prev));

    // 遍历所有单元
    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
      for (PetscInt ey = starty; ey < starty + ny; ++ey) {
        for (PetscInt ex = startx; ex < startx + nx; ++ex) {

          // ===== x 方向面 (LEFT) 的外力项 =====
          // x 方向面位于 x=prev，y 和 z 坐标在单元中心
          DMStagStencil row;
          row.i = ex;  row.j = ey;  row.k = ez;
          row.loc = LEFT;  // x方向面
          row.c = 0;        // u₂ 在 dof2 的第0个分量

          PetscScalar x = cArrX[ex][icx_prev];
          PetscScalar y = cArrY[ey][icy_center];
          PetscScalar z = cArrZ[ez][icz_center];

          PetscScalar force_val = externalForce.fx(x, y, z, time);
          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &force_val, ADD_VALUES));

          // ===== y 方向面 (DOWN) 的外力项 =====
          // y 方向面位于 y=prev，x 和 z 坐标在单元中心
          row.loc = DOWN;  // y方向面

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_prev];
          z = cArrZ[ez][icz_center];

          force_val = externalForce.fy(x, y, z, time);
          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &force_val, ADD_VALUES));

          // ===== z 方向面 (BACK) 的外力项 =====
          // z 方向面位于 z=prev，x 和 y 坐标在单元中心
          row.loc = BACK;  // z方向面

          x = cArrX[ex][icx_center];
          y = cArrY[ey][icy_center];
          z = cArrZ[ez][icz_prev];

          force_val = externalForce.fz(x, y, z, time);
          PetscCall(DMStagVecSetValuesStencil(dmSol_2, rhs, 1, &row, &force_val, ADD_VALUES));
        }
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(dmSol_2, &cArrX, &cArrY, &cArrZ));

    PetscFunctionReturn(PETSC_SUCCESS);
}

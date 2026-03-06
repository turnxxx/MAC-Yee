#include <petsc.h>
#include "../include/DUAL_MAC.h"
#include <cmath>
#include <cstring>
#include <vector>
// 线性求解器封装
// 为了方便在不同地方复用线性代数设置，这里提供一个简单的求解接口
// 注意：具体求解器类型、预条件子等可以通过命令行参数设置

#if DUAL_MAC_DEBUG
static PetscErrorCode debug_check_vec_finite_ls(Vec v, const char *tag)
{
    PetscFunctionBeginUser;
    if (!v) PetscFunctionReturn(PETSC_SUCCESS);

    PetscInt lo = 0;
    PetscCall(VecGetOwnershipRange(v, &lo, NULL));
    PetscInt nLocal = 0;
    PetscCall(VecGetLocalSize(v, &nLocal));

    const PetscScalar *arr = NULL;
    PetscCall(VecGetArrayRead(v, &arr));

    PetscInt firstBad = -1;
    PetscScalar badVal = 0.0;
    for (PetscInt i = 0; i < nLocal; ++i) {
#if defined(PETSC_USE_COMPLEX)
        const PetscBool ok = (PetscBool)(std::isfinite((double)PetscRealPart(arr[i])) && std::isfinite((double)PetscImaginaryPart(arr[i])));
#else
        const PetscBool ok = (PetscBool)(std::isfinite((double)PetscRealPart(arr[i])));
#endif
        if (!ok) {
            firstBad = lo + i;
            badVal = arr[i];
            break;
        }
    }
    PetscCall(VecRestoreArrayRead(v, &arr));

    if (firstBad >= 0) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[DEBUG][LS finite] %s first invalid idx=%" PetscInt_FMT ", value=(%.16e, %.16e)\n",
                              tag ? tag : "vec", firstBad, (double)PetscRealPart(badVal), (double)PetscImaginaryPart(badVal)));
#else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[DEBUG][LS finite] %s first invalid idx=%" PetscInt_FMT ", value=%.16e\n",
                              tag ? tag : "vec", firstBad, (double)PetscRealPart(badVal)));
#endif
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[DEBUG][LS finite] %s all finite\n", tag ? tag : "vec"));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// 为 PCFieldSplit 构造两组索引：
// - u 场：速度 + 涡量
// - p 场：压力
static PetscErrorCode build_up_fieldsplit_is(DM dm, IS *isU, IS *isP)
{
    PetscFunctionBeginUser;
    PetscInt dof0, dof1, dof2, dof3;
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
    PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL, NULL, NULL));

    const PetscBool isTwoLike = (dof3 > 0 && dof0 == 0) ? PETSC_TRUE : PETSC_FALSE;

    std::vector<DMStagStencil> stU;
    std::vector<DMStagStencil> stP;
    stU.reserve((size_t)nx * (size_t)ny * (size_t)nz * 6);
    stP.reserve((size_t)nx * (size_t)ny * (size_t)nz);

    for (PetscInt ez = startz; ez < startz + nz; ++ez) {
        for (PetscInt ey = starty; ey < starty + ny; ++ey) {
            for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                DMStagStencil s;
                s.i = ex; s.j = ey; s.k = ez; s.c = 0;

                if (isTwoLike) {
                    // two-form: u2 在面(dof2)，omega1 在棱(dof1)，p 在单元中心(dof3)
                    if (dof2 > 0) {
                        s.loc = LEFT; stU.push_back(s);
                        s.loc = DOWN; stU.push_back(s);
                        s.loc = BACK; stU.push_back(s);
                    }
                    if (dof1 > 0) {
                        s.loc = BACK_DOWN; stU.push_back(s);
                        s.loc = BACK_LEFT; stU.push_back(s);
                        s.loc = DOWN_LEFT; stU.push_back(s);
                    }
                    if (dof3 > 0) {
                        s.loc = ELEMENT; stP.push_back(s);
                    }
                } else {
                    // one-form/half: u1 在棱(dof1)，omega2 在面(dof2, half 步可能无)，p 在顶点(dof0)
                    if (dof1 > 0) {
                        s.loc = BACK_DOWN; stU.push_back(s);
                        s.loc = BACK_LEFT; stU.push_back(s);
                        s.loc = DOWN_LEFT; stU.push_back(s);
                    }
                    if (dof2 > 0) {
                        s.loc = LEFT; stU.push_back(s);
                        s.loc = DOWN; stU.push_back(s);
                        s.loc = BACK; stU.push_back(s);
                    }
                    if (dof0 > 0) {
                        s.loc = BACK_DOWN_LEFT; stP.push_back(s);
                    } else if (dof3 > 0) {
                        s.loc = ELEMENT; stP.push_back(s);
                    }
                }
            }
        }
    }

    *isU = NULL;
    *isP = NULL;
    if (!stU.empty()) PetscCall(DMStagCreateISFromStencils(dm, (PetscInt)stU.size(), stU.data(), isU));
    if (!stP.empty()) PetscCall(DMStagCreateISFromStencils(dm, (PetscInt)stP.size(), stP.data(), isP));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 对 FieldSplit 的子块做代码级默认配置，避免命令行前缀未命中时退化到不可控默认值。
static PetscErrorCode configure_fieldsplit_subksp_defaults(PC pc)
{
    PetscFunctionBeginUser;
    PetscBool isFieldSplit = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &isFieldSplit));
    if (!isFieldSplit) PetscFunctionReturn(PETSC_SUCCESS);

    PetscInt nSplit = 0;
    KSP *subksp = NULL;
    PetscCall(PCSetUp(pc));
    PetscCall(PCFieldSplitGetSubKSP(pc, &nSplit, &subksp));

    for (PetscInt i = 0; i < nSplit; ++i) {
        const char *prefix = NULL;
        PetscCall(KSPGetOptionsPrefix(subksp[i], &prefix));

        PetscBool isPBlock = PETSC_FALSE, isUBlock = PETSC_FALSE;
        if (prefix) {
            if (std::strstr(prefix, "fieldsplit_p_")) isPBlock = PETSC_TRUE;
            if (std::strstr(prefix, "fieldsplit_u_")) isUBlock = PETSC_TRUE;
        }
        // 兜底：若前缀不可判别，按添加顺序将第0块视为u，第1块视为p
        if (!isPBlock && !isUBlock) {
            isUBlock = (i == 0) ? PETSC_TRUE : PETSC_FALSE;
            isPBlock = (i == 1) ? PETSC_TRUE : PETSC_FALSE;
        }

        PC subpc = NULL;
        PetscCall(KSPGetPC(subksp[i], &subpc));

        if (isUBlock) {
            PetscCall(KSPSetType(subksp[i], KSPFGMRES));
            PetscCall(KSPSetTolerances(subksp[i], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 200));
            PetscCall(PCSetType(subpc, PCASM));
            PetscCall(PCASMSetOverlap(subpc, 2));
        } else if (isPBlock) {
            PetscCall(KSPSetType(subksp[i], KSPPREONLY));
            PetscCall(PCSetType(subpc, PCGAMG));
        }

        // 允许用户通过命令行（若命中）覆盖上述默认值
        PetscCall(KSPSetFromOptions(subksp[i]));
    }

    PetscCall(PetscFree(subksp));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 在两场分裂框架下，向系统矩阵添加 gamma * grad(div(u)) 项。
// 实现方式：利用已有块结构，构造 G=A(u,p), D=A(p,u)，并将 gamma*(G*D) 加到 A(u,u) 子块。
static PetscErrorCode add_graddiv_term_to_matrix(Mat A, DM dm, PetscReal gamma)
{
    PetscFunctionBeginUser;
    if (gamma == 0.0) PetscFunctionReturn(PETSC_SUCCESS);

    IS isU = NULL, isP = NULL;
    Mat G = NULL, D = NULL, GD = NULL;
    const PetscInt *idxU = NULL;
    PetscInt nU = 0;

    PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
    if (!isU || !isP) {
        if (isU) PetscCall(ISDestroy(&isU));
        if (isP) PetscCall(ISDestroy(&isP));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(MatCreateSubMatrix(A, isU, isP, MAT_INITIAL_MATRIX, &G)); // grad block
    PetscCall(MatCreateSubMatrix(A, isP, isU, MAT_INITIAL_MATRIX, &D)); // div block
    PetscCall(MatMatMult(G, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GD)); // GD on u-u

    PetscCall(ISGetSize(isU, &nU));
    PetscCall(ISGetIndices(isU, &idxU));

    std::vector<PetscInt> gcols;
    std::vector<PetscScalar> gvals;
    for (PetscInt i = 0; i < nU; ++i) {
        PetscInt ncols = 0;
        const PetscInt *cols = NULL;
        const PetscScalar *vals = NULL;
        const PetscInt grow = idxU[i];

        PetscCall(MatGetRow(GD, i, &ncols, &cols, &vals));
        gcols.resize((size_t)ncols);
        gvals.resize((size_t)ncols);
        for (PetscInt j = 0; j < ncols; ++j) {
            gcols[(size_t)j] = idxU[cols[j]];
            gvals[(size_t)j] = ((PetscScalar)gamma) * vals[j];
        }
        if (ncols > 0) PetscCall(MatSetValues(A, 1, &grow, ncols, gcols.data(), gvals.data(), ADD_VALUES));
        PetscCall(MatRestoreRow(GD, i, &ncols, &cols, &vals));
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCall(ISRestoreIndices(isU, &idxU));
    PetscCall(MatDestroy(&GD));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&G));
    PetscCall(ISDestroy(&isU));
    PetscCall(ISDestroy(&isP));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 基本线性求解器：解 A x = rhs
// - A   : 已经组装完毕的 PETSc 矩阵
// - rhs : 右端项向量
// - sol : 解向量（输出）
// - dm  : DM 对象，用于访问压力分量并进行均值归零处理
// - optionsPrefix : KSP 命令行前缀（如 "one_"、"two_"），为空则使用无前缀
PetscErrorCode solve_linear_system_basic(Mat A, Vec rhs, Vec sol, DM dm, const char *optionsPrefix)
{
    PetscFunctionBeginUser;

    KSP ksp;
    PC  pc;
    PetscInt dof0, dof1, dof2, dof3;

    // 创建 KSP
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    // 预条件子：默认使用 FieldSplit，并显式构造 u(速度+涡量)/p(压力) 两个分块
    // 用户仍可通过命令行参数覆盖具体子求解器
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCFIELDSPLIT));
    // 关闭自动鞍点检测，避免覆盖手工指定的 U/P 分块索引集
    PetscCall(PCFieldSplitSetDetectSaddlePoint(pc, PETSC_FALSE));
    {
        IS isU = NULL, isP = NULL;
        PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
        // 使用小写 split 名称，命令行前缀对应为 fieldsplit_u / fieldsplit_p
        if (isU) PetscCall(PCFieldSplitSetIS(pc, "u", isU));
        if (isP) PetscCall(PCFieldSplitSetIS(pc, "p", isP));
        if (isU) PetscCall(ISDestroy(&isU));
        if (isP) PetscCall(ISDestroy(&isP));
    }

    if (optionsPrefix && optionsPrefix[0] != '\0') {
        PetscCall(KSPSetOptionsPrefix(ksp, optionsPrefix));
    }

    // 从命令行读取 KSP/PC 相关参数，方便调试不同求解器
    PetscCall(KSPSetFromOptions(ksp));

    // 对 fieldsplit 子块做显式默认配置，避免 prefix 解析失败时设置失效
    PetscCall(configure_fieldsplit_subksp_defaults(pc));

    // 注意：2-form 的 p3 钉住目前在矩阵组装阶段完成（assemble_u2_divergence_matrix）。
    // 这里不再对 dof3 系统重复施加 MatZeroRowsColumnsIS，避免二次改写系统。
    PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));

#if DUAL_MAC_DEBUG
    PetscCall(debug_check_vec_finite_ls(rhs, "rhs_before_solve"));
    PetscCall(VecZeroEntries(sol));
    PetscCall(debug_check_vec_finite_ls(sol, "sol_after_zero_before_solve"));
#endif

    // 求解线性系统
    PetscCall(KSPSolve(ksp, rhs, sol));

    // two_ 子问题残差诊断：
    // 1) 全局残差 ||Ax-b||/||b||
    // 2) div 约束块（ELEMENT 行）残差 L2/Linf
    const PetscBool isTwoSystem = (optionsPrefix && std::strncmp(optionsPrefix, "two_", 4) == 0) ? PETSC_TRUE : PETSC_FALSE;
    if (isTwoSystem) {
        Vec r = NULL, localR = NULL;
        PetscScalar ****arrR = NULL;
        PetscReal rhsNorm = 0.0, rNorm = 0.0, relNorm = 0.0;

        PetscCall(VecDuplicate(rhs, &r));
        PetscCall(MatMult(A, sol, r));
        PetscCall(VecAXPY(r, -1.0, rhs)); // r = A*sol - rhs
        PetscCall(VecNorm(rhs, NORM_2, &rhsNorm));
        PetscCall(VecNorm(r, NORM_2, &rNorm));
        relNorm = (rhsNorm > 0.0) ? (rNorm / rhsNorm) : rNorm;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[DEBUG][LS two] residual: ||Ax-b||_2=%.12e, ||b||_2=%.12e, rel=%.12e\n",
                              (double)rNorm, (double)rhsNorm, (double)relNorm));

        if (dof3 > 0) {
            PetscInt startx, starty, startz, nx, ny, nz, slotElem;
            PetscReal localSqElem = 0.0, localMaxElem = 0.0;
            PetscReal globalSqElem = 0.0, globalMaxElem = 0.0;

            PetscCall(DMGetLocalVector(dm, &localR));
            PetscCall(DMGlobalToLocal(dm, r, INSERT_VALUES, localR));
            PetscCall(DMStagVecGetArrayRead(dm, localR, &arrR));
            PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL, NULL, NULL));
            PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &slotElem));

            for (PetscInt ez = startz; ez < startz + nz; ++ez) {
                for (PetscInt ey = starty; ey < starty + ny; ++ey) {
                    for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                        const PetscReal v = PetscAbsReal(PetscRealPart(arrR[ez][ey][ex][slotElem]));
                        localSqElem += v * v;
                        if (v > localMaxElem) localMaxElem = v;
                    }
                }
            }

            PetscCall(PMPI_Allreduce(&localSqElem, &globalSqElem, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
            PetscCall(PMPI_Allreduce(&localMaxElem, &globalMaxElem, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)dm)));

            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                                  "[DEBUG][LS two] ELEMENT(div-row) residual: L2=%.12e, Linf=%.12e\n",
                                  (double)PetscSqrtReal(globalSqElem), (double)globalMaxElem));

            PetscCall(DMStagVecRestoreArrayRead(dm, localR, &arrR));
            PetscCall(DMRestoreLocalVector(dm, &localR));
        }

        PetscCall(VecDestroy(&r));
    }

#if DUAL_MAC_DEBUG
    {
        KSPConvergedReason reason;
        PetscInt its = 0;
        PetscCall(KSPGetConvergedReason(ksp, &reason));
        PetscCall(KSPGetIterationNumber(ksp, &its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[DEBUG][LS] KSPSolve done: its=%" PetscInt_FMT ", reason=%d\n",
                              its, (int)reason));
    }
    PetscCall(debug_check_vec_finite_ls(sol, "sol_after_solve_before_pressure_mean"));
#endif

    // 销毁 KSP
    PetscCall(KSPDestroy(&ksp));

    // ===== 压力均值归零处理 =====
    // 这是处理压力不定性的常用方法：求解后减去压力的全局均值
    PetscInt startx, starty, startz, nx, ny, nz;
    PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz,
                               &nx, &ny, &nz, NULL, NULL, NULL));

    // 获取本地向量数组
    Vec localSol;
    PetscScalar ****arrSol;
    PetscCall(DMGetLocalVector(dm, &localSol));
    PetscCall(DMGlobalToLocal(dm, sol, INSERT_VALUES, localSol));
    PetscCall(DMStagVecGetArray(dm, localSol, &arrSol));

    PetscScalar local_sum = 0.0;
    PetscInt local_count = 0;
    PetscInt slot_pressure;

    if (dof0 > 0) {
        // P0 压力在顶点（BACK_DOWN_LEFT）
        PetscCall(DMStagGetLocationSlot(dm, BACK_DOWN_LEFT, 0, &slot_pressure));
        
        // 计算局部压力和
        // 周期边界下 dof0 的独立点数量与单元数一致，不能使用 +1 范围
        for (PetscInt ez = startz; ez < startz + nz; ++ez) {
            for (PetscInt ey = starty; ey < starty + ny; ++ey) {
                for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                    local_sum += arrSol[ez][ey][ex][slot_pressure];
                    local_count++;
                }
            }
        }
    } else if (dof3 > 0) {
        // P3 压力在单元中心（ELEMENT）
        PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &slot_pressure));
        
        // 计算局部压力和
        for (PetscInt ez = startz; ez < startz + nz; ++ez) {
            for (PetscInt ey = starty; ey < starty + ny; ++ey) {
                for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                    local_sum += arrSol[ez][ey][ex][slot_pressure];
                    local_count++;
                }
            }
        }
    }

    // 计算全局均值（使用 PETSc 的 MPI 包装函数）
    PetscScalar global_sum, global_mean;
    PetscInt global_count;
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);
    {
        PetscMPIInt ierr = MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_SCALAR, MPI_SUM, comm);
        if (ierr) SETERRQ(comm, PETSC_ERR_LIB, "MPI_Allreduce failed for sum");
    }
    {
        PetscMPIInt ierr = MPI_Allreduce(&local_count, &global_count, 1, MPIU_INT, MPI_SUM, comm);
        if (ierr) SETERRQ(comm, PETSC_ERR_LIB, "MPI_Allreduce failed for count");
    }
    
    if (global_count > 0) {
        global_mean = global_sum / static_cast<PetscScalar>(global_count);
    } else {
        global_mean = 0.0;
    }

    // 从所有压力值中减去均值
    /* if (dof0 > 0) {
        // P0 压力在顶点
        for (PetscInt ez = startz; ez < startz + nz; ++ez) {
            for (PetscInt ey = starty; ey < starty + ny; ++ey) {
                for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                    arrSol[ez][ey][ex][slot_pressure] -= global_mean;
                }
            }
        }
    } else if (dof3 > 0) {
        // P3 压力在单元中心
        for (PetscInt ez = startz; ez < startz + nz; ++ez) {
            for (PetscInt ey = starty; ey < starty + ny; ++ey) {
                for (PetscInt ex = startx; ex < startx + nx; ++ex) {
                    arrSol[ez][ey][ex][slot_pressure] -= global_mean;
                }
            }
        }
    } */

    // 写回全局向量
    PetscCall(DMLocalToGlobal(dm, localSol, INSERT_VALUES, sol));

#if DUAL_MAC_DEBUG
    PetscCall(debug_check_vec_finite_ls(sol, "sol_after_pressure_mean"));
#endif

    // 释放数组
    PetscCall(DMStagVecRestoreArray(dm, localSol, &arrSol));
    PetscCall(DMRestoreLocalVector(dm, &localSol));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// grad-div 增强版线性求解器：
// - 在不改动输入矩阵 A 的前提下，构造 A_eff = A + gamma * grad(div(u))
// - 然后复用两场分裂求解流程（与 solve_linear_system_basic 保持一致）
PetscErrorCode solve_linear_system_graddiv(Mat A, Vec rhs, Vec sol, DM dm, PetscReal gamma, const char *optionsPrefix)
{
    PetscFunctionBeginUser;
    Mat Aeff = NULL;

    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aeff));
    PetscCall(add_graddiv_term_to_matrix(Aeff, dm, gamma));

#if DUAL_MAC_DEBUG
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[DEBUG][LS graddiv] gamma=%.6e, operator added to u-u block\n",
                          (double)gamma));
#endif

    PetscCall(solve_linear_system_basic(Aeff, rhs, sol, dm, optionsPrefix));
    PetscCall(MatDestroy(&Aeff));
    PetscFunctionReturn(PETSC_SUCCESS);
}
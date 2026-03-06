#pragma once
// DUAL_MAC类：实现基于离散de Rham复形的NS方程求解器（使用DMStag有限差分）
// 求解六个变量：(u_1, omega_2, p_0) 和 (u_2, omega_1, p_3)
// 其中下标0,1,2,3分别对应离散de Rham复形下的0形式、1形式、2形式、3形式
// 在DMStag中：
//   0形式：自由度在顶点（vertex）
//   1形式：自由度在边（edge）
//   2形式：自由度在面（face，2D中对应边）
//   3形式：自由度在单元中心（element）
#include <petsc.h>
#include <petscdm.h>
#include <petscdmstag.h> // DMStag用于交错网格有限差分
#include <petscksp.h>
#include <petscsnes.h>
#include <bits/stdc++.h>
#include <mpi.h>
#include "ref_sol.h"  // 用于 ExternalForce

// 调试日志宏：
// - 默认关闭（DUAL_MAC_DEBUG=0）
// - 编译时可通过 -DDUAL_MAC_DEBUG=1 打开
#ifndef DUAL_MAC_DEBUG
#define DUAL_MAC_DEBUG 0
#endif

#if DUAL_MAC_DEBUG
#define DUAL_MAC_DEBUG_LOG(...) \
    do { PetscCall(PetscPrintf(PETSC_COMM_WORLD, __VA_ARGS__)); } while (0)
#else
#define DUAL_MAC_DEBUG_LOG(...) \
    do { } while (0)
#endif

// DMStag stencil location的简化命名
#define DOWN_LEFT DMSTAG_DOWN_LEFT
#define DOWN DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT DMSTAG_LEFT
#define ELEMENT DMSTAG_ELEMENT
#define RIGHT DMSTAG_RIGHT
#define UP_LEFT DMSTAG_UP_LEFT
#define UP DMSTAG_UP
#define UP_RIGHT DMSTAG_UP_RIGHT
#define BACK_DOWN_LEFT DMSTAG_BACK_DOWN_LEFT
#define BACK_DOWN DMSTAG_BACK_DOWN
#define BACK_DOWN_RIGHT DMSTAG_BACK_DOWN_RIGHT
#define BACK_LEFT DMSTAG_BACK_LEFT
#define BACK DMSTAG_BACK
#define BACK_RIGHT DMSTAG_BACK_RIGHT
#define BACK_UP_LEFT DMSTAG_BACK_UP_LEFT
#define BACK_UP DMSTAG_BACK_UP
#define BACK_UP_RIGHT DMSTAG_BACK_UP_RIGHT
#define FRONT_DOWN_LEFT DMSTAG_FRONT_DOWN_LEFT
#define FRONT_DOWN DMSTAG_FRONT_DOWN
#define FRONT_DOWN_RIGHT DMSTAG_FRONT_DOWN_RIGHT
#define FRONT_LEFT DMSTAG_FRONT_LEFT
#define FRONT DMSTAG_FRONT
#define FRONT_RIGHT DMSTAG_FRONT_RIGHT
#define FRONT_UP_LEFT DMSTAG_FRONT_UP_LEFT
#define FRONT_UP DMSTAG_FRONT_UP
#define FRONT_UP_RIGHT DMSTAG_FRONT_UP_RIGHT

class DUAL_MAC
{
public:
    // 构造函数
    DUAL_MAC();

    // 带参数的构造函数
    DUAL_MAC(PetscReal time, PetscReal nu, PetscInt NX, PetscInt NY, PetscInt NZ, PetscInt Nt,
             PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax, PetscReal zmin, PetscReal zmax,
             PetscBool pinPressure = PETSC_TRUE);

    // 析构函数
    ~DUAL_MAC();


    // 组装右端项
    PetscErrorCode assemble_rhs_vector(PetscReal time = 0.0);

    // 设置参考解
    PetscErrorCode set_reference_solution(RefSol refSol);

    // 设置边界条件
    PetscErrorCode setup_boundary_conditions(PetscReal time = 0.0);

    
    // 完整的求解流程：初始条件设置 -> 1/2时刻解 -> Nt次时间推进
    PetscErrorCode solve(RefSol refSol, ExternalForce externalForce);
    
    // 辅助函数：从组件组装完整解向量
    PetscErrorCode assemble_sol1_from_components(DM dmSol_1, Vec u1, Vec omega2, Vec sol1);
    PetscErrorCode assemble_sol2_from_components(DM dmSol_2, Vec u2, Vec omega1, Vec sol2);

    // 计算误差
    PetscErrorCode compute_error(PetscReal time = 0.0);

    // 输出结果
    PetscErrorCode output_solution(const char *filename);
    PetscErrorCode output_matrix(Mat &pA, const char *name);
    PetscErrorCode output_vector(Vec &pV, const char *name);

    // 清理资源
    PetscErrorCode destroy();

    // 获取解向量（用于外部访问）
    Vec get_solution() { return sol; }

    // 获取最近一次 compute_error() 缓存的各物理量误差
    PetscReal get_err_u1() const { return err_u1; }
    PetscReal get_err_omega2() const { return err_omega2; }
    PetscReal get_err_p0() const { return err_p0; }
    PetscReal get_err_u2() const { return err_u2; }
    PetscReal get_err_omega1() const { return err_omega1; }
    PetscReal get_err_p3() const { return err_p3; }


    // 获取各个变量的slot索引（用于访问sol中对应的数据）
    PetscInt get_slot_p0() { return slot_p0; }
    PetscInt get_slot_u1() { return slot_u1; }
    PetscInt get_slot_omega1() { return slot_omega1; }
    PetscInt get_slot_omega2() { return slot_omega2; }
    PetscInt get_slot_u2() { return slot_u2; }
    PetscInt get_slot_p3() { return slot_p3; }

    // 设置DM对象（周期边界）
    PetscErrorCode set_dm_periodic();

    // 打印DM信息（用于调试）
    // view_dm1: 是否打印 dmSol_1 的信息
    // view_dm2: 是否打印 dmSol_2 的信息
    // viewer: PETSc 查看器，如果为 NULL 则使用默认的 stdout 查看器
    PetscErrorCode view_dm(PetscBool view_dm1 = PETSC_TRUE, 
                           PetscBool view_dm2 = PETSC_TRUE,
                           PetscViewer viewer = NULL);

private:
    // 网格参数 
    PetscInt Nx, Ny, Nz;
    PetscInt Nt;
    PetscReal xmin, xmax, ymin, ymax, zmin, zmax;
    PetscReal dx, dy, dz;
    PetscReal time,dt;
    PetscReal nu;          // 动力粘度
    PetscBool pinPressure; // 是否固定压力点
        
    // DM对象：只创建一个组合DMStag，通过设置不同位置的dof数量来容纳所有6个变量
    // 在DMStag中：
    //   - dof0（顶点）：用于p_0（0形式）
    //   - dof1（边）：用于u_1和omega_1（1形式，需要2个dof）
    //   - dof2（面）：用于omega_2和u_2（2形式，需要2个dof）
    //   - dof3（单元中心，仅3D）：用于p_3（3形式）
    // 2D: dof0=1, dof1=2, dof2=2
    // 3D: dof0=1, dof1=2, dof2=2, dof3=1
    DM dmSol_1; // 1形式方程的dm
    DM dmSol_2;//2形式方程的dm

    // 各个变量在DM中的slot索引（用于访问不同dof）
    PetscInt slot_p0;     // p_0在dof0中的slot（总是0）
    PetscInt slot_u1;     // u_1在dof1中的slot（0）
    PetscInt slot_omega1; // omega_1在dof1中的slot（1）
    PetscInt slot_omega2; // omega_2在dof2中的slot（0）
    PetscInt slot_u2;     // u_2在dof2中的slot（1）
    PetscInt slot_p3;     // p_3在dof3中的slot（总是0，仅3D）

    // 矩阵和向量
    Mat A;   // 系统矩阵
    Vec sol; // 解向量（包含所有六个变量）
    Vec rhs; // 右端项向量

    // 各个变量的解向量（从组合向量sol中提取的子向量）
    // 注意：这些向量实际上指向sol的不同部分，不需要单独创建
    // 如果需要单独访问，可以通过DMStagVecGetArray来获取对应slot的数据
    // 这里保留这些成员变量是为了接口兼容性，但实际实现中可能不需要

    // 参考解（用于误差计算）
    Vec sol_ref;
    RefSol refSol_cached;     // 缓存参考解，供 compute_error() 使用
    Vec sol1_cached;          // 缓存最终 1-form 解：(u1, omega2, p0)
    Vec sol2_cached;          // 缓存最终 2-form 解：(u2, omega1, p3)
    PetscBool hasRefSol;      // 是否已设置参考解

    // 求解器
    KSP ksp;   // 线性求解器
    PC pc;     // 预条件子
    SNES snes; // 非线性求解器

    // 误差信息
    PetscReal err_abs, err_rel;
    PetscReal err_u1, err_omega2, err_p0;
    PetscReal err_u2, err_omega1, err_p3;

    // 辅助函数：de Rham复形操作（基于DMStag的有限差分实现）
    // 这些操作直接在组合向量sol的不同slot之间进行
    PetscErrorCode apply_exterior_derivative_0_to_1(Vec v0_slot, Vec v1_slot); // d_0: 0形式 -> 1形式（顶点 -> 边）
    PetscErrorCode apply_exterior_derivative_1_to_2(Vec v1_slot, Vec v2_slot); // d_1: 1形式 -> 2形式（边 -> 面）
    PetscErrorCode apply_exterior_derivative_2_to_3(Vec v2_slot, Vec v3_slot); // d_2: 2形式 -> 3形式（面 -> 单元中心，仅3D）
    PetscErrorCode apply_hodge_star_1(Vec v1_slot, Vec v2_slot);               // *: 1形式 -> 2形式（边 -> 面）
    PetscErrorCode apply_hodge_star_2(Vec v2_slot, Vec v1_slot);               // *: 2形式 -> 1形式（面 -> 边）

    // 辅助函数：获取坐标数组
    PetscErrorCode get_coordinate_arrays(PetscScalar ***cArrX, PetscScalar ***cArrY, PetscScalar ***cArrZ = NULL);

    // 辅助函数：获取stencil location的slot索引
    PetscErrorCode get_location_slot(DMStagStencilLocation loc, PetscInt dof_index, PetscInt *slot);

    // 辅助函数：初始化slot索引
    PetscErrorCode initialize_slot_indices();
    
    // 设置初始解
    PetscErrorCode set_up_initial_cond();
    PetscErrorCode setup_initial_solution(RefSol refSol, Vec u1_0, Vec u2_0, Vec omega1_0, Vec omega2_0);
    
    // 根据论文的 Starting procedure，使用显式欧拉法计算二分之一时刻的初值
    PetscErrorCode compute_half_solution(Vec u1_0, Vec omega1_0, Vec omega2_0, 
                                         Vec u1_half, Vec omega2_half, 
                                         ExternalForce externalForce);
    
    // 辅助函数：用于 compute_half_solution
    PetscErrorCode assemble_omega1_u1_conv_rhs(DM dmRhs, Vec rhs, DM dmSol_2, Vec omega1_0, DM dmU1, Vec u1_0);
    PetscErrorCode assemble_omega2_curl_rhs(DM dmRhs, Vec rhs, DM dmOmega2, Vec omega2_0, PetscReal Re);
    PetscErrorCode extract_u1_from_solution(DM dmSolSrc, Vec sol, DM dmSolDst, Vec u1_half);
    PetscErrorCode compute_curl_u1_to_omega2(DM dmSol_1, Vec u1_half, Vec omega2_half);

    // 时间推进：1-形式与2-形式子步
    // time: 当前整数时间步 k（对于 time_evolve1，直接用于外力项 f^k；对于 time_evolve2，用于计算半整数时间步 k-1/2）
    PetscErrorCode time_evolve1(DM dmSol_1,DM dmSol_2,
                                Vec sol1_old, Vec sol1_new,
                                Vec sol2_old, Vec sol2_new,
                                ExternalForce externalForce,
                                PetscReal time);
    PetscErrorCode time_evolve2(DM dmSol_1,DM dmSol_2,
                                Vec sol1_old, Vec sol1_new,
                                Vec sol2_old, Vec sol2_new,
                                ExternalForce externalForce,
                                PetscReal time);
    // 初始化PETSc对象为NULL
    void initialize_petsc_objects();

    // ===== 1形式系统矩阵组装相关函数 =====
    // 组装完整的1形式系统矩阵和右端项（包括外力项）
    PetscErrorCode assemble_1form_system_matrix(DM dmSol_1, DM dmSol_2, Mat A, Vec rhs, Vec u1_prev, Vec omega1_known, Vec omega2_prev, ExternalForce externalForce, PetscReal time, PetscReal dt);
    
    // 组装右端项（不包括外力）
    // 包含时间导数项、对流项和旋度项
    PetscErrorCode assemble_rhs1_vector(DM dmSol_1, Vec rhs, Vec u1_prev, DM dmSol_2, Vec omega1_known, Vec omega2_prev, PetscReal dt);
    
    // 组装1形式速度时间导数矩阵：1/dt * I（对 u₁）
    PetscErrorCode assemble_u1_dt_matrix(DM dmSol_1, Mat A, PetscReal dt);
    
    // 组装对流项矩阵：0.5 * ω₁^{h,k} ×（对 u₁）
    PetscErrorCode assemble_u1_conv_matrix(DM dmSol_1, Mat A, DM dmSol_2, Vec omega1_known);
    
    // 组装2形式涡度旋度矩阵：0.5/Re * ∇×（对 ω₂）
    PetscErrorCode assemble_omega2_curl_matrix(PetscReal Re, DM dmSol_1, Mat A);
    
    // 组装0形式压力梯度矩阵：∇（对 P₀）
    PetscErrorCode assemble_p0_gradient_matrix(DM dmSol_1, Mat A);
    
    // 组装u₁-ω₂耦合矩阵：∇×u₁ - ω₂ = 0
    PetscErrorCode assemble_u1_omega2_coupling_matrix(DM dmSol_1, Mat A);
    
    // 组装u₁散度矩阵：∇·u₁ = 0
    PetscErrorCode assemble_u1_divergence_matrix(DM dmSol_1, Mat A);
    
    // 组装外力项向量
    PetscErrorCode assemble_force1_vector(DM dmSol_1, Vec rhs, ExternalForce externalForce, PetscReal time);
    
    // 组装robust外力项向量（待实现）
    PetscErrorCode assemble_robust_force1_vector();

    // ===== 2形式系统矩阵组装相关函数 =====
    // 组装完整的2形式系统矩阵和右端项（包括外力项）
    PetscErrorCode assemble_2form_system_matrix(DM dmSol_1, DM dmSol_2, Mat A, Vec rhs, Vec u2_prev, Vec omega2_known, Vec omega1_prev, ExternalForce externalForce, PetscReal time, PetscReal dt);
    
    // 组装右端项（不包括外力）
    // 包含时间导数项、对流项和旋度项
    PetscErrorCode assemble_rhs2_vector(DM dmSol_2, Vec rhs, Vec u2_prev, DM dmSol_1, Vec omega2_known, Vec omega1_prev, PetscReal dt);
    
    // 组装2形式速度时间导数矩阵：1/dt * I（对 u₂）
    PetscErrorCode assemble_u2_dt_matrix(DM dmSol_2, Mat A, PetscReal dt);
    
    // 组装对流项矩阵：0.5 * ω₂^{h,k-1/2} ×（对 u₂）
    PetscErrorCode assemble_u2_conv_matrix(DM dmSol_2, Mat A, DM dmSol_1, Vec omega2_known);
    
    // 组装1形式涡度旋度矩阵：0.5/Re * ∇×（对 ω₁）
    PetscErrorCode assemble_omega1_curl_matrix(PetscReal Re, DM dmSol_2, Mat A);
    
    // 组装3形式压力梯度矩阵：∇（对 P₃）
    PetscErrorCode assemble_p3_gradient_matrix(DM dmSol_2, Mat A);
    
    // 组装u₂-ω₁耦合矩阵：∇×u₂ - ω₁ = 0
    PetscErrorCode assemble_u2_omega1_coupling_matrix(DM dmSol_2, Mat A);
    
    // 组装u₂散度矩阵：∇·u₂ = 0
    PetscErrorCode assemble_u2_divergence_matrix(DM dmSol_2, Mat A);
    
    // 组装外力项向量
    PetscErrorCode assemble_force2_vector(DM dmSol_2, Vec rhs, ExternalForce externalForce, PetscReal time);
};

// 线性求解器接口（定义于 src/linearsolver.cpp）
PetscErrorCode solve_linear_system_basic(Mat A, Vec rhs, Vec sol, DM dm, const char *optionsPrefix = NULL);
PetscErrorCode solve_linear_system_graddiv(Mat A, Vec rhs, Vec sol, DM dm, PetscReal gamma, const char *optionsPrefix = NULL);

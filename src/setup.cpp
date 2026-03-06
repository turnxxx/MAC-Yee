#include <mpi.h>
#include <petsc.h>
#include "../include/DUAL_MAC.h"
#include "../include/ref_sol.h"
#include "petscerror.h"
// 包含了构造函数，网格设置，精确解设置，边界条件设置等函数
// 设置 dm 对象,只做3d，周期边界情况
DUAL_MAC::DUAL_MAC(PetscReal time, PetscReal nu, PetscInt NX, PetscInt NY, PetscInt NZ, PetscInt Nt,
                   PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax, PetscReal zmin, PetscReal zmax,
                   PetscBool pinPressure) : DUAL_MAC()
{
    this->time = time;
    this->nu = nu;
    this->Nx = NX;
    this->Ny = NY;
    this->Nz = NZ;
    this->Nt = Nt;
    this->xmin = xmin;
    this->xmax = xmax;
    this->ymin = ymin;
    this->ymax = ymax;
    this->zmin = zmin;
    this->zmax = zmax;
    this->dx = (xmax - xmin) / NX;
    this->dy = (ymax - ymin) / NY;
    this->dz = (zmax - zmin) / NZ;
    this->dt = time / Nt;
    this->pinPressure = pinPressure;
}
// 创建周期边界DMStag和坐标，需要使用两套DM

PetscErrorCode DUAL_MAC::set_dm_periodic()
{
    PetscFunctionBeginUser;
    // 创建1形式的dm
    const PetscInt dof0 = 1, dof1 = 1, dof2 = 1, dof3 = 0; /* 1 dof on each face and element center */
    const PetscInt stencilWidth = 1;
    PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,
                             DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, this->Nx, this->Ny, this->Nz, PETSC_DECIDE, PETSC_DECIDE,
                             PETSC_DECIDE, dof0, dof1, dof2, dof3, DMSTAG_STENCIL_BOX, stencilWidth, NULL, NULL,
                             NULL, &dmSol_1));
    PetscCall(DMSetFromOptions(dmSol_1));
    PetscCall(DMSetUp(dmSol_1));
    PetscCall(DMStagSetUniformCoordinatesProduct(dmSol_1, this->xmin, this->xmax, 
                                                this->ymin, this->ymax, 
                                                this->zmin, this->zmax));
    // 接下来创建索引相同的2形式的dm
    const PetscInt dof0_2=0,dof1_2=1,dof2_2=1,dof3_2=1;
    PetscCall(DMStagCreateCompatibleDMStag(dmSol_1, dof0_2, dof1_2, dof2_2, dof3_2, &dmSol_2));
    PetscCall(DMStagSetUniformCoordinatesProduct(dmSol_2, this->xmin, this->xmax,
                                                 this->ymin, this->ymax,
                                                 this->zmin, this->zmax));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// 设置参考解,暂时先不实现
PetscErrorCode DUAL_MAC::set_reference_solution(RefSol refSol)
{
    PetscFunctionBeginUser;
    this->refSol_cached = refSol;
    this->hasRefSol = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// 打印DM信息（用于调试）
PetscErrorCode DUAL_MAC::view_dm(PetscBool view_dm1, PetscBool view_dm2, PetscViewer viewer)
{
    PetscFunctionBeginUser;

    // 如果没有指定查看器，使用默认的 stdout 查看器
    PetscBool viewer_created = PETSC_FALSE;
    if (!viewer) {
        PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
        PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
        PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL));
        viewer_created = PETSC_TRUE;
    }

    // 打印 dmSol_1 的信息
    if (view_dm1 && dmSol_1) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n========================================\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DM Sol 1 (1-form system) 信息:\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "========================================\n"));
        PetscCall(DMView(dmSol_1, viewer));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }

    // 打印 dmSol_2 的信息
    if (view_dm2 && dmSol_2) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "========================================\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DM Sol 2 (2-form system) 信息:\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "========================================\n"));
        PetscCall(DMView(dmSol_2, viewer));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }

    // 如果创建了查看器，需要恢复格式并销毁它
    if (viewer_created) {
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


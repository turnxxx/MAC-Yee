// 结果输出相关函数
#include <petsc.h>
#include "../include/DUAL_MAC.h"

PetscErrorCode DUAL_MAC::output_vector(Vec &pV, const char *name)
{
    PetscFunctionBeginUser;

    if (!pV) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_vector: 向量为空，跳过输出。\n"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (!name || name[0] == '\0') {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_vector: 输出文件名为空。\n"));
        PetscFunctionReturn(PETSC_ERR_ARG_NULL);
    }

    PetscViewer viewer = NULL;
    PetscCall(PetscObjectSetName((PetscObject)pV, "b"));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, name, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(pV, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DUAL_MAC::output_matrix(Mat &pA, const char *name)
{
    PetscFunctionBeginUser;

    if (!pA) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_matrix: 矩阵为空，跳过输出。\n"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (!name || name[0] == '\0') {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_matrix: 输出文件名为空。\n"));
        PetscFunctionReturn(PETSC_ERR_ARG_NULL);
    }

    PetscViewer viewer = NULL;
    PetscCall(PetscObjectSetName((PetscObject)pA, "A"));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, name, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(MatView(pA, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DUAL_MAC::output_solution(const char *filename)
{
    PetscFunctionBeginUser;

    if (!filename || filename[0] == '\0') {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_solution: 输出文件名为空。\n"));
        PetscFunctionReturn(PETSC_ERR_ARG_NULL);
    }

    // 优先输出缓存的最终 1-form 解；若不存在则回退到成员 sol。
    if (sol1_cached) {
        PetscCall(output_vector(sol1_cached, filename));
    } else if (sol) {
        PetscCall(output_vector(sol, filename));
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "output_solution: 当前没有可输出的解向量。\n"));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
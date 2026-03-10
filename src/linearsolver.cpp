#include "../include/DUAL_MAC.h"
#include "petscsys.h"
#include <array>
#include <cmath>
#include <cstring>
#include <petsc.h>
#include <unordered_map>
#include <vector>
// 线性求解器封装
// 为了方便在不同地方复用线性代数设置，这里提供一个简单的求解接口
// 注意：具体求解器类型、预条件子等可以通过命令行参数设置

#if DUAL_MAC_DEBUG
static PetscErrorCode debug_check_vec_finite_ls(Vec v, const char *tag) {
  PetscFunctionBeginUser;
  if (!v)
    PetscFunctionReturn(PETSC_SUCCESS);

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
    const PetscBool ok =
        (PetscBool)(std::isfinite((double)PetscRealPart(arr[i])) &&
                    std::isfinite((double)PetscImaginaryPart(arr[i])));
#else
    const PetscBool ok =
        (PetscBool)(std::isfinite((double)PetscRealPart(arr[i])));
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
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD,
                    "[DEBUG][LS finite] %s first invalid idx=%" PetscInt_FMT
                    ", value=(%.16e, %.16e)\n",
                    tag ? tag : "vec", firstBad, (double)PetscRealPart(badVal),
                    (double)PetscImaginaryPart(badVal)));
#else
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "[DEBUG][LS finite] %s first invalid idx=%" PetscInt_FMT
        ", value=%.16e\n",
        tag ? tag : "vec", firstBad, (double)PetscRealPart(badVal)));
#endif
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[DEBUG][LS finite] %s all finite\n",
                          tag ? tag : "vec"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

// 为 PCFieldSplit 构造两组索引：
// - u 场：速度 + 涡量
// - p 场：压力
static PetscErrorCode build_up_fieldsplit_is(DM dm, IS *isU, IS *isP) {
  PetscFunctionBeginUser;
  PetscInt dof0, dof1, dof2, dof3;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));

  const PetscBool isTwoLike =
      (dof3 > 0 && dof0 == 0) ? PETSC_TRUE : PETSC_FALSE;

  std::vector<DMStagStencil> stU;
  std::vector<DMStagStencil> stP;
  stU.reserve((size_t)nx * (size_t)ny * (size_t)nz * 6);
  stP.reserve((size_t)nx * (size_t)ny * (size_t)nz);

  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        DMStagStencil s;
        s.i = ex;
        s.j = ey;
        s.k = ez;
        s.c = 0;

        if (isTwoLike) {
          // two-form: u2 在面(dof2)，omega1 在棱(dof1)，p 在单元中心(dof3)
          if (dof2 > 0) {
            s.loc = LEFT;
            stU.push_back(s);
            s.loc = DOWN;
            stU.push_back(s);
            s.loc = BACK;
            stU.push_back(s);
          }
          if (dof1 > 0) {
            s.loc = BACK_DOWN;
            stU.push_back(s);
            s.loc = BACK_LEFT;
            stU.push_back(s);
            s.loc = DOWN_LEFT;
            stU.push_back(s);
          }
          if (dof3 > 0) {
            s.loc = ELEMENT;
            stP.push_back(s);
          }
        } else {
          // one-form/half: u1 在棱(dof1)，omega2 在面(dof2, half 步可能无)，p
          // 在顶点(dof0)
          if (dof1 > 0) {
            s.loc = BACK_DOWN;
            stU.push_back(s);
            s.loc = BACK_LEFT;
            stU.push_back(s);
            s.loc = DOWN_LEFT;
            stU.push_back(s);
          }
          if (dof2 > 0) {
            s.loc = LEFT;
            stU.push_back(s);
            s.loc = DOWN;
            stU.push_back(s);
            s.loc = BACK;
            stU.push_back(s);
          }
          if (dof0 > 0) {
            s.loc = BACK_DOWN_LEFT;
            stP.push_back(s);
          } else if (dof3 > 0) {
            s.loc = ELEMENT;
            stP.push_back(s);
          }
        }
      }
    }
  }

  *isU = NULL;
  *isP = NULL;
  if (!stU.empty())
    PetscCall(
        DMStagCreateISFromStencils(dm, (PetscInt)stU.size(), stU.data(), isU));
  if (!stP.empty())
    PetscCall(
        DMStagCreateISFromStencils(dm, (PetscInt)stP.size(), stP.data(), isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 构造三组全局索引：v(速度), w(涡度), p(压力)
static PetscErrorCode build_vwp_fieldsplit_is(DM dm, IS *isV, IS *isW,
                                              IS *isP) {
  PetscFunctionBeginUser;
  PetscInt dof0, dof1, dof2, dof3;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));

  const PetscBool isTwoLike =
      (dof3 > 0 && dof0 == 0) ? PETSC_TRUE : PETSC_FALSE;

  std::vector<DMStagStencil> stV, stW, stP;
  stV.reserve((size_t)nx * (size_t)ny * (size_t)nz * 3);
  stW.reserve((size_t)nx * (size_t)ny * (size_t)nz * 3);
  stP.reserve((size_t)nx * (size_t)ny * (size_t)nz);

  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        DMStagStencil s;
        s.i = ex;
        s.j = ey;
        s.k = ez;
        s.c = 0;
        if (isTwoLike) {
          // two-form: v=u2(face), w=omega1(edge), p=p3(element)
          if (dof2 > 0) {
            s.loc = LEFT;
            stV.push_back(s);
            s.loc = DOWN;
            stV.push_back(s);
            s.loc = BACK;
            stV.push_back(s);
          }
          if (dof1 > 0) {
            s.loc = BACK_DOWN;
            stW.push_back(s);
            s.loc = BACK_LEFT;
            stW.push_back(s);
            s.loc = DOWN_LEFT;
            stW.push_back(s);
          }
          if (dof3 > 0) {
            s.loc = ELEMENT;
            stP.push_back(s);
          }
        } else {
          // one-form/half: v=u1(edge), w=omega2(face), p=p0(vertex)
          if (dof1 > 0) {
            s.loc = BACK_DOWN;
            stV.push_back(s);
            s.loc = BACK_LEFT;
            stV.push_back(s);
            s.loc = DOWN_LEFT;
            stV.push_back(s);
          }
          if (dof2 > 0) {
            s.loc = LEFT;
            stW.push_back(s);
            s.loc = DOWN;
            stW.push_back(s);
            s.loc = BACK;
            stW.push_back(s);
          }
          if (dof0 > 0) {
            s.loc = BACK_DOWN_LEFT;
            stP.push_back(s);
          } else if (dof3 > 0) {
            s.loc = ELEMENT;
            stP.push_back(s);
          }
        }
      }
    }
  }

  *isV = NULL;
  *isW = NULL;
  *isP = NULL;
  if (!stV.empty())
    PetscCall(
        DMStagCreateISFromStencils(dm, (PetscInt)stV.size(), stV.data(), isV));
  if (!stW.empty())
    PetscCall(
        DMStagCreateISFromStencils(dm, (PetscInt)stW.size(), stW.data(), isW));
  if (!stP.empty())
    PetscCall(
        DMStagCreateISFromStencils(dm, (PetscInt)stP.size(), stP.data(), isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Hypre ADS 需要顶点（V^0, BACK_DOWN_LEFT）坐标来内部构造
// Raviart-Thomas 插值矩阵。坐标数量 = 本地顶点数 = nx*ny*nz。
static PetscErrorCode set_twoform_ads_coordinates(DM dm, PC pcAds) {
  PetscFunctionBeginUser;
  PetscInt dof0, dof1, dof2, dof3;
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  if (!(dof3 > 0 && dof0 == 0 && dof2 > 0))
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));

  PetscScalar **cArrX = NULL, **cArrY = NULL, **cArrZ = NULL;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cArrX, &cArrY, &cArrZ));
  PetscInt icx_prev, icy_prev, icz_prev;
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icx_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icy_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icz_prev));

  const PetscInt nVertexLocal = nx * ny * nz;
  std::vector<PetscReal> coords;
  coords.reserve((size_t)nVertexLocal * 3);
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        coords.push_back((PetscReal)PetscRealPart(cArrX[ex][icx_prev]));
        coords.push_back((PetscReal)PetscRealPart(cArrY[ey][icy_prev]));
        coords.push_back((PetscReal)PetscRealPart(cArrZ[ez][icz_prev]));
      }
    }
  }

  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dm, &cArrX, &cArrY, &cArrZ));
  PetscCall(PCSetCoordinates(pcAds, 3, nVertexLocal, coords.data()));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* OLD set_twoform_ads_coordinates body removed — was providing face
   coordinates (V^2) instead of the vertex coordinates (V^0) that ADS needs.
#if 0
  PetscInt nVLocal = 0;
  PetscCall(build_vwp_fieldsplit_is(dm, &isV, &isW, &isP));
  if (!isV) {
    if (isW)
      PetscCall(ISDestroy(&isW));
    if (isP)
      PetscCall(ISDestroy(&isP));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(ISGetLocalSize(isV, &nVLocal));
  PetscCall(ISGetIndices(isV, &idxV));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));

  PetscScalar **cArrX = NULL, **cArrY = NULL, **cArrZ = NULL;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cArrX, &cArrY, &cArrZ));
  PetscInt icx_center, icx_prev, icy_center, icy_prev, icz_center, icz_prev;
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, ELEMENT, &icx_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icx_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, ELEMENT, &icy_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icy_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, ELEMENT, &icz_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, LEFT, &icz_prev));

  // 先构造与 stencil 序一致的 (stencil,xyz)，再按 isV 的全局索引重排，
  // 避免“坐标顺序 != 子块自由度顺序”导致 ADS 失败。
  std::vector<DMStagStencil> stV;
  stV.reserve((size_t)nVLocal);
  std::vector<std::array<PetscReal, 3>> xyzByStencilOrder;
  xyzByStencilOrder.reserve((size_t)nVLocal);
  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        DMStagStencil s;
        s.i = ex;
        s.j = ey;
        s.k = ez;
        s.c = 0;
        // LEFT 面速度
        s.loc = LEFT;
        stV.push_back(s);
        xyzByStencilOrder.push_back(
            {(PetscReal)PetscRealPart(cArrX[ex][icx_prev]),
             (PetscReal)PetscRealPart(cArrY[ey][icy_center]),
             (PetscReal)PetscRealPart(cArrZ[ez][icz_center])});
        // DOWN 面速度
        s.loc = DOWN;
        stV.push_back(s);
        xyzByStencilOrder.push_back(
            {(PetscReal)PetscRealPart(cArrX[ex][icx_center]),
             (PetscReal)PetscRealPart(cArrY[ey][icy_prev]),
             (PetscReal)PetscRealPart(cArrZ[ez][icz_center])});
        // BACK 面速度
        s.loc = BACK;
        stV.push_back(s);
        xyzByStencilOrder.push_back(
            {(PetscReal)PetscRealPart(cArrX[ex][icx_center]),
             (PetscReal)PetscRealPart(cArrY[ey][icy_center]),
             (PetscReal)PetscRealPart(cArrZ[ez][icz_prev])});
      }
    }
  }

  PetscCall(
      DMStagRestoreProductCoordinateArraysRead(dm, &cArrX, &cArrY, &cArrZ));

  if ((PetscInt)stV.size() != nVLocal) {
    PetscCall(ISRestoreIndices(isV, &idxV));
    if (isV)
      PetscCall(ISDestroy(&isV));
    if (isW)
      PetscCall(ISDestroy(&isW));
    if (isP)
      PetscCall(ISDestroy(&isP));
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
            "ADS coordinates size mismatch: built_n=%" PetscInt_FMT
            ", isV_local=%" PetscInt_FMT,
            (PetscInt)stV.size(), nVLocal);
  }

  PetscCall(DMStagCreateISFromStencils(dm, (PetscInt)stV.size(), stV.data(),
                                       &isVBuilt));
  PetscInt nVBuiltLocal = 0;
  PetscCall(ISGetLocalSize(isVBuilt, &nVBuiltLocal));
  PetscCall(ISGetIndices(isVBuilt, &idxVBuilt));
  if (nVBuiltLocal != nVLocal) {
    PetscCall(ISRestoreIndices(isVBuilt, &idxVBuilt));
    PetscCall(ISDestroy(&isVBuilt));
    PetscCall(ISRestoreIndices(isV, &idxV));
    if (isV)
      PetscCall(ISDestroy(&isV));
    if (isW)
      PetscCall(ISDestroy(&isW));
    if (isP)
      PetscCall(ISDestroy(&isP));
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
            "ADS coordinates index mismatch: built_local=%" PetscInt_FMT
            ", isV_local=%" PetscInt_FMT,
            nVBuiltLocal, nVLocal);
  }

  std::unordered_map<PetscInt, std::array<PetscReal, 3>> idx2xyz;
  idx2xyz.reserve((size_t)nVLocal);
  for (PetscInt i = 0; i < nVBuiltLocal; ++i) {
    idx2xyz[idxVBuilt[i]] = xyzByStencilOrder[(size_t)i];
  }

  std::vector<PetscReal> coords;
  coords.reserve((size_t)nVLocal * 3);
  for (PetscInt i = 0; i < nVLocal; ++i) {
    const auto it = idx2xyz.find(idxV[i]);
    if (it == idx2xyz.end()) {
      PetscCall(ISRestoreIndices(isVBuilt, &idxVBuilt));
      PetscCall(ISDestroy(&isVBuilt));
      PetscCall(ISRestoreIndices(isV, &idxV));
      if (isV)
        PetscCall(ISDestroy(&isV));
      if (isW)
        PetscCall(ISDestroy(&isW));
      if (isP)
        PetscCall(ISDestroy(&isP));
      SETERRQ(
          PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
          "ADS coordinate reorder failed: missing global index %" PetscInt_FMT,
          idxV[i]);
    }
    coords.push_back(it->second[0]);
    coords.push_back(it->second[1]);
    coords.push_back(it->second[2]);
  }

  PetscCall(PCSetCoordinates(pcAds, 3, nVLocal, coords.data()));

  PetscCall(ISRestoreIndices(isV, &idxV));
  PetscCall(ISRestoreIndices(isVBuilt, &idxVBuilt));
  PetscCall(ISDestroy(&isVBuilt));
  if (isV)
    PetscCall(ISDestroy(&isV));
  if (isW)
    PetscCall(ISDestroy(&isW));
  if (isP)
    PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif */

// 把全局 v/w 索引转换到外层 u 子块坐标（用于内层 v/w 分裂）
static PetscErrorCode build_vw_subspace_is(DM dm, IS *isVSub, IS *isWSub) {
  PetscFunctionBeginUser;
  IS isU = NULL, isP = NULL, isV = NULL, isW = NULL, isPvw = NULL,
     isUAll = NULL;
  const PetscInt *idxUAll = NULL, *idxV = NULL, *idxW = NULL;
  PetscInt nUAll = 0, nVLocal = 0, nWLocal = 0;

  PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
  PetscCall(build_vwp_fieldsplit_is(dm, &isV, &isW, &isPvw));
  if (!isU || !isV || !isW) {
    if (isU)
      PetscCall(ISDestroy(&isU));
    if (isP)
      PetscCall(ISDestroy(&isP));
    if (isV)
      PetscCall(ISDestroy(&isV));
    if (isW)
      PetscCall(ISDestroy(&isW));
    if (isPvw)
      PetscCall(ISDestroy(&isPvw));
    *isVSub = NULL;
    *isWSub = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(ISAllGather(isU, &isUAll));
  PetscCall(ISGetSize(isUAll, &nUAll));
  PetscCall(ISGetIndices(isUAll, &idxUAll));
  std::unordered_map<PetscInt, PetscInt> g2u;
  g2u.reserve((size_t)nUAll);
  for (PetscInt i = 0; i < nUAll; ++i)
    g2u[idxUAll[i]] = i;

  PetscCall(ISGetLocalSize(isV, &nVLocal));
  PetscCall(ISGetLocalSize(isW, &nWLocal));
  PetscCall(ISGetIndices(isV, &idxV));
  PetscCall(ISGetIndices(isW, &idxW));
  std::vector<PetscInt> vSub((size_t)nVLocal), wSub((size_t)nWLocal);
  for (PetscInt i = 0; i < nVLocal; ++i) {
    const auto it = g2u.find(idxV[i]);
    vSub[(size_t)i] = (it != g2u.end()) ? it->second : 0;
  }
  for (PetscInt i = 0; i < nWLocal; ++i) {
    const auto it = g2u.find(idxW[i]);
    wSub[(size_t)i] = (it != g2u.end()) ? it->second : 0;
  }

  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nVLocal,
                            vSub.data(), PETSC_COPY_VALUES, isVSub));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nWLocal,
                            wSub.data(), PETSC_COPY_VALUES, isWSub));

  PetscCall(ISRestoreIndices(isW, &idxW));
  PetscCall(ISRestoreIndices(isV, &idxV));
  PetscCall(ISRestoreIndices(isUAll, &idxUAll));
  PetscCall(ISDestroy(&isUAll));
  PetscCall(ISDestroy(&isU));
  PetscCall(ISDestroy(&isP));
  PetscCall(ISDestroy(&isV));
  PetscCall(ISDestroy(&isW));
  PetscCall(ISDestroy(&isPvw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 构造 S_p = -(1/gamma) * M_p（采用 lumped 质量矩阵，对角常数）
static PetscErrorCode
build_pressure_schur_user_mat(Mat A, DM dm, PetscReal gamma, Mat *Suser) {
  PetscFunctionBeginUser;
  *Suser = NULL;
  if (gamma == 0.0)
    PetscFunctionReturn(PETSC_SUCCESS);

  IS isU = NULL, isP = NULL;
  PetscInt nPGlobal = 0, nPLocal = 0;

  PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
  if (!isP) {
    if (isU)
      PetscCall(ISDestroy(&isU));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(ISGetSize(isP, &nPGlobal));
  PetscCall(ISGetLocalSize(isP, &nPLocal));
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)A), nPLocal, nPLocal,
                         nPGlobal, nPGlobal, 1, NULL, 1, NULL, Suser));
  PetscCall(MatSetOption(*Suser, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  PetscScalar massDiag = (PetscScalar)(-1.0 / gamma);

  PetscInt rstart = 0, rend = 0;
  PetscCall(MatGetOwnershipRange(*Suser, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; ++row)
    PetscCall(MatSetValue(*Suser, row, row, massDiag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*Suser, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Suser, MAT_FINAL_ASSEMBLY));

  if (isU)
    PetscCall(ISDestroy(&isU));
  PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode
build_oneform_pressure_identity_schur_user_mat(Mat A, DM dm, PetscReal alpha,
                                               PetscReal dt, Mat *Suser) {
  PetscFunctionBeginUser;
  *Suser = NULL;
  if (dt <= 0.0)
    PetscFunctionReturn(PETSC_SUCCESS);

  IS isU = NULL, isP = NULL;
  PetscInt nPGlobal = 0, nPLocal = 0;
  PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
  if (!isP) {
    if (isU)
      PetscCall(ISDestroy(&isU));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(ISGetSize(isP, &nPGlobal));
  PetscCall(ISGetLocalSize(isP, &nPLocal));
  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)A), nPLocal, nPLocal,
                         nPGlobal, nPGlobal, 1, NULL, 1, NULL, Suser));
  PetscCall(MatSetOption(*Suser, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  const PetscScalar diag = (PetscScalar)(-1.0 / (alpha + 1.0 / dt));
  PetscInt rstart = 0, rend = 0;
  PetscCall(MatGetOwnershipRange(*Suser, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; ++row)
    PetscCall(MatSetValue(*Suser, row, row, diag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*Suser, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Suser, MAT_FINAL_ASSEMBLY));

  if (isU)
    PetscCall(ISDestroy(&isU));
  PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 构造速度块的内层 Schur 补预条件矩阵：
// P_v = A_vv - A_vw * A_wv
// 耦合方程统一采用 ω - ∇×u = 0 约定，即 A_ww = +I，
// 内层 Schur 补 S_v = A_vv - A_vw * A_ww^{-1} * A_wv = A_vv - A_vw * A_wv
static PetscErrorCode build_velocity_schur_precond_mat(Mat A, DM dm, Mat *Pv) {
  PetscFunctionBeginUser;
  *Pv = NULL;

  IS isV = NULL, isW = NULL, isP = NULL;
  PetscCall(build_vwp_fieldsplit_is(dm, &isV, &isW, &isP));
  if (!isV || !isW) {
    if (isV)
      PetscCall(ISDestroy(&isV));
    if (isW)
      PetscCall(ISDestroy(&isW));
    if (isP)
      PetscCall(ISDestroy(&isP));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Mat Avv = NULL, Avw = NULL, Awv = NULL, CurlCurl = NULL;
  PetscCall(MatCreateSubMatrix(A, isV, isV, MAT_INITIAL_MATRIX, &Avv));
  PetscCall(MatCreateSubMatrix(A, isV, isW, MAT_INITIAL_MATRIX, &Avw));
  PetscCall(MatCreateSubMatrix(A, isW, isV, MAT_INITIAL_MATRIX, &Awv));

  PetscCall(MatMatMult(Avw, Awv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CurlCurl));

  PetscCall(MatDuplicate(Avv, MAT_COPY_VALUES, Pv));
  PetscCall(MatAXPY(*Pv, -1.0, CurlCurl, DIFFERENT_NONZERO_PATTERN));

  PetscCall(MatDestroy(&CurlCurl));
  PetscCall(MatDestroy(&Awv));
  PetscCall(MatDestroy(&Avw));
  PetscCall(MatDestroy(&Avv));
  PetscCall(ISDestroy(&isV));
  PetscCall(ISDestroy(&isW));
  if (isP)
    PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Hypre ADS 辅助离散算子：
// G_ads = d0: 离散梯度 (vertex -> edge), 维度 N_edge x N_vertex
// C_ads = d1: 离散旋度 (edge -> face), 维度 N_face x N_edge
// 在辅助 DM (dof=(1,1,1,0)) 上组装纯拓扑算子，满足 d1*d0 = 0。
static PetscErrorCode build_twoform_ads_aux_mats(Mat A, DM dm, Mat *G_ads,
                                                 Mat *C_ads) {
  PetscFunctionBeginUser;
  *G_ads = NULL;
  *C_ads = NULL;
  PetscInt dof0, dof1, dof2, dof3;

  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  if (!(dof3 > 0 && dof0 == 0 && dof1 > 0 && dof2 > 0))
    PetscFunctionReturn(PETSC_SUCCESS);

  (void)A;

  DM dmTopo = NULL;
  PetscCall(DMStagCreateCompatibleDMStag(dm, 1, 1, 1, 0, &dmTopo));
  PetscCall(DMSetUp(dmTopo));

  Mat Topo = NULL;
  PetscCall(DMCreateMatrix(dmTopo, &Topo));
  PetscCall(MatZeroEntries(Topo));
  PetscCall(MatSetOption(Topo, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dmTopo, &startx, &starty, &startz, &nx, &ny, &nz,
                             NULL, NULL, NULL));

  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        const PetscInt exp1 = ex + 1;
        const PetscInt eyp1 = ey + 1;
        const PetscInt ezp1 = ez + 1;

        // ---- d0: vertex -> edge (discrete gradient) ----
        {
          DMStagStencil row, col[2];
          PetscScalar val[2] = {1.0, -1.0};
          row.c = 0;
          col[0].c = col[1].c = 0;
          col[0].loc = col[1].loc = BACK_DOWN_LEFT;

          // BACK_DOWN (x-edge): +vertex(ex+1) - vertex(ex)
          row.loc = BACK_DOWN;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].i = exp1;
          col[0].j = ey;
          col[0].k = ez;
          col[1].i = ex;
          col[1].j = ey;
          col[1].k = ez;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 2, col,
                                              val, ADD_VALUES));

          // BACK_LEFT (y-edge): +vertex(ey+1) - vertex(ey)
          row.loc = BACK_LEFT;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].i = ex;
          col[0].j = eyp1;
          col[0].k = ez;
          col[1].i = ex;
          col[1].j = ey;
          col[1].k = ez;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 2, col,
                                              val, ADD_VALUES));

          // DOWN_LEFT (z-edge): +vertex(ez+1) - vertex(ez)
          row.loc = DOWN_LEFT;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].i = ex;
          col[0].j = ey;
          col[0].k = ezp1;
          col[1].i = ex;
          col[1].j = ey;
          col[1].k = ez;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 2, col,
                                              val, ADD_VALUES));
        }

        // ---- d1: edge -> face (discrete curl) ----
        {
          DMStagStencil row, col[4];
          PetscScalar val[4];
          row.c = 0;
          for (PetscInt q = 0; q < 4; ++q)
            col[q].c = 0;

          // BACK face (xy-plane, z-normal):
          //  +BACK_DOWN(ex,ey,ez) - BACK_DOWN(ex,ey+1,ez)
          //  -BACK_LEFT(ex,ey,ez) + BACK_LEFT(ex+1,ey,ez)
          row.loc = BACK;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].loc = BACK_DOWN;
          col[0].i = ex;
          col[0].j = ey;
          col[0].k = ez;
          val[0] = 1.0;
          col[1].loc = BACK_DOWN;
          col[1].i = ex;
          col[1].j = eyp1;
          col[1].k = ez;
          val[1] = -1.0;
          col[2].loc = BACK_LEFT;
          col[2].i = ex;
          col[2].j = ey;
          col[2].k = ez;
          val[2] = -1.0;
          col[3].loc = BACK_LEFT;
          col[3].i = exp1;
          col[3].j = ey;
          col[3].k = ez;
          val[3] = 1.0;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 4, col,
                                              val, ADD_VALUES));

          // DOWN face (xz-plane, y-normal):
          //  -BACK_DOWN(ex,ey,ez) + BACK_DOWN(ex,ey,ez+1)
          //  +DOWN_LEFT(ex,ey,ez) - DOWN_LEFT(ex+1,ey,ez)
          row.loc = DOWN;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].loc = BACK_DOWN;
          col[0].i = ex;
          col[0].j = ey;
          col[0].k = ez;
          val[0] = -1.0;
          col[1].loc = BACK_DOWN;
          col[1].i = ex;
          col[1].j = ey;
          col[1].k = ezp1;
          val[1] = 1.0;
          col[2].loc = DOWN_LEFT;
          col[2].i = ex;
          col[2].j = ey;
          col[2].k = ez;
          val[2] = 1.0;
          col[3].loc = DOWN_LEFT;
          col[3].i = exp1;
          col[3].j = ey;
          col[3].k = ez;
          val[3] = -1.0;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 4, col,
                                              val, ADD_VALUES));

          // LEFT face (yz-plane, x-normal):
          //  +BACK_LEFT(ex,ey,ez) - BACK_LEFT(ex,ey,ez+1)
          //  -DOWN_LEFT(ex,ey,ez) + DOWN_LEFT(ex,ey+1,ez)
          row.loc = LEFT;
          row.i = ex;
          row.j = ey;
          row.k = ez;
          col[0].loc = BACK_LEFT;
          col[0].i = ex;
          col[0].j = ey;
          col[0].k = ez;
          val[0] = 1.0;
          col[1].loc = BACK_LEFT;
          col[1].i = ex;
          col[1].j = ey;
          col[1].k = ezp1;
          val[1] = -1.0;
          col[2].loc = DOWN_LEFT;
          col[2].i = ex;
          col[2].j = ey;
          col[2].k = ez;
          val[2] = -1.0;
          col[3].loc = DOWN_LEFT;
          col[3].i = ex;
          col[3].j = eyp1;
          col[3].k = ez;
          val[3] = 1.0;
          PetscCall(DMStagMatSetValuesStencil(dmTopo, Topo, 1, &row, 4, col,
                                              val, ADD_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(Topo, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Topo, MAT_FINAL_ASSEMBLY));

  // Build IS for vertex (0-cell), edge (1-cell), face (2-cell) on topo DM
  std::vector<DMStagStencil> stVertex, stEdge, stFace;
  stVertex.reserve((size_t)nx * (size_t)ny * (size_t)nz);
  stEdge.reserve((size_t)nx * (size_t)ny * (size_t)nz * 3);
  stFace.reserve((size_t)nx * (size_t)ny * (size_t)nz * 3);

  for (PetscInt ez = startz; ez < startz + nz; ++ez) {
    for (PetscInt ey = starty; ey < starty + ny; ++ey) {
      for (PetscInt ex = startx; ex < startx + nx; ++ex) {
        DMStagStencil s;
        s.i = ex;
        s.j = ey;
        s.k = ez;
        s.c = 0;

        s.loc = BACK_DOWN_LEFT;
        stVertex.push_back(s);

        s.loc = BACK_DOWN;
        stEdge.push_back(s);
        s.loc = BACK_LEFT;
        stEdge.push_back(s);
        s.loc = DOWN_LEFT;
        stEdge.push_back(s);

        s.loc = LEFT;
        stFace.push_back(s);
        s.loc = DOWN;
        stFace.push_back(s);
        s.loc = BACK;
        stFace.push_back(s);
      }
    }
  }

  IS isVertex = NULL, isEdge = NULL, isFace = NULL;
  PetscCall(DMStagCreateISFromStencils(dmTopo, (PetscInt)stVertex.size(),
                                       stVertex.data(), &isVertex));
  PetscCall(DMStagCreateISFromStencils(dmTopo, (PetscInt)stEdge.size(),
                                       stEdge.data(), &isEdge));
  PetscCall(DMStagCreateISFromStencils(dmTopo, (PetscInt)stFace.size(),
                                       stFace.data(), &isFace));

  PetscCall(
      MatCreateSubMatrix(Topo, isEdge, isVertex, MAT_INITIAL_MATRIX, G_ads));
  PetscCall(
      MatCreateSubMatrix(Topo, isFace, isEdge, MAT_INITIAL_MATRIX, C_ads));

  PetscCall(ISDestroy(&isVertex));
  PetscCall(ISDestroy(&isEdge));
  PetscCall(ISDestroy(&isFace));
  PetscCall(MatDestroy(&Topo));
  PetscCall(DMDestroy(&dmTopo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode configure_fieldsplit_subksp_defaults(
    PC pc, Mat A, DM dm, const char *optionsPrefix,
    PetscReal gammaForPressureSchur, PetscBool usePressureMassSchur,
    PetscReal alphaOneForm, PetscReal dt) {
  PetscFunctionBeginUser;
  PetscBool isFieldSplit = PETSC_FALSE;
  PetscCall(
      PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &isFieldSplit));
  if (!isFieldSplit)
    PetscFunctionReturn(PETSC_SUCCESS);

  const PetscBool isOneSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "one_", 4) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool isHalfSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "half_", 5) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool isTwoSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "two_", 4) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;
  PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));
  // 外层 u/p 顺序：
  // - half_ 使用 LOWER：先解 field0=u，再解 Schur(field1=p)
  // - one_/two_ 保持 UPPER：先解 field1=p，再回代 field0=u
  PetscCall(PCFieldSplitSetSchurFactType(
      pc, isHalfSystem ? PC_FIELDSPLIT_SCHUR_FACT_LOWER
                       : PC_FIELDSPLIT_SCHUR_FACT_UPPER));
  if (usePressureMassSchur && gammaForPressureSchur > 0.0) {
    Mat Suser = NULL;
    PetscCall(
        build_pressure_schur_user_mat(A, dm, gammaForPressureSchur, &Suser));
    if (Suser) {
      PetscCall(
          PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, Suser));
      PetscCall(MatDestroy(&Suser));
    }
  } else if ((isOneSystem || isHalfSystem) && dt > 0.0) {
    Mat Suser = NULL;
    PetscCall(build_oneform_pressure_identity_schur_user_mat(
        A, dm, alphaOneForm, dt, &Suser));
    if (Suser) {
      PetscCall(
          PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, Suser));
      PetscCall(MatDestroy(&Suser));
    }
  }

  PetscInt nSplit = 0;
  KSP *subksp = NULL;
  PetscCall(PCSetUp(pc));
  PetscCall(PCFieldSplitGetSubKSP(pc, &nSplit, &subksp));
  PetscBool adsFallbackLocal = PETSC_FALSE, adsFallbackGlobal = PETSC_FALSE;
  PetscBool hasAdsFallbackLocal = PETSC_FALSE,
            hasAdsFallbackGlobal = PETSC_FALSE;
  if (optionsPrefix && optionsPrefix[0] != '\0') {
    PetscCall(PetscOptionsGetBool(NULL, optionsPrefix, "-ads_fallback_to_gamg",
                                  &adsFallbackLocal, &hasAdsFallbackLocal));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ads_fallback_to_gamg",
                                &adsFallbackGlobal, &hasAdsFallbackGlobal));
  const PetscBool adsFallbackToGamg =
      hasAdsFallbackLocal
          ? adsFallbackLocal
          : (hasAdsFallbackGlobal ? adsFallbackGlobal : PETSC_FALSE);

  for (PetscInt i = 0; i < nSplit; ++i) {
    const char *prefix = NULL;
    PetscCall(KSPGetOptionsPrefix(subksp[i], &prefix));
    const PetscBool isUBlock = (prefix && std::strstr(prefix, "fieldsplit_u_"))
                                   ? PETSC_TRUE
                                   : (i == 0 ? PETSC_TRUE : PETSC_FALSE);
    const PetscBool isPBlock = isUBlock ? PETSC_FALSE : PETSC_TRUE;

    // 显式绑定子块前缀，保证 quick_test 中的
    // -half/-one/-two_fieldsplit_* 选项可以命中。
    if (optionsPrefix && optionsPrefix[0] != '\0') {
      char subPrefix[256];
      PetscCall(PetscSNPrintf(subPrefix, sizeof(subPrefix), "%sfieldsplit_%s_",
                              optionsPrefix, isUBlock ? "u" : "p"));
      PetscCall(KSPSetOptionsPrefix(subksp[i], subPrefix));
    }

    PC subpc = NULL;
    PetscCall(KSPGetPC(subksp[i], &subpc));

    if (isPBlock) {
      // 不再硬编码 p 块求解器/预条件，完全由命令行控制
      PetscCall(KSPSetFromOptions(subksp[i]));
      continue;
    }

    // u 块内层分裂：仅当 v/w 两个场都存在时，才启用 Schur(v,w) 嵌套。
    // half_ 场景通常只有速度场，无独立涡度场，需回退为单块 u 求解。
    // 不再硬编码 u 块外层 KSP 类型，完全由命令行控制
    IS isVSub = NULL, isWSub = NULL;
    PetscInt nVSub = 0, nWSub = 0;
    PetscCall(build_vw_subspace_is(dm, &isVSub, &isWSub));
    if (isVSub)
      PetscCall(ISGetSize(isVSub, &nVSub));
    if (isWSub)
      PetscCall(ISGetSize(isWSub, &nWSub));
    const PetscBool canNestedVW =
        (PetscBool)(isVSub && isWSub && nVSub > 0 && nWSub > 0);

    if (canNestedVW) {
      PetscCall(PCSetType(subpc, PCFIELDSPLIT));
      PetscCall(PCFieldSplitSetDetectSaddlePoint(subpc, PETSC_FALSE));
      PetscCall(PCFieldSplitSetType(subpc, PC_COMPOSITE_SCHUR));
      PetscCall(
          PCFieldSplitSetSchurFactType(subpc, PC_FIELDSPLIT_SCHUR_FACT_UPPER));
      PetscCall(PCFieldSplitSetIS(subpc, "w", isWSub));
      PetscCall(PCFieldSplitSetIS(subpc, "v", isVSub));

      {
        PetscBool vwSchurUser = PETSC_TRUE;
        if (optionsPrefix && optionsPrefix[0] != '\0')
          PetscCall(PetscOptionsGetBool(NULL, optionsPrefix, "-vw_schur_user",
                                        &vwSchurUser, NULL));
        if (vwSchurUser) {
          Mat Pv = NULL;
          PetscCall(build_velocity_schur_precond_mat(A, dm, &Pv));
          if (Pv) {
            PetscCall(PCFieldSplitSetSchurPre(subpc,
                                              PC_FIELDSPLIT_SCHUR_PRE_USER, Pv));
            PetscCall(MatDestroy(&Pv));
          }
        }
      }

      PetscCall(KSPSetFromOptions(subksp[i]));
      PetscCall(KSPSetUp(subksp[i]));

      PetscInt nInner = 0;
      KSP *inner = NULL;
      PetscCall(PCFieldSplitGetSubKSP(subpc, &nInner, &inner));
      Mat Gads = NULL, Cads = NULL;
      if (isTwoSystem) {
        PetscCall(build_twoform_ads_aux_mats(A, dm, &Gads, &Cads));
      }
      for (PetscInt j = 0; j < nInner; ++j) {
        PC ipc = NULL;
        const char *iprefix = NULL;
        PetscCall(KSPGetPC(inner[j], &ipc));
        PetscCall(KSPGetOptionsPrefix(inner[j], &iprefix));
        // 硬绑定优先：内层 2 个子块时，j==0 视为 w，j==1 视为 v
        // （注册顺序为 w 先、v 后，UPPER Schur 先解 field1=v 再回代 field0=w）
        PetscBool isVInner = PETSC_FALSE;
        if (nInner >= 2) {
          isVInner = (j == 1) ? PETSC_TRUE : PETSC_FALSE;
        } else {
          // 兜底：当内层子块数异常时，再用前缀判定。
          const PetscBool hitNamedV =
              (iprefix && std::strstr(iprefix, "fieldsplit_v_")) ? PETSC_TRUE
                                                                 : PETSC_FALSE;
          const PetscBool hitNumV =
              (iprefix && std::strstr(iprefix, "fieldsplit_0_")) ? PETSC_TRUE
                                                                 : PETSC_FALSE;
          isVInner = (hitNamedV || hitNumV) ? PETSC_TRUE : PETSC_FALSE;
        }
#if DUAL_MAC_DEBUG
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "[DEBUG][LS inner] nInner=%" PetscInt_FMT
                              ", j=%" PetscInt_FMT ", prefix=%s, bind=%s\n",
                              nInner, j, iprefix ? iprefix : "(null)",
                              isVInner ? "v" : "w"));
#endif

        // 显式绑定内层子块前缀，保证 quick_test 中的
        // -half/-one/-two_fieldsplit_u_fieldsplit_[v|w|0|1]_* 可命中。
        if (optionsPrefix && optionsPrefix[0] != '\0') {
          char innerPrefix[256];
          const char *name = isVInner ? "v" : "w";
          PetscCall(PetscSNPrintf(innerPrefix, sizeof(innerPrefix),
                                  "%sfieldsplit_u_fieldsplit_%s_",
                                  optionsPrefix, name));
          PetscCall(KSPSetOptionsPrefix(inner[j], innerPrefix));
        }

        if (isTwoSystem && isVInner) {
          if (adsFallbackToGamg) {
            // 容错开关：ADS 不稳定/失败时，可切换到 GAMG 先保证流程可跑通
            PetscCall(KSPSetType(inner[j], KSPFGMRES));
            PetscCall(PCSetType(ipc, PCGAMG));
            PetscCall(KSPSetFromOptions(inner[j]));
          } else {
            // 默认不硬编码 two/v 组合，先按命令行设置
            PetscCall(KSPSetFromOptions(inner[j]));

            // 若用户确实设置为 hypre(ads)，需要注入 ADS 辅助算子与坐标。
            // KSPSetUp(subksp[i]) 已递归触发 PCSetUp(ipc)，但那时 G/C/坐标
            // 尚未注入。由于 Schur 补 KSP 的算子矩阵是 shell（不可转换为
            // HYPRE 格式），不能简单地 PCReset + PCSetUp。
            // 解决方案：创建全新的 PCHYPRE(ads) PC 替换旧的。
            PetscBool isHypre = PETSC_FALSE;
            PetscCall(
                PetscObjectTypeCompare((PetscObject)ipc, PCHYPRE, &isHypre));
            if (isHypre) {
              const char *hypreType = NULL;
              PetscCall(PCHYPREGetType(ipc, &hypreType));
              const PetscBool isAds =
                  (hypreType && !std::strcmp(hypreType, "ads")) ? PETSC_TRUE
                                                                : PETSC_FALSE;
              if (isAds && (Gads || Cads)) {
                Mat dummy, kspPmat;
                PetscCall(KSPGetOperators(inner[j], &dummy, &kspPmat));

                PC newpc = NULL;
                PetscCall(PCCreate(PetscObjectComm((PetscObject)ipc), &newpc));
                PetscCall(PCSetType(newpc, PCHYPRE));
                PetscCall(PCHYPRESetType(newpc, "ads"));
                PetscCall(PCSetOperators(newpc, kspPmat, kspPmat));
                PetscCall(set_twoform_ads_coordinates(dm, newpc));
                if (Gads)
                  PetscCall(PCHYPRESetDiscreteGradient(newpc, Gads));
                if (Cads)
                  PetscCall(PCHYPRESetDiscreteCurl(newpc, Cads));
                PetscCall(PCSetFromOptions(newpc));
                PetscCall(PCSetUp(newpc));
                PetscCall(KSPSetPC(inner[j], newpc));
                PetscCall(PCDestroy(&newpc));
              }
            }
          }
        } else {
          // 非 two/v 子块也不再硬编码，完全遵循命令行
          PetscCall(KSPSetFromOptions(inner[j]));
        }
      }
      if (Gads)
        PetscCall(MatDestroy(&Gads));
      if (Cads)
        PetscCall(MatDestroy(&Cads));
      PetscCall(PetscFree(inner));
    } else {
      // 保护逻辑：无双场可分时，直接回退到单块预条件。
      // 不硬编码具体求解器/预条件，仅给出一个可运行默认值，
      // 仍可由 quick_test 覆盖。
      PetscCall(PCSetType(subpc, PCASM));
      PetscCall(KSPSetFromOptions(subksp[i]));
    }

    if (isVSub)
      PetscCall(ISDestroy(&isVSub));
    if (isWSub)
      PetscCall(ISDestroy(&isWSub));
  }

  PetscCall(PetscFree(subksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 当不钉压时，给系统附加“常压零空间”：
// - 仅压力分量为常数
// - 速度/涡量分量为0
// 同时从 rhs 中移除零空间分量，保证奇异系统可解。
static PetscErrorCode
attach_pressure_nullspace_if_needed(Mat A, Vec rhs, DM dm,
                                    PetscBool attachNullspace) {
  PetscFunctionBeginUser;
  if (!attachNullspace)
    PetscFunctionReturn(PETSC_SUCCESS);

  IS isU = NULL, isP = NULL;
  Vec basis = NULL;
  MatNullSpace nsp = NULL;
  const PetscInt *pidx = NULL;
  PetscInt nLocalP = 0;
  PetscReal nrm = 0.0;

  PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
  if (!isP) {
    if (isU)
      PetscCall(ISDestroy(&isU));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(MatCreateVecs(A, &basis, NULL));
  PetscCall(VecZeroEntries(basis));
  PetscCall(ISGetLocalSize(isP, &nLocalP));
  PetscCall(ISGetIndices(isP, &pidx));
  if (nLocalP > 0) {
    std::vector<PetscScalar> ones((size_t)nLocalP, 1.0);
    PetscCall(VecSetValues(basis, nLocalP, pidx, ones.data(), INSERT_VALUES));
  }
  PetscCall(ISRestoreIndices(isP, &pidx));
  PetscCall(VecAssemblyBegin(basis));
  PetscCall(VecAssemblyEnd(basis));

  PetscCall(VecNorm(basis, NORM_2, &nrm));
  if (nrm > 0.0)
    PetscCall(VecScale(basis, 1.0 / nrm));

  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)A), PETSC_FALSE, 1,
                               &basis, &nsp));
  PetscCall(MatSetNullSpace(A, nsp));
  PetscCall(MatSetTransposeNullSpace(A, nsp));
  if (rhs)
    PetscCall(MatNullSpaceRemove(nsp, rhs));

#if DUAL_MAC_DEBUG
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "[DEBUG][LS] pressure nullspace attached (pinPressure=false)\n"));
#endif

  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(VecDestroy(&basis));
  if (isU)
    PetscCall(ISDestroy(&isU));
  PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 在两场分裂框架下，向系统矩阵添加 gamma * grad(div(u)) 项。
// 实现方式：利用已有块结构，构造 G=A(u,p), D=A(p,u)，并将 gamma*(G*D) 加到
// A(u,u) 子块。
static PetscErrorCode add_graddiv_term_to_matrix(Mat A, DM dm,
                                                 PetscReal gamma) {
  PetscFunctionBeginUser;
  if (gamma == 0.0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "gamma: %g\n", (double)gamma));
  IS isU = NULL, isP = NULL;
  IS isUAll = NULL;
  Mat G = NULL, D = NULL, GD = NULL;
  const PetscInt *idxUAll = NULL, *idxULocal = NULL;
  PetscInt nUAll = 0, nULocal = 0;

  PetscCall(build_up_fieldsplit_is(dm, &isU, &isP));
  if (!isU || !isP) {
    if (isU)
      PetscCall(ISDestroy(&isU));
    if (isP)
      PetscCall(ISDestroy(&isP));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(
      MatCreateSubMatrix(A, isU, isP, MAT_INITIAL_MATRIX, &G)); // grad block
  PetscCall(
      MatCreateSubMatrix(A, isP, isU, MAT_INITIAL_MATRIX, &D)); // div block
  PetscCall(
      MatMatMult(G, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GD)); // GD on u-u

  // GD 可能引入 A 中原先未预分配/新位置的非零结构，显式允许扩展。
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));

  // 建立全局子空间索引 -> 全局矩阵索引映射
  PetscCall(ISAllGather(isU, &isUAll));
  PetscCall(ISGetSize(isUAll, &nUAll));
  PetscCall(ISGetIndices(isUAll, &idxUAll));
  PetscCall(ISGetLocalSize(isU, &nULocal));
  PetscCall(ISGetIndices(isU, &idxULocal));

  // 仅遍历 GD 的本地拥有行，避免并行 MatGetRow 越界
  PetscInt rstart = 0, rend = 0;
  PetscCall(MatGetOwnershipRange(GD, &rstart, &rend));

  std::vector<PetscInt> gcols;
  std::vector<PetscScalar> gvals;
  for (PetscInt i = rstart; i < rend; ++i) {
    PetscInt ncols = 0;
    const PetscInt *cols = NULL;
    const PetscScalar *vals = NULL;
    const PetscInt iloc = i - rstart;
    if (iloc < 0 || iloc >= nULocal)
      continue;
    const PetscInt grow = idxULocal[iloc];

    PetscCall(MatGetRow(GD, i, &ncols, &cols, &vals));
    gcols.resize((size_t)ncols);
    gvals.resize((size_t)ncols);
    for (PetscInt j = 0; j < ncols; ++j) {
      // cols[j] 是 GD 子空间的全局列号，映射回原矩阵全局列号
      if (cols[j] >= 0 && cols[j] < nUAll)
        gcols[(size_t)j] = idxUAll[cols[j]];
      else
        gcols[(size_t)j] = grow;
      gvals[(size_t)j] = ((PetscScalar)(-gamma)) * vals[j];
    }
    if (ncols > 0)
      PetscCall(MatSetValues(A, 1, &grow, ncols, gcols.data(), gvals.data(),
                             ADD_VALUES));
    PetscCall(MatRestoreRow(GD, i, &ncols, &cols, &vals));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(ISRestoreIndices(isU, &idxULocal));
  PetscCall(ISRestoreIndices(isUAll, &idxUAll));
  PetscCall(ISDestroy(&isUAll));
  PetscCall(MatDestroy(&GD));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&G));
  PetscCall(ISDestroy(&isU));
  PetscCall(ISDestroy(&isP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode get_prefixed_real_option(const char *optionsPrefix,
                                               const char *optName,
                                               PetscReal *value,
                                               PetscBool *found) {
  PetscFunctionBeginUser;
  PetscReal local = 0.0, global = 0.0;
  PetscBool hasLocal = PETSC_FALSE, hasGlobal = PETSC_FALSE;
  if (optionsPrefix && optionsPrefix[0] != '\0')
    PetscCall(
        PetscOptionsGetReal(NULL, optionsPrefix, optName, &local, &hasLocal));
  PetscCall(PetscOptionsGetReal(NULL, NULL, optName, &global, &hasGlobal));
  if (hasLocal) {
    *value = local;
    *found = PETSC_TRUE;
  } else if (hasGlobal) {
    *value = global;
    *found = PETSC_TRUE;
  } else {
    *value = 0.0;
    *found = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode estimate_hx_from_dm(DM dm, PetscReal *hx) {
  PetscFunctionBeginUser;
  PetscInt Nx = 1, Ny = 1, Nz = 1;
  PetscReal xmin = 0.0, xmax = 1.0;
  PetscBool hasXmin = PETSC_FALSE, hasXmax = PETSC_FALSE;
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, &Nz));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-xmin", &xmin, &hasXmin));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-xmax", &xmax, &hasXmax));
  if (!hasXmin || !hasXmax || xmax <= xmin) {
    xmin = 0.0;
    xmax = 1.0;
  }
  *hx = (Nx > 0) ? ((xmax - xmin) / (PetscReal)Nx) : 1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// 基本线性求解器：解 A x = rhs
// - A   : 已经组装完毕的 PETSc 矩阵
// - rhs : 右端项向量
// - sol : 解向量（输出）
// - dm  : DM 对象，用于访问压力分量并进行均值归零处理
// - optionsPrefix : KSP 命令行前缀（如 "one_"、"two_"），为空则使用无前缀
PetscErrorCode solve_linear_system_basic(Mat A, Vec rhs, Vec sol, DM dm,
                                         const char *optionsPrefix,
                                         PetscBool attachPressureNullspace,
                                         PetscReal dt, PetscReal alphaExternal,
                                         PetscReal gammaExternal) {
  PetscFunctionBeginUser;

  Mat Aop = A;
  Mat Aowned = NULL;
  KSP ksp;
  PC pc;
  PetscInt dof0, dof1, dof2, dof3;
  const PetscBool isOneSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "one_", 4) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool isHalfSystem =
      (optionsPrefix && std::strncmp(optionsPrefix, "half_", 5) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool isTwoSystemForSchur =
      (optionsPrefix && std::strncmp(optionsPrefix, "two_", 4) == 0)
          ? PETSC_TRUE
          : PETSC_FALSE;

  PetscReal alphaOneForm = alphaExternal;
  PetscBool hasAlpha = PETSC_FALSE;
  if (isOneSystem || isHalfSystem) {
    PetscReal alphaOpt = 0.0;
    PetscCall(get_prefixed_real_option(optionsPrefix, "-alpha", &alphaOpt,
                                       &hasAlpha));
    if (hasAlpha)
      alphaOneForm = alphaOpt;
  }
  // 显式欧拉启动半步(half_)不添加额外稳定项，仅 one_ 添加。
  if (isOneSystem) {
    PetscReal hx = 1.0;
    PetscCall(estimate_hx_from_dm(dm, &hx));
    const PetscReal oneCoeff = alphaOneForm * ((hx * hx) / 6.0);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "oneCoeff: %g\n", oneCoeff));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aowned));
    Aop = Aowned;
    PetscCall(add_graddiv_term_to_matrix(Aop, dm, oneCoeff));
  }

  // 创建 KSP
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, Aop, Aop));

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
    if (isU)
      PetscCall(PCFieldSplitSetIS(pc, "u", isU));
    if (isP)
      PetscCall(PCFieldSplitSetIS(pc, "p", isP));
    if (isU)
      PetscCall(ISDestroy(&isU));
    if (isP)
      PetscCall(ISDestroy(&isP));
  }

  if (optionsPrefix && optionsPrefix[0] != '\0') {
    PetscCall(KSPSetOptionsPrefix(ksp, optionsPrefix));
  }

  // 从命令行读取 KSP/PC 相关参数，方便调试不同求解器
  PetscCall(KSPSetFromOptions(ksp));

  // two_ 系统可用 gamma 构造压力 Schur 预条件（S_p = -1/gamma * M_p）
  PetscReal gammaLocal = 0.0, gammaGlobal = 0.0,
            gammaForPressureSchur = gammaExternal;
  PetscBool hasGammaLocal = PETSC_FALSE, hasGammaGlobal = PETSC_FALSE;
  if (optionsPrefix && optionsPrefix[0] != '\0')
    PetscCall(PetscOptionsGetReal(NULL, optionsPrefix, "-graddiv_gamma",
                                  &gammaLocal, &hasGammaLocal));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-graddiv_gamma", &gammaGlobal,
                                &hasGammaGlobal));
  gammaForPressureSchur = hasGammaLocal
                              ? gammaLocal
                              : (hasGammaGlobal ? gammaGlobal : gammaExternal);

  // 对 fieldsplit 子块做显式默认配置：
  // 外层 u/p Schur，内层 u->(v,w) 上三角 Schur
  const PetscReal alphaForSchurPre = isHalfSystem ? 0.0 : alphaOneForm;
  PetscCall(configure_fieldsplit_subksp_defaults(
      pc, Aop, dm, optionsPrefix, gammaForPressureSchur, isTwoSystemForSchur,
      alphaForSchurPre, dt));

  // 注意：2-form 的 p3
  // 钉住目前在矩阵组装阶段完成（assemble_u2_divergence_matrix）。 这里不再对
  // dof3 系统重复施加 MatZeroRowsColumnsIS，避免二次改写系统。
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, &dof3));
  PetscCall(attach_pressure_nullspace_if_needed(Aop, rhs, dm,
                                                attachPressureNullspace));

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
  const PetscBool isTwoSystem = isTwoSystemForSchur;
  if (isTwoSystem) {
    Vec r = NULL, localR = NULL;
    PetscScalar ****arrR = NULL;
    PetscReal rhsNorm = 0.0, rNorm = 0.0, relNorm = 0.0;

    PetscCall(VecDuplicate(rhs, &r));
    PetscCall(MatMult(Aop, sol, r));
    PetscCall(VecAXPY(r, -1.0, rhs)); // r = A*sol - rhs
    PetscCall(VecNorm(rhs, NORM_2, &rhsNorm));
    PetscCall(VecNorm(r, NORM_2, &rNorm));
    relNorm = (rhsNorm > 0.0) ? (rNorm / rhsNorm) : rNorm;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "[DEBUG][LS two] residual: ||Ax-b||_2=%.12e, "
                          "||b||_2=%.12e, rel=%.12e\n",
                          (double)rNorm, (double)rhsNorm, (double)relNorm));

    if (dof3 > 0) {
      PetscInt startx, starty, startz, nx, ny, nz, slotElem;
      PetscReal localSqElem = 0.0, localMaxElem = 0.0;
      PetscReal globalSqElem = 0.0, globalMaxElem = 0.0;

      PetscCall(DMGetLocalVector(dm, &localR));
      PetscCall(DMGlobalToLocal(dm, r, INSERT_VALUES, localR));
      PetscCall(DMStagVecGetArrayRead(dm, localR, &arrR));
      PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz,
                                 NULL, NULL, NULL));
      PetscCall(DMStagGetLocationSlot(dm, ELEMENT, 0, &slotElem));

      for (PetscInt ez = startz; ez < startz + nz; ++ez) {
        for (PetscInt ey = starty; ey < starty + ny; ++ey) {
          for (PetscInt ex = startx; ex < startx + nx; ++ex) {
            const PetscReal v =
                PetscAbsReal(PetscRealPart(arrR[ez][ey][ex][slotElem]));
            localSqElem += v * v;
            if (v > localMaxElem)
              localMaxElem = v;
          }
        }
      }

      PetscCall(PMPI_Allreduce(&localSqElem, &globalSqElem, 1, MPIU_REAL,
                               MPIU_SUM, PetscObjectComm((PetscObject)dm)));
      PetscCall(PMPI_Allreduce(&localMaxElem, &globalMaxElem, 1, MPIU_REAL,
                               MPIU_MAX, PetscObjectComm((PetscObject)dm)));

      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
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
                          "[DEBUG][LS] KSPSolve done: its=%" PetscInt_FMT
                          ", reason=%d\n",
                          its, (int)reason));
  }
  PetscCall(
      debug_check_vec_finite_ls(sol, "sol_after_solve_before_pressure_mean"));
#endif

  // 销毁 KSP
  PetscCall(KSPDestroy(&ksp));
  if (Aowned)
    PetscCall(MatDestroy(&Aowned));

  // ===== 压力均值归零处理 =====
  // 这是处理压力不定性的常用方法：求解后减去压力的全局均值
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscCall(DMStagGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz, NULL,
                             NULL, NULL));

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
    PetscMPIInt ierr =
        MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_SCALAR, MPI_SUM, comm);
    if (ierr)
      SETERRQ(comm, PETSC_ERR_LIB, "MPI_Allreduce failed for sum");
  }
  {
    PetscMPIInt ierr =
        MPI_Allreduce(&local_count, &global_count, 1, MPIU_INT, MPI_SUM, comm);
    if (ierr)
      SETERRQ(comm, PETSC_ERR_LIB, "MPI_Allreduce failed for count");
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
PetscErrorCode
solve_linear_system_graddiv(Mat A, Vec rhs, Vec sol, DM dm, PetscReal gamma,
                            const char *optionsPrefix,
                            PetscBool attachPressureNullspace, PetscReal dt,
                            PetscReal alphaExternal, PetscReal gammaExternal) {
  PetscFunctionBeginUser;
  Mat Aeff = NULL;

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Aeff));
  PetscCall(add_graddiv_term_to_matrix(Aeff, dm, gamma));

#if DUAL_MAC_DEBUG
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "[DEBUG][LS graddiv] gamma=%.6e, operator added to u-u block\n",
      (double)gamma));
#endif

  PetscCall(solve_linear_system_basic(Aeff, rhs, sol, dm, optionsPrefix,
                                      attachPressureNullspace, dt,
                                      alphaExternal, gammaExternal));
  PetscCall(MatDestroy(&Aeff));
  PetscFunctionReturn(PETSC_SUCCESS);
}
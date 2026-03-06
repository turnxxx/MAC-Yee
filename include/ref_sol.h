#pragma once
#include <petsc.h>
#include <functional>

namespace detail {
inline PetscScalar zeroFunc(PetscScalar, PetscScalar, PetscScalar, PetscScalar) { return 0.0; }
} // namespace detail

class RefSol
{ // 3D参考解
public:
    using ScalarFunc = std::function<PetscScalar(PetscScalar, PetscScalar, PetscScalar, PetscScalar)>;

    // 默认构造函数：使用零函数
    RefSol()
        : uxRef_func(detail::zeroFunc),
          uyRef_func(detail::zeroFunc),
          uzRef_func(detail::zeroFunc),
          omegaxRef_func(detail::zeroFunc),
          omegayRef_func(detail::zeroFunc),
          omegazRef_func(detail::zeroFunc),
          pRef_func(detail::zeroFunc) {}

    // 构造函数：允许在构造时设置函数
    explicit RefSol(
        ScalarFunc uxRef,
        ScalarFunc uyRef,
        ScalarFunc uzRef,
        ScalarFunc omegaxRef,
        ScalarFunc omegayRef,
        ScalarFunc omegazRef,
        ScalarFunc pRef)
        : uxRef_func(uxRef),
          uyRef_func(uyRef),
          uzRef_func(uzRef),
          omegaxRef_func(omegaxRef),
          omegayRef_func(omegayRef),
          omegazRef_func(omegazRef),
          pRef_func(pRef) {}

    // 调用接口：添加时间参数，默认值为0（用于初值设置）
    PetscScalar uxRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return uxRef_func(x, y, z, t);
    }

    PetscScalar uyRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return uyRef_func(x, y, z, t);
    }

    PetscScalar uzRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return uzRef_func(x, y, z, t);
    }

    PetscScalar omegaxRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return omegaxRef_func(x, y, z, t);
    }

    PetscScalar omegayRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return omegayRef_func(x, y, z, t);
    }

    PetscScalar omegazRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return omegazRef_func(x, y, z, t);
    }

    PetscScalar pRef(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t = 0.0) {
        return pRef_func(x, y, z, t);
    }

    // 设置函数：允许在运行时更改函数定义
    void setUxRef(ScalarFunc func) { uxRef_func = func; }
    void setUyRef(ScalarFunc func) { uyRef_func = func; }
    void setUzRef(ScalarFunc func) { uzRef_func = func; }
    void setOmegaxRef(ScalarFunc func) { omegaxRef_func = func; }
    void setOmegayRef(ScalarFunc func) { omegayRef_func = func; }
    void setOmegazRef(ScalarFunc func) { omegazRef_func = func; }
    void setPRef(ScalarFunc func) { pRef_func = func; }
    
    // 一次性设置所有函数
    void setAll(
        ScalarFunc uxRef,
        ScalarFunc uyRef,
        ScalarFunc uzRef,
        ScalarFunc omegaxRef,
        ScalarFunc omegayRef,
        ScalarFunc omegazRef,
        ScalarFunc pRef) {
        uxRef_func = uxRef;
        uyRef_func = uyRef;
        uzRef_func = uzRef;
        omegaxRef_func = omegaxRef;
        omegayRef_func = omegayRef;
        omegazRef_func = omegazRef;
        pRef_func = pRef;
    }

private:
    ScalarFunc uxRef_func;
    ScalarFunc uyRef_func;
    ScalarFunc uzRef_func;
    ScalarFunc omegaxRef_func;
    ScalarFunc omegayRef_func;
    ScalarFunc omegazRef_func;
    ScalarFunc pRef_func;
};
namespace detail {
inline PetscScalar zeroForceFunc(PetscScalar, PetscScalar, PetscScalar, PetscScalar) { return 0.0; }
} // namespace detail

class ExternalForce {
public:
    using ForceFunc = std::function<PetscScalar(PetscScalar, PetscScalar, PetscScalar, PetscScalar)>;

    // 默认构造函数：使用零函数
    ExternalForce()
        : fx_func(detail::zeroForceFunc),
          fy_func(detail::zeroForceFunc),
          fz_func(detail::zeroForceFunc) {}

    // 构造函数：允许在构造时设置函数
    explicit ExternalForce(ForceFunc fx, ForceFunc fy, ForceFunc fz)
        : fx_func(fx), fy_func(fy), fz_func(fz) {}

    // 调用接口：保持原有接口不变
    PetscScalar fx(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar time) {
        return fx_func(x, y, z, time);
    }

    PetscScalar fy(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar time) {
        return fy_func(x, y, z, time);
    }

    PetscScalar fz(PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar time) {
        return fz_func(x, y, z, time);
    }

    // 设置函数：允许在运行时更改函数定义
    void setFx(ForceFunc func) { fx_func = func; }
    void setFy(ForceFunc func) { fy_func = func; }
    void setFz(ForceFunc func) { fz_func = func; }
    void setAll(ForceFunc fx, ForceFunc fy, ForceFunc fz) {
        fx_func = fx;
        fy_func = fy;
        fz_func = fz;
    }

private:
    ForceFunc fx_func;
    ForceFunc fy_func;
    ForceFunc fz_func;
};
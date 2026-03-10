#include "../include/DUAL_MAC.h"
#include "../include/ref_sol.h"
#include "petscsystypes.h"
#include <cmath>
#include <functional>
#include <limits>
#include <petsc.h>
#include <vector>

// 帮助信息
static char help[] =
    "测试 DUAL_MAC 求解器\n"
    "选项：\n"
    "  -nx <n>          : x 方向网格点数 (默认: 16)\n"
    "  -ny <n>          : y 方向网格点数 (默认: 16)\n"
    "  -nz <n>          : z 方向网格点数 (默认: 16)\n"
    "  -nt <n>          : 时间步数 (默认: 10)\n"
    "  -nu <val>        : 动力粘度 (默认: 0.01)\n"
    "  -tfinal <val>    : 最终时间 (默认: 0.1)\n"
    "  -xmin <val>      : x 方向最小值 (默认: 0.0)\n"
    "  -xmax <val>      : x 方向最大值 (默认: 1.0)\n"
    "  -ymin <val>      : y 方向最小值 (默认: 0.0)\n"
    "  -ymax <val>      : y 方向最大值 (默认: 1.0)\n"
    "  -zmin <val>      : z 方向最小值 (默认: 0.0)\n"
    "  -zmax <val>      : z 方向最大值 (默认: 1.0)\n"
    "  -pinpressure     : 是否固定压力点 (默认: true)\n"
    "  -stab_alpha <a>  : 1-form 稳定化参数 alpha (默认: 1.0)\n"
    "  -stab_gamma <g>  : 2-form 稳定化参数 gamma (默认: 1.0)\n";

int main(int argc, char **argv) {
  PetscFunctionBeginUser;

  // ===== 1. 初始化 PETSc =====
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  // ===== 2. 设置参数（可从命令行读取）=====
  PetscInt baseNx = 4, baseNy = 4, baseNz = 4, baseNt = 4;
  PetscInt refineLevels = 2; // 默认运行3层：N, 2N, 4N
  PetscReal nu = 0.5;        // 动力粘度
  PetscReal tfinal = 0.5;    // 最终时间
  PetscReal xmin = 0.0, xmax = 1.0;
  PetscReal ymin = 0.0, ymax = 1.0;
  PetscReal zmin = 0.0, zmax = 1.0;
  PetscBool pinPressure = PETSC_FALSE;
  PetscReal stabAlpha = 1000.0, stabGamma = 1000.0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &baseNx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &baseNy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &baseNz, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nt", &baseNt, NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-refine_levels", &refineLevels, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nu", &nu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tfinal", &tfinal, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-xmin", &xmin, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-xmax", &xmax, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ymin", &ymin, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ymax", &ymax, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-zmin", &zmin, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-zmax", &zmax, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stab_alpha", &stabAlpha, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stab_gamma", &stabGamma, NULL));
  PetscCall(
      PetscOptionsGetBool(NULL, NULL, "-pinpressure", &pinPressure, NULL));
  if (refineLevels < 1)
    refineLevels = 1;

  // ===== 5. 创建参考解和外部力对象 =====
  // TODO: 用户需要在这里实现参考解和外部力的具体函数
  // 示例：创建默认的零函数对象，用户可以通过 setAll() 或 setUxRef() 等方法设置

  RefSol refSol;
  // ===== 设置参考解函数 =====
  // 方法1：使用 lambda 表达式（推荐）
  // 注意：需要显式构造 std::function，因为某些编译器无法自动转换 lambda
  // 示例：设置一个简单的测试函数（当前为零函数，用户可以修改）
  // 用户只需要修改 return 语句中的表达式即可
  //
  // 示例1：零函数（当前设置）
  RefSol::ScalarFunc ux_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                  PetscScalar t) -> PetscScalar {
    return (2.0 - t) * std::cos(2 * M_PI * z);
  };
  RefSol::ScalarFunc uy_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                  PetscScalar t) -> PetscScalar {
    return (1.0 + t) * std::sin(2 * M_PI * z);
  };
  RefSol::ScalarFunc uz_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                  PetscScalar t) -> PetscScalar {
    return (1.0 - t) * std::sin(2 * M_PI * x);
  };

  RefSol::ScalarFunc omegax_func = [](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
    return -2.0 * M_PI * std::cos(2 * M_PI * z) * (t + 1);
  };
  RefSol::ScalarFunc omegay_func = [](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
    return 2 * M_PI * std::cos(2 * M_PI * x) * (t - 1) +
           2 * M_PI * std::sin(2 * M_PI * z) * (t - 2);
  };
  RefSol::ScalarFunc omegaz_func =
      [](PetscScalar x, PetscScalar y, PetscScalar z,
         PetscScalar t) -> PetscScalar { return 0.0; };
  RefSol::ScalarFunc p_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                 PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return sin(2 * pi * (t + x + y)) +
           (cos(2 * pi * z) * cos(2 * pi * z) * (t - 2) * (t - 2)) / 2 +
           (sin(2 * pi * x) * sin(2 * pi * x) * (t - 1) * (t - 1)) / 2 +
           (sin(2 * pi * z) * sin(2 * pi * z) * (t + 1) * (t + 1)) / 2 -
           ((3 * t * t) / 4 - t + 1.5);
  };
  // 示例2：自定义函数（取消注释并修改）
  // RefSol::ScalarFunc ux_func = std::function<PetscScalar(PetscScalar,
  // PetscScalar, PetscScalar, PetscScalar)>(
  //     +[](PetscScalar x, PetscScalar y, PetscScalar z, PetscScalar t) ->
  //     PetscScalar {
  //         return sin(x) * cos(y) * exp(-t);  // 你的自定义函数表达式
  //     }
  // );

  refSol.setUxRef(ux_func);
  refSol.setUyRef(uy_func);
  refSol.setUzRef(uz_func);
  refSol.setOmegaxRef(omegax_func);
  refSol.setOmegayRef(omegay_func);
  refSol.setOmegazRef(omegaz_func);
  refSol.setPRef(p_func);

  ExternalForce externalForce;
  // TODO: 用户实现外部力
  // 示例：
  // externalForce.setFx([](PetscScalar x, PetscScalar y, PetscScalar z,
  // PetscScalar t) { return ...; }); externalForce.setFy([](PetscScalar x,
  // PetscScalar y, PetscScalar z, PetscScalar t) { return ...; });
  // externalForce.setFz([](PetscScalar x, PetscScalar y, PetscScalar z,
  // PetscScalar t) { return ...; });

  ExternalForce::ForceFunc fx_func = [](PetscScalar x, PetscScalar y,
                                        PetscScalar z,
                                        PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    PetscReal coef = 0.5;
    return 2 * pi * cos(2 * pi * (t + x + y)) - cos(2 * pi * z) -
           sin(2 * pi * x) *
               (2 * pi * cos(2 * pi * x) * (t - 1) +
                2 * pi * sin(2 * pi * z) * (t - 2)) *
               (t - 1) -
           4 * coef * pi * pi * cos(2 * pi * z) * (t - 2) +
           2 * pi * cos(2 * pi * x) * sin(2 * pi * x) * (t - 1) * (t - 1);
  };
  ExternalForce::ForceFunc fy_func = [](PetscScalar x, PetscScalar y,
                                        PetscScalar z,
                                        PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    PetscReal coef = 0.5;
    return sin(2 * pi * z) + 2 * pi * cos(2 * pi * (t + x + y)) +
           4 * coef * pi * pi * sin(2 * pi * z) * (t + 1) -
           2 * pi * cos(2 * pi * z) * sin(2 * pi * x) * (t - 1) * (t + 1);
  };
  ExternalForce::ForceFunc fz_func = [](PetscScalar x, PetscScalar y,
                                        PetscScalar z,
                                        PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    PetscReal coef = 0.5;
    return cos(2 * pi * z) *
               (2 * pi * cos(2 * pi * x) * (t - 1) +
                2 * pi * sin(2 * pi * z) * (t - 2)) *
               (t - 2) -
           sin(2 * pi * x) - 4 * coef * pi * pi * sin(2 * pi * x) * (t - 1) -
           2 * pi * cos(2 * pi * z) * sin(2 * pi * z) * (t - 2) * (t - 2);
  };
  externalForce.setFx(fx_func);
  externalForce.setFy(fy_func);
  externalForce.setFz(fz_func);

  struct LevelErrorData {
    PetscInt level;
    PetscInt Nx, Ny, Nz, Nt;
    PetscReal err_u1, err_omega2, err_p0, err_grad_p0;
    PetscReal err_u2, err_omega1, err_p3, err_grad_p3;
  };
  std::vector<LevelErrorData> allErrors;

  auto safe_rate = [](PetscReal eCoarse, PetscReal eFine) -> PetscReal {
    if (eCoarse > 0.0 && eFine > 0.0)
      return std::log(eCoarse / eFine) / std::log(2.0);
    return std::numeric_limits<PetscReal>::quiet_NaN();
  };

  // ===== 6. 多层加密循环（每层网格步长减半，N翻倍）=====
  for (PetscInt level = 0; level < refineLevels; ++level) {
    const PetscInt scale = (PetscInt)1 << level;
    const PetscInt Nx = baseNx * scale;
    const PetscInt Ny = baseNy * scale;
    const PetscInt Nz = baseNz * scale;
    const PetscInt Nt = baseNt * scale;

    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "========================================\n"
        "DUAL_MAC 求解器测试 (level=%" PetscInt_FMT "/%" PetscInt_FMT ")\n"
        "========================================\n"
        "网格大小: %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT "\n"
        "时间步数: %" PetscInt_FMT "\n"
        "最终时间: %g\n"
        "动力粘度: %g\n"
        "计算域: [%g, %g] x [%g, %g] x [%g, %g]\n"
        "固定压力: %s\n"
        "========================================\n\n",
        level + 1, refineLevels, Nx, Ny, Nz, Nt, (double)tfinal, (double)nu,
        (double)xmin, (double)xmax, (double)ymin, (double)ymax, (double)zmin,
        (double)zmax, pinPressure ? "是" : "否"));

    // ===== 7. 创建求解器 =====
    DUAL_MAC solver(tfinal, nu, Nx, Ny, Nz, Nt, xmin, xmax, ymin, ymax, zmin,
                    zmax, pinPressure);
    PetscCall(solver.set_stabilization_parameters(stabAlpha, stabGamma));

    // ===== 8. 设置网格 =====
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "设置网格...\n"));
    PetscCall(solver.set_dm_periodic());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "网格设置完成\n\n"));

    // ===== 9. 执行求解 =====
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "开始求解...\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "----------------------------------------\n"));

    PetscReal solve_start_time, solve_end_time;
    PetscCall(PetscTime(&solve_start_time));
    PetscCall(solver.solve(refSol, externalForce));
    PetscCall(PetscTime(&solve_end_time));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "----------------------------------------\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "求解完成！耗时: %g 秒\n\n",
                          (double)(solve_end_time - solve_start_time)));

    // ===== 10. 计算误差 =====
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "计算误差...\n"));
    PetscCall(solver.compute_error(tfinal));
    {
      PetscReal mean_p0 = 0.0, mean_p3 = 0.0;
      PetscCall(solver.get_pressure_means(&mean_p0, &mean_p3));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "压力离散均值: mean(p0)=%.12e, mean(p3)=%.12e\n",
                            (double)mean_p0, (double)mean_p3));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "误差计算完成\n\n"));

    LevelErrorData row;
    row.level = level + 1;
    row.Nx = Nx;
    row.Ny = Ny;
    row.Nz = Nz;
    row.Nt = Nt;
    row.err_u1 = solver.get_err_u1();
    row.err_omega2 = solver.get_err_omega2();
    row.err_p0 = solver.get_err_p0();
    row.err_grad_p0 = solver.get_err_grad_p0();
    row.err_u2 = solver.get_err_u2();
    row.err_omega1 = solver.get_err_omega1();
    row.err_p3 = solver.get_err_p3();
    row.err_grad_p3 = solver.get_err_grad_p3();
    allErrors.push_back(row);

    // ===== 11. 清理资源 =====
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "清理资源...\n"));
    PetscCall(solver.destroy());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "资源清理完成\n\n"));
  }

  // ===== 12. 打印收敛率统计 =====
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "========================================\n"
                        "多层加密误差汇总\n"
                        "========================================\n"));
  for (const auto &e : allErrors) {
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "level=%" PetscInt_FMT ", N=(%" PetscInt_FMT ",%" PetscInt_FMT
        ",%" PetscInt_FMT "), Nt=%" PetscInt_FMT "\n"
        "  u1=%.12e, omega2=%.12e, p0=%.12e, grad(p0)=%.12e\n"
        "  u2=%.12e, omega1=%.12e, p3=%.12e, grad(p3)=%.12e\n",
        e.level, e.Nx, e.Ny, e.Nz, e.Nt, (double)e.err_u1, (double)e.err_omega2,
        (double)e.err_p0, (double)e.err_grad_p0, (double)e.err_u2,
        (double)e.err_omega1, (double)e.err_p3, (double)e.err_grad_p3));
  }

  if (allErrors.size() >= 2) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "----------------------------------------\n"
                          "相邻层收敛率 r = log2(E_h / E_{h/2})\n"
                          "----------------------------------------\n"));
    for (size_t i = 1; i < allErrors.size(); ++i) {
      const auto &c = allErrors[i - 1];
      const auto &f = allErrors[i];
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "level %" PetscInt_FMT " -> %" PetscInt_FMT ":\n"
                            "  u1=%.6f, omega2=%.6f, p0=%.6f, grad(p0)=%.6f\n"
                            "  u2=%.6f, omega1=%.6f, p3=%.6f, grad(p3)=%.6f\n",
                            c.level, f.level,
                            (double)safe_rate(c.err_u1, f.err_u1),
                            (double)safe_rate(c.err_omega2, f.err_omega2),
                            (double)safe_rate(c.err_p0, f.err_p0),
                            (double)safe_rate(c.err_grad_p0, f.err_grad_p0),
                            (double)safe_rate(c.err_u2, f.err_u2),
                            (double)safe_rate(c.err_omega1, f.err_omega1),
                            (double)safe_rate(c.err_p3, f.err_p3),
                            (double)safe_rate(c.err_grad_p3, f.err_grad_p3)));
    }
  }

  // ===== 13. 结束 PETSc =====
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

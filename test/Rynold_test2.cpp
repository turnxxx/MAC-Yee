#include "../include/DUAL_MAC.h"
#include "../include/ref_sol.h"
#include <cmath>
#include <functional>
#include <limits>
#include <petsc.h>
#include <vector>

static char help[] =
    "压力幅值 lambda 扫描测试：固定网格、时间步和 nu，逐步增大 lambda\n"
    "选项：\n"
    "  -nx <n>          : 网格点数 (默认: 10)\n"
    "  -nt <n>          : 时间步数 (默认: 5)\n"
    "  -nu <val>        : 动力粘度 (默认: 1)\n"
    "  -lambda <val>    : 基础压力幅值 (默认: 1)\n"
    "  -nlevels <n>     : lambda 扫描层数 (默认: 6，即 lambda*10^0 .. "
    "lambda*10^5)\n"
    "  -tfinal <val>    : 最终时间 (默认: 0.5)\n"
    "  -pinpressure     : 是否固定压力点 (默认: false)\n"
    "  -stab_alpha <a>  : 1-form 稳定化参数 alpha (默认: 1000.0)\n"
    "  -stab_gamma <g>  : 2-form 稳定化参数 gamma (默认: 1000.0)\n";

int main(int argc, char **argv) {
  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscInt Nx = 8, Nt = 5, nLevels = 6;
  PetscReal nu = 1.0;
  PetscReal lambda_base = 1.0;
  PetscReal tfinal = 0.5;
  PetscReal xmin = 0.0, xmax = 1.0;
  PetscReal ymin = 0.0, ymax = 1.0;
  PetscReal zmin = 0.0, zmax = 1.0;
  PetscBool pinPressure = PETSC_FALSE;
  PetscReal stabAlpha = 1000.0, stabGamma = 1000.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &Nx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nt", &Nt, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nlevels", &nLevels, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nu", &nu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &lambda_base, NULL));
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

  const PetscInt Ny = Nx, Nz = Nx;

  RefSol refSol;
  RefSol::ScalarFunc ux_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                  PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return (2 - t) * 0.5 * sin(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z);
  };
  RefSol::ScalarFunc uy_func = [](PetscScalar x, PetscScalar y, PetscScalar z,

                                  PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return (2 - t) * 0.5 * cos(2 * pi * x) * sin(2 * pi * y) * cos(2 * pi * z);
  };
  RefSol::ScalarFunc uz_func = [](PetscScalar x, PetscScalar y, PetscScalar z,
                                  PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return (t - 2) * cos(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z);
  };

  RefSol::ScalarFunc omegax_func = [](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return -2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
               (t - 2) -
           2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
               (t / 2 - 1);
  };
  RefSol::ScalarFunc omegay_func = [](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return 2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
               (t - 2) +
           2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
               (t / 2 - 1);
  };
  RefSol::ScalarFunc omegaz_func = [](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
    auto pi = M_PI;
    return 0.0;
  };

  refSol.setUxRef(ux_func);
  refSol.setUyRef(uy_func);
  refSol.setUzRef(uz_func);
  refSol.setOmegaxRef(omegax_func);
  refSol.setOmegayRef(omegay_func);
  refSol.setOmegazRef(omegaz_func);

  ExternalForce externalForce;

  // p = lam * [sin(2πx)sin(2πy)sin(2πz) + 0.5cos(4πx)sin(2πy) + 0.3sin(6πx)cos(4πy)sin(2πz)]
  // f = du/dt + ω×u + ν∇×ω + ∇p
  auto setup_level = [&refSol, &externalForce](PetscReal nu_val,
                                               PetscReal lam) {
    refSol.setPRef([lam](PetscScalar x, PetscScalar y, PetscScalar z,
                         PetscScalar t) -> PetscScalar {
      auto pi = M_PI;
      return lam * (sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) +
                    0.5 * cos(4 * pi * x) * sin(2 * pi * y) +
                    0.3 * sin(6 * pi * x) * cos(4 * pi * y) * sin(2 * pi * z));
    });

    externalForce.setFx([nu_val, lam](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
      auto pi = M_PI;
      return cos(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z) *
                 (2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
                      (t - 2) +
                  2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
                      (t / 2 - 1)) *
                 (t - 2) -
             (cos(2 * pi * y) * cos(2 * pi * z) * sin(2 * pi * x)) / 2 -
             nu_val * (4 * pi * pi * cos(2 * pi * y) * cos(2 * pi * z) *
                           sin(2 * pi * x) * (t - 2) +
                       4 * pi * pi * cos(2 * pi * y) * cos(2 * pi * z) *
                           sin(2 * pi * x) * (t / 2 - 1)) +
             lam * pi *
                 (2 * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) -
                  2 * sin(4 * pi * x) * sin(2 * pi * y) +
                  1.8 * cos(6 * pi * x) * cos(4 * pi * y) * sin(2 * pi * z));
    });
    externalForce.setFy([nu_val, lam](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
      auto pi = M_PI;
      return cos(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z) *
                 (2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
                      (t - 2) +
                  2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
                      (t / 2 - 1)) *
                 (t - 2) -
             (cos(2 * pi * x) * cos(2 * pi * z) * sin(2 * pi * y)) / 2 -
             nu_val * (4 * pi * pi * cos(2 * pi * x) * cos(2 * pi * z) *
                           sin(2 * pi * y) * (t - 2) +
                       4 * pi * pi * cos(2 * pi * x) * cos(2 * pi * z) *
                           sin(2 * pi * y) * (t / 2 - 1)) +
             lam * pi *
                 (2 * sin(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z) +
                  cos(4 * pi * x) * cos(2 * pi * y) -
                  1.2 * sin(6 * pi * x) * sin(4 * pi * y) * sin(2 * pi * z));
    });
    externalForce.setFz([nu_val, lam](PetscScalar x, PetscScalar y,
                                      PetscScalar z,
                                      PetscScalar t) -> PetscScalar {
      auto pi = M_PI;
      return nu_val * (8 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y) *
                           sin(2 * pi * z) * (t - 2) +
                       8 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y) *
                           sin(2 * pi * z) * (t / 2 - 1)) +
             cos(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z) +
             cos(2 * pi * x) * cos(2 * pi * z) * sin(2 * pi * y) * (t / 2 - 1) *
                 (2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
                      (t - 2) +
                  2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) *
                      (t / 2 - 1)) +
             cos(2 * pi * y) * cos(2 * pi * z) * sin(2 * pi * x) * (t / 2 - 1) *
                 (2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
                      (t - 2) +
                  2 * pi * cos(2 * pi * y) * sin(2 * pi * x) * sin(2 * pi * z) *
                      (t / 2 - 1)) +
             lam * pi *
                 (2 * sin(2 * pi * x) * sin(2 * pi * y) * cos(2 * pi * z) +
                  0.6 * sin(6 * pi * x) * cos(4 * pi * y) * cos(2 * pi * z));
    });
  };

  struct LevelErrorData {
    PetscInt level;
    PetscReal nu, lam;
    PetscReal err_u1, err_omega2, err_p0;
    PetscReal err_u2, err_omega1, err_p3;
  };
  std::vector<LevelErrorData> allErrors;

  // ===== lambda 扫描循环：lam_k = lambda_base * 10^k =====
  for (PetscInt k = 0; k < nLevels; ++k) {
    const PetscReal lam_k = lambda_base * std::pow(10.0, (double)k);
    const PetscReal nu_k = nu * std::pow(10.0, -(double)k);
    PetscCall(PetscPrintf(
        PETSC_COMM_WORLD,
        "========================================\n"
        "lambda 扫描 (level=%" PetscInt_FMT "/%" PetscInt_FMT ")\n"
        "========================================\n"
        "网格大小: %" PetscInt_FMT " x %" PetscInt_FMT " x %" PetscInt_FMT "\n"
        "时间步数: %" PetscInt_FMT "\n"
        "最终时间: %g\n"
        "动力粘度 nu: %.6e\n"
        "压力幅值 lambda: %.6e\n"
        "计算域: [%g, %g] x [%g, %g] x [%g, %g]\n"
        "固定压力: %s\n"
        "========================================\n\n",
        k + 1, nLevels, Nx, Ny, Nz, Nt, (double)tfinal, (double)nu,
        (double)lam_k, (double)xmin, (double)xmax, (double)ymin, (double)ymax,
        (double)zmin, (double)zmax, pinPressure ? "是" : "否"));

    setup_level(nu_k, lambda_base);

    DUAL_MAC solver(tfinal, nu, Nx, Ny, Nz, Nt, xmin, xmax, ymin, ymax, zmin,
                    zmax, pinPressure);
    PetscCall(solver.set_stabilization_parameters(stabAlpha, stabGamma));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "设置网格...\n"));
    PetscCall(solver.set_dm_periodic());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "网格设置完成\n\n"));

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

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "计算误差...\n"));
    PetscCall(solver.compute_error(tfinal));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "误差计算完成\n\n"));

    LevelErrorData row;
    row.level = k + 1;
    row.nu = nu_k;
    row.lam = lam_k;
    row.err_u1 = solver.get_err_u1();
    row.err_omega2 = solver.get_err_omega2();
    row.err_p0 = solver.get_err_p0();
    row.err_u2 = solver.get_err_u2();
    row.err_omega1 = solver.get_err_omega1();
    row.err_p3 = solver.get_err_p3();
    allErrors.push_back(row);

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "清理资源...\n"));
    PetscCall(solver.destroy());
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "资源清理完成\n\n"));
  }

  // ===== 打印 lambda 扫描误差汇总 =====
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "========================================\n"
                        "lambda 扫描误差汇总\n"
                        "网格: %" PetscInt_FMT " x %" PetscInt_FMT
                        " x %" PetscInt_FMT ", Nt=%" PetscInt_FMT
                        ", T=%g, nu=%g\n"
                        "========================================\n"
                        "%12s %5s | %12s %12s %12s | %12s %12s %12s\n",
                        Nx, Ny, Nz, Nt, (double)tfinal, (double)nu, "lambda",
                        "level", "u1", "omega2", "p0", "u2", "omega1", "p3"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "-------------+------+-"
                        "-------------+-------------+-------------+-"
                        "-------------+-------------+-------------\n"));

  for (const auto &e : allErrors) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "%12.2e %5" PetscInt_FMT " | %12.4e %12.4e %12.4e | "
                          "%12.4e %12.4e %12.4e\n",
                          (double)e.lam, e.level, (double)e.err_u1,
                          (double)e.err_omega2, (double)e.err_p0,
                          (double)e.err_u2, (double)e.err_omega1,
                          (double)e.err_p3));
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

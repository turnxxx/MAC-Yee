
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t visc=0.0;

void u_exact(const Vector &xi, Vector &u)
{
   real_t x,y,z;
   x = xi(0);
   y = xi(1);
   z = xi(2);
   
   u(0) = cos(y);
   u(1) = sin(z);
   u(2) = sin(x);
}

void w_exact(const Vector &xi, Vector &w)
{
   real_t x,y,z;
   x = xi(0);
   y = xi(1);
   z = xi(2);
   
   w(0) = -cos(z);
   w(1) = -cos(x);
   w(2) = sin(y);
}

void curlw_exact(const Vector &xi, Vector &curlw)
{
   real_t x,y,z;
   x = xi(0);
   y = xi(1);
   z = xi(2);
   
   curlw(0) = cos(y);
   curlw(1) = sin(z);
   curlw(2) = sin(x);
}

void f_func(const Vector &xi, Vector &f)
{
   real_t x,y,z;
   x = xi(0);
   y = xi(1);
   z = xi(2);
   
   Vector u(3);
   u_exact(xi, u);
   
   Vector curlw(3);
   curlw_exact(xi, curlw);
   
   f(0) = u(0) + curlw(0)*visc;
   f(1) = u(1) + curlw(1)*visc;
   f(2) = u(2) + curlw(2)*visc;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "./mesh/box-3x3x3-cube.mesh";
   int ser_ref_levels = 0;
   int order = 1;

   int precision = 8;
   cout.precision(precision);
   
   StopWatch timer;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");   
   
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }
   
   MFEMInitializePetsc(NULL,NULL,"petsc.opts",NULL);
   
   // Read the serial mesh from the given mesh file on all processors. We can
   // handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   for (int lev = 0; lev < ser_ref_levels; lev++) mesh->UniformRefinement();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   pmesh->PrintInfo();
   delete mesh;
   
   Array<int> ess_bdr_weak({1,1,1,1,1,1});
   Array<int> ess_bdr_strong({0,0,0,0,0,0});
   
   RT_FECollection rt_fec(order-1, dim);
   ND_FECollection nd_fec(order, dim);
   
   ParFiniteElementSpace *rt_fes = new ParFiniteElementSpace(pmesh, &rt_fec);
   ParFiniteElementSpace *nd_fes = new ParFiniteElementSpace(pmesh, &nd_fec);
   
   ParBilinearForm *rt_mass_bf = new ParBilinearForm(rt_fes);
   rt_mass_bf->AddDomainIntegrator(new VectorFEMassIntegrator);
   rt_mass_bf->Assemble();
   rt_mass_bf->Finalize();
   
   ParBilinearForm *nd_mass_bf = new ParBilinearForm(nd_fes);
   nd_mass_bf->AddDomainIntegrator(new VectorFEMassIntegrator);
   nd_mass_bf->Assemble();
   nd_mass_bf->Finalize();
   
   ConstantCoefficient viscous(visc);
   ParMixedBilinearForm *rt_nd_mixed_bf = new ParMixedBilinearForm(nd_fes, rt_fes);
   rt_nd_mixed_bf->AddDomainIntegrator(new MixedVectorCurlIntegrator(viscous));
   rt_nd_mixed_bf->Assemble();
   rt_nd_mixed_bf->Finalize();
   
   ConstantCoefficient m_one(-1.0);
   ParMixedBilinearForm *nd_rt_mixed_bf = new ParMixedBilinearForm(rt_fes, nd_fes);
   nd_rt_mixed_bf->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(m_one));
   nd_rt_mixed_bf->Assemble();
   nd_rt_mixed_bf->Finalize();
   
   HypreParMatrix *UU_mat = rt_mass_bf->ParallelAssemble();
   HypreParMatrix *WW_mat = nd_mass_bf->ParallelAssemble();
   HypreParMatrix *UW_mat = rt_nd_mixed_bf->ParallelAssemble();
   HypreParMatrix *WU_mat = nd_rt_mixed_bf->ParallelAssemble();
   
   ParLinearForm fu_lf(rt_fes);
   VectorFunctionCoefficient fu_coeff(3, f_func);
   fu_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fu_coeff));
   fu_lf.Assemble();
   
   ParLinearForm fw_lf(nd_fes);
   VectorFunctionCoefficient ubdry_coeff(3, u_exact);
   fw_lf.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(ubdry_coeff), ess_bdr_weak);
   fw_lf.Assemble();
   
   ParGridFunction u_gf(rt_fes);
   ParGridFunction w_gf(nd_fes);
   u_gf = 0.0;
   w_gf = 0.0;
   u_gf.ProjectBdrCoefficientNormal(ubdry_coeff, ess_bdr_strong);
   VectorFunctionCoefficient wbdry_coeff(3, w_exact);
   w_gf.ProjectBdrCoefficientTangent(wbdry_coeff, ess_bdr_strong);
   
   Vector u_X(rt_fes->GetTrueVSize());
   Vector w_X(nd_fes->GetTrueVSize());
   u_gf.GetTrueDofs(u_X);
   w_gf.GetTrueDofs(w_X);
   
   // Dirichlet BC
   // rt_nd_mixed_bf->EliminateTestDofs(ess_bdr_strong);
   // nd_rt_mixed_bf->EliminateTestDofs(ess_bdr_strong);
   // nd_rt_mixed_bf->EliminateTrialDofs(ess_bdr_strong, u_gf, fw_lf);
   // rt_nd_mixed_bf->EliminateTrialDofs(ess_bdr_strong, w_gf, fu_lf);
   // rt_mass_bf->EliminateEssentialBC(ess_bdr_strong, u_gf, fu_lf);
   // nd_mass_bf->EliminateEssentialBC(ess_bdr_strong, w_gf, fw_lf);
   
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = rt_fes->GetTrueVSize();
   offsets[2] = rt_fes->GetTrueVSize() + nd_fes->GetTrueVSize();
   
   BlockVector B(offsets);
   BlockVector X(offsets);
   
   fu_lf.ParallelAssemble(B.GetBlock(0));
   fw_lf.ParallelAssemble(B.GetBlock(1));
   X = 0.0;
   
   Array<int> ess_tdofs_rt, ess_tdofs_nd;
   rt_fes->GetEssentialTrueDofs(ess_bdr_strong, ess_tdofs_rt);
   nd_fes->GetEssentialTrueDofs(ess_bdr_strong, ess_tdofs_nd);
   UU_mat->EliminateRows(ess_tdofs_rt);
   UW_mat->EliminateRows(ess_tdofs_rt);
   WU_mat->EliminateRows(ess_tdofs_nd);
   WW_mat->EliminateRows(ess_tdofs_nd);
   
   HypreParMatrix *UWmat_e = UW_mat->EliminateCols(ess_tdofs_nd);
   HypreParMatrix *WUmat_e = WU_mat->EliminateCols(ess_tdofs_rt);
   HypreParMatrix *UUmat_e = UU_mat->EliminateCols(ess_tdofs_rt);
   HypreParMatrix *WWmat_e = WW_mat->EliminateCols(ess_tdofs_nd);
   
   UU_mat->EliminateBC(ess_tdofs_rt, Operator::DIAG_ONE);
   WW_mat->EliminateBC(ess_tdofs_nd, Operator::DIAG_ONE);
   
   UUmat_e->AddMult(u_X, B.GetBlock(0), -1.0);
   WUmat_e->AddMult(u_X, B.GetBlock(1), -1.0);
   UWmat_e->AddMult(w_X, B.GetBlock(0), -1.0);
   WWmat_e->AddMult(w_X, B.GetBlock(1), -1.0);
   
   B.GetBlock(0).SetSubVector(ess_tdofs_rt, u_X);
   B.GetBlock(1).SetSubVector(ess_tdofs_nd, w_X);
   
   BlockOperator *block_op = new BlockOperator(offsets);
   block_op->SetBlock(0, 0, new HypreParMatrix(*UU_mat));
   block_op->SetBlock(0, 1, new HypreParMatrix(*UW_mat));
   block_op->SetBlock(1, 0, new HypreParMatrix(*WU_mat));
   block_op->SetBlock(1, 1, new HypreParMatrix(*WW_mat));
   
   // solver
   PetscLinearSolver *solver = new PetscLinearSolver(pmesh->GetComm(), "", false, true);
   solver->SetOperator(*block_op);
   
   timer.Restart();
   solver->Mult(B, X);
   if (Mpi::Root())
   {
      cout << "Time to solve: " << timer.RealTime() << "s" << endl;
   }
   
   u_gf.SetFromTrueDofs(X.GetBlock(0));
   w_gf.SetFromTrueDofs(X.GetBlock(1));
   
   VectorFunctionCoefficient u_exact_sol(3, u_exact);
   VectorFunctionCoefficient w_exact_sol(3, w_exact);
   VectorFunctionCoefficient curlw_exact_sol(3, curlw_exact);
   ConstantCoefficient zero(0.0);
   
   real_t L2err_u = u_gf.ComputeL2Error(u_exact_sol);
   real_t Hdiverr_u = u_gf.ComputeHDivError(&u_exact_sol, &zero);
   real_t L2err_w = w_gf.ComputeL2Error(w_exact_sol);
   real_t Hcurlerr_w = w_gf.ComputeHCurlError(&w_exact_sol, &curlw_exact_sol);
   
   if(Mpi::Root())
   {
      cout << "L2 error in u: " << L2err_u << endl;
      cout << "Hdiv error in u: " << Hdiverr_u << endl;
      cout << "L2 error in w: " << L2err_w << endl;
      cout << "Hcurl error in w: " << Hcurlerr_w << endl;
   }

   // 13. Free the used memory.
   delete UU_mat;
   delete WW_mat;
   delete UW_mat;
   delete WU_mat;
   delete block_op;
   delete solver;
   delete nd_rt_mixed_bf;
   delete rt_nd_mixed_bf;
   delete nd_mass_bf;
   delete rt_mass_bf;
   delete nd_fes;
   delete rt_fes;
   delete pmesh;
   
   // 14. Finalize MPI and HYPRE.
   MFEMFinalizePetsc();
   Mpi::Finalize();
   Hypre::Finalize();

   return 0;
}
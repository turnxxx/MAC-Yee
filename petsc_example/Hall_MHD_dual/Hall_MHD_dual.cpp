#include "Hall_MHD_dual.hpp"
#include "testcase.hpp"
#include "MHD_solver.hpp"
#include "tools.hpp"

Vector *ExtractVector(const Vector &x, Array<int> offsets, int start, int size)
{
    MFEM_VERIFY(offsets.Size() > start + size, "Invalid component size in ExtractVector.");
    return new Vector(x.GetData() + offsets[start],
                      offsets[start + size] - offsets[start]);
}

Array<real_t> GetPlotTime(const char *time_file)
{
    ifstream time_in(time_file);
    Array<real_t> time;
    real_t t;
    while (time_in >> t)
    {
        time.Append(t);
    }
    time_in.close();

    mfemPrintf("time to plot: size = %d, plot_time list = [", time.Size());
    for (int i = 0; i < time.Size(); i++)
    {
        mfemPrintf(" %f, ", time[i]);
    }
    mfemPrintf("]\n");
    return time;
}

bool CheckPlot(const Array<real_t> &plot_time, real_t t)
{
    static int target = 0;
    real_t eps = 1e-7;

    if (target >= plot_time.Size())
    {
        return false;
    }

    if (t >= plot_time[target] - eps)
    {
        target++;
        return true;
    }

    return false;
}

int main(int argc, char *argv[])
{

    Mpi::Init();
    Hypre::Init();

    OptionsParser args(argc, argv);

    // Mesh info
    int Nx = 4, Ny = 4, Nz = 4;
    real_t Sx = 1.0, Sy = 1.0, Sz = 1.0; // size
    real_t Ax = 0.0, Ay = 0.0, Az = 0.0; // left bottom corner
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;

    args.AddOption(&Nx, "-nx", "--num-elements-x",
                   "Number of elements in the x direction.");
    args.AddOption(&Ny, "-ny", "--num-elements-y",
                   "Number of elements in the y direction.");
    args.AddOption(&Nz, "-nz", "--num-elements-z",
                   "Number of elements in the z direction.");
    args.AddOption(&Sx, "-sx", "--size-x",
                   "Size of the domain in the x direction.");
    args.AddOption(&Sy, "-sy", "--size-y",
                   "Size of the domain in the y direction.");
    args.AddOption(&Sz, "-sz", "--size-z",
                   "Size of the domain in the z direction.");
    args.AddOption(&Ax, "-ax", "--corner-x",
                   "Left bottom corner of the domain in the x direction.");
    args.AddOption(&Ay, "-ay", "--corner-y",
                   "Left bottom corner of the domain in the y direction.");
    args.AddOption(&Az, "-az", "--corner-z",
                   "Left bottom corner of the domain in the z direction.");
    args.AddOption(&periodic_x, "-px", "--periodic-x",
                   "-npx", "--non-periodic-x",
                   "Enable/disable periodic boundary condition in x direction.");
    args.AddOption(&periodic_y, "-py", "--periodic-y",
                   "-npy", "--non-periodic-y",
                   "Enable/disable periodic boundary condition in y direction.");
    args.AddOption(&periodic_z, "-pz", "--periodic-z",
                   "-npz", "--non-periodic-z",
                   "Enable/disable periodic boundary condition in z direction.");

    MHDSolverInfo solver_info;
    solver_info.order = 1;
    solver_info.dim = 3;
    solver_info.dt = 0.01;
    solver_info.t_final = 0.1;
    solver_info.Hall = false;
    solver_info.viscosity = true;
    solver_info.resistivity = true;
    solver_info.H1space = nullptr;
    solver_info.NDspace = nullptr;
    solver_info.RTspace = nullptr;
    solver_info.L2space = nullptr;

    args.AddOption(&solver_info.order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&solver_info.dt, "-dt", "--time-step",
                   "Time step size.");
    args.AddOption(&solver_info.t_final, "-tf", "--final-time",
                   "Final time.");
    args.AddOption(&solver_info.Hall, "-Hall", "--Hall",
                   "-noHall", "--no-Hall",
                   "Enable/disable Hall term.");
    args.AddOption(&solver_info.viscosity, "-visc", "--viscosity",
                   "-novisc", "--no-viscosity",
                   "Enable/disable viscosity.");
    args.AddOption(&solver_info.resistivity, "-resist", "--resistivity",
                   "-noresist", "--no-resistivity",
                   "Enable/disable resistivity.");

    // linear solver info
    LinearSolverInfo lin_sol_int;
    lin_sol_int.type = MFEM;
    lin_sol_int.gamma = 1000.0;
    lin_sol_int.iterative_mode = false;

    LinearSolverInfo lin_sol_half;
    lin_sol_half.type = MFEM;
    lin_sol_half.gamma = 1000.0;
    lin_sol_half.iterative_mode = false;

    real_t rtol = 1e-10;
    real_t atol = 1e-12;
    int maxit = 500;

    // todo: set from options
    real_t sub_pc_rtol = 1e-2;
    real_t sub_pc_atol = 1e-3;
    int sub_pc_maxit = 100;
    int print_level = 3;
    SolverType mag_type = MFEM;

    const char *petsc_opts_file = "petsc-opts.dat";

    args.AddOption(&lin_sol_int.gamma, "-gammaint", "--gamma-integer", "Penalty parameter used in integer time steps.");
    args.AddOption(&lin_sol_half.gamma, "-gammahalf", "--gamma-half", "Penalty parameter used in half time steps.");
    args.AddOption(&lin_sol_int.iterative_mode, "-iterint", "--iterative-integer",
                   "-noiterint", "--no-iterative-integer",
                   "Enable/disable iterative mode for integer time steps.");
    args.AddOption(&lin_sol_half.iterative_mode, "-iterhalf", "--iterative-half",
                     "-noiterhalf", "--no-iterative-half",
                     "Enable/disable iterative mode for half time steps.");
    args.AddOption(&rtol, "-rtol", "--relative-tolerance",
                   "Relative tolerance for the iterative solver.");
    args.AddOption(&atol, "-atol", "--absolute-tolerance",
                   "Absolute tolerance for the iterative solver.");
    args.AddOption(&maxit, "-maxit", "--max-iterations",
                   "Maximum number of iterations for the iterative solver.");
    args.AddOption((int *)&lin_sol_int.type, "-stint", "--solver-type-integer",
                   "Solver type for integer time steps.");
    args.AddOption((int *)&lin_sol_half.type, "-sthalf", "--solver-type-half",
                   "Solver type for half time steps.");
    args.AddOption(&petsc_opts_file, "-petsc-opts", "--petsc-options-file",
                   "File containing PETSc options.");

    // AMR info
    AMRInfo amr_info;

    amr_info.amr = false;
    amr_info.max_amr_iter_init = 10;
    amr_info.max_elements_init = 1000;
    amr_info.refine_frac_init = 0.9;
    amr_info.coarse_frac_init = 0.3;
    amr_info.max_amr_iter = 3;
    amr_info.refine_frac = 0.9;
    amr_info.coarse_frac = 0.3;
    amr_info.max_elements = 5000;
    amr_info.total_err_goal = 1e-3;

    args.AddOption(&amr_info.amr, "-amr", "--adaptive-mesh-refinement",
                   "-noamr", "--no-adaptive-mesh-refinement",
                   "Enable/disable adaptive mesh refinement.");
    args.AddOption(&amr_info.max_amr_iter_init, "-maxamrinit", "--max-amr-iter-init",
                   "Maximum number of iterations for adaptive mesh refinement in the initial step.");
    args.AddOption(&amr_info.max_elements_init, "-maxelinit", "--max-elements-init",
                   "Maximum number of elements for adaptive mesh refinement in the initial step.");
    args.AddOption(&amr_info.refine_frac_init, "-refinefracinit", "--refine-frac-init",
                   "Refinement fraction for adaptive mesh refinement in the initial step.");
    args.AddOption(&amr_info.coarse_frac_init, "-coarsefracinit", "--coarse-frac-init",
                   "Coarsening fraction for adaptive mesh refinement in the initial step.");
    args.AddOption(&amr_info.max_amr_iter, "-maxamr", "--max-amr-iter",
                   "Maximum number of iterations for adaptive mesh refinement.");
    args.AddOption(&amr_info.refine_frac, "-refinefrac", "--refine-frac",
                   "Refinement fraction for adaptive mesh refinement.");
    args.AddOption(&amr_info.coarse_frac, "-coarsefrac", "--coarse-frac",
                   "Coarsening fraction for adaptive mesh refinement.");
    args.AddOption(&amr_info.max_elements, "-maxel", "--max-elements",
                   "Maximum number of elements for adaptive mesh refinement.");
    args.AddOption(&amr_info.total_err_goal, "-errgoal", "--total-error-goal",
                   "Total error goal for adaptive mesh refinement.");

    // output info
    const char *plot_time_file = "./mesh/time_to_plot.dat";
    bool visualization = false;
    const char *output_file_dir = "output";

    args.AddOption(&output_file_dir, "-od", "--output-dir",
                   "Directory to save output files.");
    args.AddOption(&plot_time_file, "-pt", "--plot-time-file",
                   "File containing time to plot.");
    args.AddOption(&visualization, "-vis", "--visualization",
                   "-novis", "--no-visualization",
                   "Enable/disable ParaView visualization.");

    // problem info
    ProblemType test_case = SPATIAL;
    ParamList param;
    param.Re = 1.0;
    param.Rm = 1.0;
    param.s = 1.0;
    param.RH = 1.0;

    real_t Whistler_m = 1.0;

    args.AddOption((int *)&test_case, "-tc", "--test-case",
                   "Test case: 0 for temporal error, 1 for spatial error.");
    args.AddOption(&param.Re, "-Re", "--Reynolds-number",
                   "Reynolds number.");
    args.AddOption(&param.Rm, "-Rm", "--Magnetic-Reynolds-number",
                   "Magnetic Reynolds number.");
    args.AddOption(&param.s, "-s", "--coupling-parameter",
                   "Coupling parameter.");
    args.AddOption(&param.RH, "-RH", "--Hall-parameter",
                   "Hall parameter.");
    args.AddOption(&Whistler_m, "-WhistlerM", "--Whistler-M",
                   "Magnetic field number for Whistler wave test case.");

    // debug info
    bool debug = false;
    args.AddOption(&debug, "-debug", "--debug",
                   "-nodebug", "--no-debug",
                   "Enable/disable debug mode.");

    args.Parse();
    if (!args.Good())
    {
        if (Mpi::Root())
            args.PrintUsage(cout);
        return 1;
    }
    if (Mpi::Root())
        args.PrintOptions(cout);

    lin_sol_int.rtol = rtol;
    lin_sol_int.atol = atol;
    lin_sol_int.maxit = maxit;
    lin_sol_int.print_level = print_level;
    lin_sol_int.sub_pc_rtol = sub_pc_rtol;
    lin_sol_int.sub_pc_atol = sub_pc_atol;
    lin_sol_int.sub_pc_maxit = sub_pc_maxit;
    lin_sol_int.mag_type = mag_type;

    lin_sol_half.rtol = rtol;
    lin_sol_half.atol = atol;
    lin_sol_half.maxit = maxit;
    lin_sol_half.print_level = print_level;
    lin_sol_half.sub_pc_rtol = sub_pc_rtol;
    lin_sol_half.sub_pc_atol = sub_pc_atol;
    lin_sol_half.sub_pc_maxit = sub_pc_maxit;
    lin_sol_half.mag_type = mag_type;

    if (debug)
    {
        int i_debugging = 1;
        MPI_Barrier(MPI_COMM_WORLD);
        printf("pid %d:\n", getpid());
        if (Mpi::Root())
        {
            printf("pid of root: %d \n", getpid());
            while (i_debugging)
            {
                sleep(1);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    /*   Main program   */
    {

        MFEMInitializePetsc(NULL, NULL, petsc_opts_file, NULL);

        real_t t = 0.0;
        char energy_integer_file[100];
        char energy_half_file[100];
        char m_helicity1_file[100];
        char m_helicity2_file[100];
        char f_helicity1_file[100];
        char f_helicity2_file[100];
        char iter_integer_file[100];
        char iter_half_file[100];
        char divergence_integer_file[100];
        char divergence_half_file[100];
        char primal_dual_error_file[100];
        char current_peak_integer_file[100];
        char current_peak_half_file[100];
        char opoint_integer_file[100];
        char opoint_half_file[100];
        char paraview_dir[100];

        sprintf(energy_integer_file, "%s/energy_integer.dat", output_file_dir);
        sprintf(energy_half_file, "%s/energy_half.dat", output_file_dir);
        sprintf(m_helicity1_file, "%s/m_helicity1.dat", output_file_dir);
        sprintf(m_helicity2_file, "%s/m_helicity2.dat", output_file_dir);
        sprintf(f_helicity1_file, "%s/f_helicity1.dat", output_file_dir);
        sprintf(f_helicity2_file, "%s/f_helicity2.dat", output_file_dir);
        sprintf(iter_integer_file, "%s/iter_integer.dat", output_file_dir);
        sprintf(iter_half_file, "%s/iter_half.dat", output_file_dir);
        sprintf(divergence_integer_file, "%s/divergence_integer.dat", output_file_dir);
        sprintf(divergence_half_file, "%s/divergence_half.dat", output_file_dir);
        sprintf(primal_dual_error_file, "%s/primal_dual_error.dat", output_file_dir);
        sprintf(current_peak_integer_file, "%s/current_peak_integer.dat", output_file_dir);
        sprintf(current_peak_half_file, "%s/current_peak_half.dat", output_file_dir);
        sprintf(opoint_integer_file, "%s/opoint_integer.dat", output_file_dir);
        sprintf(opoint_half_file, "%s/opoint_half.dat", output_file_dir);
        sprintf(paraview_dir, "%s/paraview", output_file_dir);

        mfemPrintf("energy_integer_file: %s\n", energy_integer_file);
        mfemPrintf("energy_half_file: %s\n", energy_half_file);
        mfemPrintf("m_helicity1_file: %s\n", m_helicity1_file);
        mfemPrintf("m_helicity2_file: %s\n", m_helicity2_file);
        mfemPrintf("f_helicity1_file: %s\n", f_helicity1_file);
        mfemPrintf("f_helicity2_file: %s\n", f_helicity2_file);
        mfemPrintf("iter_integer_file: %s\n", iter_integer_file);
        mfemPrintf("iter_half_file: %s\n", iter_half_file);
        mfemPrintf("divergence_integer_file: %s\n", divergence_integer_file);
        mfemPrintf("divergence_half_file: %s\n", divergence_half_file);
        mfemPrintf("primal_dual_error_file: %s\n", primal_dual_error_file);
        mfemPrintf("current_peak_integer_file: %s\n", current_peak_integer_file);
        mfemPrintf("current_peak_half_file: %s\n", current_peak_half_file);
        mfemPrintf("opoint_integer_file: %s\n", opoint_integer_file);
        mfemPrintf("opoint_half_file: %s\n", opoint_half_file);
        mfemPrintf("paraview_dir: %s\n", paraview_dir);

        Array<real_t> plot_time = GetPlotTime(plot_time_file);

        // get problem data
        ProblemData *pd;
        switch (test_case)
        {
        case TEMPORAL:
            pd = GetTemporalProblemData(param, solver_info.Hall, solver_info.viscosity, solver_info.resistivity);
            break;
        case SPATIAL:
            pd = GetSpatialProblemData(param, solver_info.Hall, solver_info.viscosity, solver_info.resistivity);
            break;
        case SPATIAL2:
            pd = GetSpatial2ProblemData(param, solver_info.Hall, solver_info.viscosity, solver_info.resistivity);
            break;
        case SPATIALTEST:
            pd = GetSpatialTestProblemData(param, solver_info.Hall, solver_info.viscosity, solver_info.resistivity);
            break;
        case CONSERVATION:
            pd = GetConservationProblemData(param);
            break;
        case ISLAND:
            pd = GetIslandProblemData(param);
            break;
        case ADAPTIVE:
            pd = GetAdaptiveProblemData(param);
            break;
        case ISLAND_GUO:
            pd = GetIslandGuoProblemData(param);
            break;
        case WHISTLER:
            pd = GetWhistlerProblemData(param, Whistler_m);
            break;
        case MHD_VORTEX:
            pd = GetMHDVORTEXProblemData(param);
            break;
        case LOOP:
            pd = GetLoopProblemData(param);
            break;
        case ORSZAG_TANG:
            pd = GetOrszagTangProblemData(param);
            break;
        case ORSZAG_TANG_KRAUS:
            pd = GetOrszagTangKrausProblemData(param);
            break;
        case TG_VORTEX:
            pd = GetTGVORTEXProblemData(param, solver_info.viscosity);
            break;
        case HOPF:
            pd = GetHopfProblemData(param);
            break;
        case ORSZAG_TANG_3D:
            pd = GetOrszagTang3DProblemData(param);
            break;
        case FRICTION:
            pd = GetFrictionProblemData(param);
            break;
        case ISLAND_ORIGIN:
            pd = GetIslandOriginProblemData(param);
            break;
        case ISLAND_ADLER:
            pd = GetIslandAdlerProblemData(param);
            break;
        default:
            MFEM_ABORT("Unknown test case");
            break;
        }

        /* prepare the mesh */
        Mesh mesh = Mesh::MakeCartesian3D(Nx, Ny, Nz, Element::HEXAHEDRON, Sx, Sy, Sz);
        // shift the mesh
        Vector displacements(mesh.SpaceDimension() * mesh.GetNV());
        for (int i = 0; i < mesh.GetNV(); i++)
        {
            displacements(0 * mesh.GetNV() + i) = Ax;
            displacements(1 * mesh.GetNV() + i) = Ay;
            displacements(2 * mesh.GetNV() + i) = Az;
        }
        mesh.MoveVertices(displacements);

        if (periodic_x || periodic_y || periodic_z)
        {
            Vector x_translation({Sx, 0.0, 0.0});
            Vector y_translation({0.0, Sy, 0.0});
            Vector z_translation({0.0, 0.0, Sz});
            std::vector<Vector> translations;
            if (periodic_x)
                translations.push_back(x_translation);
            if (periodic_y)
                translations.push_back(y_translation);
            if (periodic_z)
                translations.push_back(z_translation);
            mesh = Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations));
        }

        if (amr_info.amr)
        {

            Array<int> ordering;
            mesh.GetHilbertElementOrdering(ordering);
            mesh.ReorderElements(ordering);

            mesh.EnsureNCMesh(true);
            mesh.Finalize(true);
        }

        // refine the parallel mesh
        ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        pmesh->PrintInfo();

        int n_steps = int(solver_info.t_final / solver_info.dt);
        solver_info.dt = solver_info.t_final / n_steps;
        mfemPrintf("n_steps = %d, dt = %lg\n", n_steps, solver_info.dt);

        MHD_solver mhd_solver(pd, pmesh, solver_info, lin_sol_int, lin_sol_half, amr_info, visualization);

        mhd_solver.Init();

        real_t t_int = 0.0;
        real_t t_half = 0.0;

        // energy
        real_t Energy_int, Energy_half;
        real_t Energy_int_kinetic, Energy_int_magnetic;
        real_t Energy_half_kinetic, Energy_half_magnetic;
        ofstream energy_integer_out(energy_integer_file);
        energy_integer_out.precision(16);
        ofstream energy_half_out(energy_half_file);
        energy_half_out.precision(16);
        mhd_solver.CalcEnergy(*mhd_solver.gfs.u2_gf, *mhd_solver.gfs.B2_gf, Energy_int_kinetic, Energy_int_magnetic);
        mhd_solver.CalcEnergy(*mhd_solver.gfs.u1_gf, *mhd_solver.gfs.B1_gf, Energy_half_kinetic, Energy_half_magnetic);
        Energy_int = Energy_int_kinetic + Energy_int_magnetic;
        Energy_half = Energy_half_kinetic + Energy_half_magnetic;
        energy_integer_out << t_int << " " << Energy_int << ' ' << Energy_int_kinetic << ' ' << Energy_int_magnetic << endl;
        energy_half_out << t_half << " " << Energy_half << ' ' << Energy_half_kinetic << ' ' << Energy_half_magnetic << endl;

        // helicity
        ofstream m_helicity1_out(m_helicity1_file);
        m_helicity1_out.precision(16);        
        ofstream m_helicity2_out(m_helicity2_file);
        m_helicity2_out.precision(16);
        ofstream f_helicity1_out(f_helicity1_file);
        f_helicity1_out.precision(16);
        ofstream f_helicity2_out(f_helicity2_file);
        f_helicity2_out.precision(16);
        real_t m_helicity_1, m_helicity_2;
        real_t f_helicity_1, f_helicity_2;
        m_helicity_1 = 0.0;
        m_helicity_2 = 0.0;
        f_helicity_1 = 0.0;
        f_helicity_2 = 0.0;
        
        // iteration number
        ofstream iter_integer_out(iter_integer_file);
        iter_integer_out.precision(16);
        ofstream iter_half_out(iter_half_file);
        iter_half_out.precision(16);
        
        // divergence
        ofstream divergence_integer_out(divergence_integer_file);
        divergence_integer_out.precision(16);
        ofstream divergence_half_out(divergence_half_file);
        divergence_half_out.precision(16);
        
        ofstream primal_dual_error_out(primal_dual_error_file);
        primal_dual_error_out.precision(16);

        // if island problem, print current and vector potential
        char current_file_integer[100];
        char current_file_half[100];
        char A_file_integer[100];
        char A_file_half[100];
        sprintf(current_file_integer, "%s/current_integer.dat", output_file_dir);
        sprintf(current_file_half, "%s/current_half.dat", output_file_dir);
        sprintf(A_file_integer, "%s/A_integer.dat", output_file_dir);
        sprintf(A_file_half, "%s/A_half.dat", output_file_dir);
        ofstream current_out_integer(current_file_integer);
        current_out_integer.precision(16);
        ofstream current_out_half(current_file_half);
        current_out_half.precision(16);
        ofstream A_out_integer(A_file_integer);
        A_out_integer.precision(16);
        ofstream A_out_half(A_file_half);
        A_out_half.precision(16);
        Vector XPoint(3);
        XPoint(0) = Ax + 0.5 * Sx;
        XPoint(1) = Ay + 0.5 * Sy;
        XPoint(2) = Az + 0.5 * Sz;
        ofstream opoint_integer_out(opoint_integer_file);
        opoint_integer_out.precision(16);
        ofstream opoint_half_out(opoint_half_file);
        opoint_half_out.precision(16);
        Vector LeftOpoint_int(3);
        Vector LeftOpoint_half(3);
        LeftOpoint_int(0) = Ax + 0.25 * Sx;
        LeftOpoint_int(1) = Ay + 0.5 * Sy;
        LeftOpoint_int(2) = Az + 0.5 * Sz;
        LeftOpoint_half(0) = Ax + 0.25 * Sx;
        LeftOpoint_half(1) = Ay + 0.5 * Sy;
        LeftOpoint_half(2) = Az + 0.5 * Sz;
        Vector RightOpoint_int(3);
        Vector RightOpoint_half(3);
        RightOpoint_int(0) = Ax + 0.75 * Sx;
        RightOpoint_int(1) = Ay + 0.5 * Sy;
        RightOpoint_int(2) = Az + 0.5 * Sz;
        RightOpoint_half(0) = Ax + 0.75 * Sx;
        RightOpoint_half(1) = Ay + 0.5 * Sy;
        RightOpoint_half(2) = Az + 0.5 * Sz;
        {
            Vector CurrentValue(3);
            EvaluateVectorGFAtPoint(*mhd_solver.gfs.j1_gf, XPoint, CurrentValue);
            current_out_integer << t_int << " " << CurrentValue(0) << " " << CurrentValue(1) << " " << CurrentValue(2) << endl;
            EvaluateVectorGFAtPoint(*mhd_solver.gfs.j2_gf, XPoint, CurrentValue);
            current_out_half << t_half << " " << CurrentValue(0) << " " << CurrentValue(1) << " " << CurrentValue(2) << endl;

            Vector AValue(3);
            EvaluateVectorGFAtPoint(*mhd_solver.gfs.A1_gf, XPoint, AValue);
            A_out_integer << t_int << " " << AValue(0) << " " << AValue(1) << " " << AValue(2) << endl;
            EvaluateVectorGFAtPoint(*mhd_solver.gfs.A2_gf, XPoint, AValue);
            A_out_half << t_half << " " << AValue(0) << " " << AValue(1) << " " << AValue(2) << endl;
            
            opoint_half_out << t_half << " " << LeftOpoint_half(0) << " " << LeftOpoint_half(1) << " " << LeftOpoint_half(2) << " " << RightOpoint_half(0) << " " << RightOpoint_half(1) << " " << RightOpoint_half(2) << endl;
            opoint_integer_out << t_int << " " << LeftOpoint_int(0) << " " << LeftOpoint_int(1) << " " << LeftOpoint_int(2) << " " << RightOpoint_int(0) << " " << RightOpoint_int(1) << " " << RightOpoint_int(2) << endl;
        }

        // if Whistler problem, print components of B
        char B_file_integer[100];
        char B_file_half[100];
        sprintf(B_file_integer, "%s/B_integer.dat", output_file_dir);
        sprintf(B_file_half, "%s/B_half.dat", output_file_dir);
        ofstream B_out_integer(B_file_integer);
        B_out_integer.precision(16);
        ofstream B_out_half(B_file_half);
        B_out_half.precision(16);
        Vector BPoint(3);
        BPoint = XPoint;
        {
            // real_t B0, B1, B2;
            // B0 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 0));
            // B1 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 1));
            // B2 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 2));
            // B_out_integer << t_int << " " << B0 << " " << B1 << " " << B2 << endl;

            // B0 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 0));
            // B1 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 1));
            // B2 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 2));
            // B_out_half << t_half << " " << B0 << " " << B1 << " " << B2 << endl;

            Vector BValue(3);
            EvaluateVectorGFAtPoint(*mhd_solver.gfs.B2_gf, BPoint, BValue);
            B_out_integer << t_int << " " << BValue(0) << " " << BValue(1) << " " << BValue(2) << endl;

            EvaluateVectorGFAtPoint(*mhd_solver.gfs.B1_gf, BPoint, BValue);
            B_out_half << t_half << " " << BValue(0) << " " << BValue(1) << " " << BValue(2) << endl;
        }
        
        // output current peak
        ofstream current_peak_half_out(current_peak_half_file);
        current_peak_half_out.precision(16);
        ofstream current_peak_integer_out(current_peak_integer_file);
        current_peak_integer_out.precision(16);
        
        real_t current_peak_half = 0.0;
        real_t current_peak_integer = 0.0;
        
        Vector zerovec(3);
        zerovec = 0.0;
        VectorConstantCoefficient zerovec_coeff(zerovec);
        current_peak_half = mhd_solver.gfs.j2_gf->ComputeMaxError(zerovec_coeff);
        current_peak_integer = mhd_solver.gfs.j1_gf->ComputeMaxError(zerovec_coeff);
        current_peak_half_out << t_half << " " << current_peak_half << endl;
        current_peak_integer_out << t_int << " " << current_peak_integer << endl;

        // initial visualization
        int i_plot = 0;
        if (visualization)
        {
            mhd_solver.SetupParaview(paraview_dir);
            mhd_solver.OutputParaview(paraview_dir, i_plot, 0.0);
        }
        i_plot++;

        // total iteration number
        int n_iter_int = 0;
        int n_iter_half = 0;
        
        bool interp_half = false;
        bool interp_int = false;

        for (int i = 0; i < n_steps; i++)
        {

            VectorFunctionCoefficient ucoeff(3, pd->u_fun);
            VectorFunctionCoefficient wcoeff(3, pd->w_fun);
            FunctionCoefficient pcoeff(pd->p_fun);
            VectorFunctionCoefficient Acoeff(3, pd->A_fun);
            VectorFunctionCoefficient Bcoeff(3, pd->B_fun);
            VectorFunctionCoefficient jcoeff(3, pd->j_fun);

            if(!interp_half)
            {
                bool update_operators = (i > 1) ? false : true;
                n_iter_half = mhd_solver.HalfStep(t_half, (i == 0) ? (0.5 * solver_info.dt) : solver_info.dt, 0.5, update_operators);
                iter_half_out << t_half << " " << n_iter_half << endl;
            }
            else
            {
                t_half += (i == 0 ? 0.5 : 1.0) * solver_info.dt;
                ucoeff.SetTime(t_half);
                mhd_solver.gfs.u1_gf->ProjectCoefficient(ucoeff);
                wcoeff.SetTime(t_half);
                mhd_solver.gfs.w2_gf->ProjectCoefficient(wcoeff);
                pcoeff.SetTime(t_half);
                mhd_solver.gfs.p0_gf->ProjectCoefficient(pcoeff);
                Acoeff.SetTime(t_half);
                mhd_solver.gfs.A2_gf->ProjectCoefficient(Acoeff);
                Bcoeff.SetTime(t_half);
                mhd_solver.gfs.B1_gf->ProjectCoefficient(Bcoeff);
                jcoeff.SetTime(t_half);
                mhd_solver.gfs.j2_gf->ProjectCoefficient(jcoeff);
            }
            
            // output divergence
            {
                real_t divJ = CheckDivergenceFree(mhd_solver.gfs.j2_gf);
                divergence_half_out << t_half << " " << divJ << endl;
            }
            
            current_peak_half = mhd_solver.gfs.j2_gf->ComputeMaxError(zerovec_coeff);
            current_peak_half_out << t_half << " " << current_peak_half << endl;
             
            mhd_solver.CalcEnergy(*mhd_solver.gfs.u1_gf, *mhd_solver.gfs.B1_gf, Energy_half_kinetic, Energy_half_magnetic);
            
            
            {
                real_t dt = (i == 0 ? 0.5 : 1.0) * solver_info.dt;
                ParGridFunction w2_mid(*mhd_solver.gfs.w2_gf);
                w2_mid += *mhd_solver.old_gfs.w2_gf;
                w2_mid *= 0.5;
                ParGridFunction j2_mid(*mhd_solver.gfs.j2_gf);
                j2_mid += *mhd_solver.old_gfs.j2_gf;
                j2_mid *= 0.5;
                real_t energy_half_var = (solver_info.viscosity? -dt / pd->param.Re * mhd_solver.GFInnerProduct(w2_mid, w2_mid) : 0.0) 
                + (solver_info.resistivity? -dt *pd->param.s / pd->param.Rm * mhd_solver.GFInnerProduct(j2_mid, j2_mid) : 0.0);
                
                Energy_half = Energy_half_kinetic + Energy_half_magnetic;
                energy_half_out << t_half << " " << Energy_half << ' ' << Energy_half_kinetic << ' ' << Energy_half_magnetic << ' ' <<energy_half_var<< endl;
            }
            
             // output helicity
             if(i > 0)
             {
                ParGridFunction B1_mid(*mhd_solver.gfs.B1_gf);
                B1_mid += *mhd_solver.old_gfs.B1_gf;
                B1_mid *= 0.5;
                
                ParGridFunction A2_mid(*mhd_solver.gfs.A2_gf);
                A2_mid += *mhd_solver.old_gfs.A2_gf;
                A2_mid *= 0.5;
                
                ParGridFunction u1_mid(*mhd_solver.gfs.u1_gf);
                u1_mid += *mhd_solver.old_gfs.u1_gf;
                u1_mid *= 0.5;
                
                ParGridFunction w2_mid(*mhd_solver.gfs.w2_gf);
                w2_mid += *mhd_solver.old_gfs.w2_gf;
                w2_mid *= 0.5;
                
                // just for fixing magnetic helicity
                // real_t A1dotB1plus = mhd_solver.GFInnerProduct(*mhd_solver.gfs.A1_gf, *mhd_solver.gfs.B1_gf);
                // real_t A1dotB1minus = mhd_solver.GFInnerProduct(*mhd_solver.gfs.A1_gf, *mhd_solver.old_gfs.B1_gf);
                // ParGridFunction B1_diff(*mhd_solver.gfs.B1_gf);
                // B1_diff -= *mhd_solver.old_gfs.B1_gf;
                // real_t A1dotB1diff = mhd_solver.GFInnerProduct(*mhd_solver.gfs.A1_gf, B1_diff);
                // ParGridFunction A1_diff(*mhd_solver.gfs.A1_gf);
                // A1_diff -= *mhd_solver.old_gfs.A1_gf;
                // real_t A1diffdotB1 = mhd_solver.GFInnerProduct(A1_diff, *mhd_solver.old_gfs.B1_gf);
                // mfemPrintf("A1dotB1plus = %.12f, A1dotB1minus = %.12f\n", A1dotB1plus, A1dotB1minus);
                // mfemPrintf("A1dotB1diff = %.12f, A1diffdotB1 = %.12f\n", A1dotB1diff, A1diffdotB1);
                
                // ParGridFunction A2_diff(*mhd_solver.gfs.A2_gf);
                // A2_diff -= *mhd_solver.old_gfs.A2_gf;
                // real_t A2diffdotB2 = mhd_solver.GFInnerProduct(A2_diff, *mhd_solver.gfs.B2_gf);
                // mfemPrintf("A2diffdotB2 = %.12f\n", A2diffdotB2);
                
                // real_t B1dotA1 = mhd_solver.GFInnerProduct(*mhd_solver.gfs.B1_gf, *mhd_solver.gfs.A1_gf);
                // real_t A2dotcurlA1 = mhd_solver.GFDotCurlGF(*mhd_solver.gfs.A2_gf, *mhd_solver.gfs.A1_gf);
                // mfemPrintf("B1dotA1 = %.12f, A2dotcurlA1 = %.12f\n", B1dotA1, A2dotcurlA1);
                
                m_helicity_1 = mhd_solver.GFInnerProduct(B1_mid, *mhd_solver.gfs.A1_gf);
                m_helicity1_out << t_int << " " << m_helicity_1 << endl;
                
                m_helicity_2 = mhd_solver.GFInnerProduct(A2_mid, *mhd_solver.gfs.B2_gf);
                m_helicity2_out << t_int << " " << m_helicity_2 << endl;
                
                f_helicity_1 = mhd_solver.GFInnerProduct(u1_mid, *mhd_solver.gfs.w1_gf);
                f_helicity1_out << t_int << " " << f_helicity_1 << endl;
                
                f_helicity_2 = mhd_solver.GFInnerProduct(*mhd_solver.gfs.u2_gf, w2_mid);
                f_helicity2_out << t_int << " " << f_helicity_2 << endl;
            }

            // output current and vector potential
            {
                Vector CurrentValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.j2_gf, XPoint, CurrentValue);
                current_out_half << t_half << " " << CurrentValue(0) << " " << CurrentValue(1) << " " << CurrentValue(2) << endl;
                Vector AValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.A2_gf, XPoint, AValue);
                A_out_half << t_half << " " << AValue(0) << " " << AValue(1) << " " << AValue(2) << endl;
                
                // update opoint
                Vector Uvalue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.u1_gf, LeftOpoint_int, Uvalue);
                LeftOpoint_int.Add(solver_info.dt, Uvalue);
                if(i == 0) LeftOpoint_half.Add(0.5*solver_info.dt, Uvalue);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.u1_gf, RightOpoint_int, Uvalue);
                RightOpoint_int.Add(solver_info.dt, Uvalue);
                if(i == 0) RightOpoint_half.Add(0.5*solver_info.dt, Uvalue);
                opoint_integer_out << t_int + solver_info.dt << " " << LeftOpoint_int(0) << " " << LeftOpoint_int(1) << " " << LeftOpoint_int(2) << " " << RightOpoint_int(0) << " " << RightOpoint_int(1) << " " << RightOpoint_int(2) << endl;
                
                if(i == 0)
                {
                    opoint_half_out << t_half << " " << LeftOpoint_half(0) << " " << LeftOpoint_half(1) << " " << LeftOpoint_half(2) << " " << RightOpoint_half(0) << " " << RightOpoint_half(1) << " " << RightOpoint_half(2) << endl;
                }
                
                
            }

            {
                // real_t B0, B1, B2;
                // B0 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 0));
                // B1 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 1));
                // B2 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B1_gf, *mhd_solver.B1_gf, 2));
                // B_out_half << t_half << " " << B0 << " " << B1 << " " << B2 << endl;

                Vector BValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.B1_gf, BPoint, BValue);
                B_out_half << t_half << " " << BValue(0) << " " << BValue(1) << " " << BValue(2) << endl;
            }

            if(!interp_int)
            {
                bool update_operators = (i > 0) ? false : true;
                n_iter_int = mhd_solver.IntegerStep(t_int, solver_info.dt, update_operators);
                iter_integer_out << t_int << " " << n_iter_int << endl;
            }
            else
            {
                t_int += solver_info.dt;
                ucoeff.SetTime(t_int);
                mhd_solver.gfs.u2_gf->ProjectCoefficient(ucoeff);
                wcoeff.SetTime(t_int);
                mhd_solver.gfs.w1_gf->ProjectCoefficient(wcoeff);
                pcoeff.SetTime(t_int);
                mhd_solver.gfs.p3_gf->ProjectCoefficient(pcoeff);
                Acoeff.SetTime(t_int);
                mhd_solver.gfs.A1_gf->ProjectCoefficient(Acoeff);
                Bcoeff.SetTime(t_int);
                mhd_solver.gfs.B2_gf->ProjectCoefficient(Bcoeff);
                jcoeff.SetTime(t_int);
                mhd_solver.gfs.j1_gf->ProjectCoefficient(jcoeff);
            }
            
            // output divergence
            {
                real_t divu = CheckDivergenceFree(mhd_solver.gfs.u2_gf);
                real_t divB = CheckDivergenceFree(mhd_solver.gfs.B2_gf);
                divergence_integer_out << t_int << " " << divu << " " << divB << endl;
            }
            
            current_peak_integer = mhd_solver.gfs.j1_gf->ComputeMaxError(zerovec_coeff);
            current_peak_integer_out << t_int << " " << current_peak_integer << endl;

            mhd_solver.CalcEnergy(*mhd_solver.gfs.u2_gf, *mhd_solver.gfs.B2_gf, Energy_int_kinetic, Energy_int_magnetic);
            
            {
            ParGridFunction w1_mid(*mhd_solver.gfs.w1_gf);
            w1_mid += *mhd_solver.old_gfs.w1_gf;
            w1_mid *= 0.5;
            ParGridFunction j1_mid(*mhd_solver.gfs.j1_gf);
            j1_mid += *mhd_solver.old_gfs.j1_gf;
            j1_mid *= 0.5;
            real_t energy_int_var = (solver_info.viscosity? -solver_info.dt / pd->param.Re * mhd_solver.GFInnerProduct(w1_mid, w1_mid) : 0.0) 
            + (solver_info.resistivity? -solver_info.dt *pd->param.s / pd->param.Rm * mhd_solver.GFInnerProduct(j1_mid, j1_mid): 0.0);
            
            Energy_int = Energy_int_kinetic + Energy_int_magnetic;
            energy_integer_out << t_int << " " << Energy_int << ' ' << Energy_int_kinetic << ' ' << Energy_int_magnetic << ' ' << energy_int_var << endl;
            }

            {
                Vector CurrentValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.j1_gf, XPoint, CurrentValue);
                current_out_integer << t_int << " " << CurrentValue(0) << " " << CurrentValue(1) << " " << CurrentValue(2) << endl;
                Vector AValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.A1_gf, XPoint, AValue);
                A_out_integer << t_int << " " << AValue(0) << " " << AValue(1) << " " << AValue(2) << endl;
                
                // update opoint
                Vector Uvalue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.u2_gf, LeftOpoint_half, Uvalue);
                LeftOpoint_half.Add(solver_info.dt, Uvalue);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.u2_gf, RightOpoint_half, Uvalue);
                RightOpoint_half.Add(solver_info.dt, Uvalue);
                opoint_half_out << t_half + solver_info.dt << " " << LeftOpoint_half(0) << " " << LeftOpoint_half(1) << " " << LeftOpoint_half(2) << " " << RightOpoint_half(0) << " " << RightOpoint_half(1) << " " << RightOpoint_half(2) << endl;
            }

            {
                // real_t B0, B1, B2;
                // B0 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 0));
                // B1 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 1));
                // B2 = sqrt(mhd_solver.GFComponentInnerProduct(*mhd_solver.B2_gf, *mhd_solver.B2_gf, 2));
                // B_out_integer << t_int << " " << B0 << " " << B1 << " " << B2 << endl;

                Vector BValue(3);
                EvaluateVectorGFAtPoint(*mhd_solver.gfs.B2_gf, BPoint, BValue);
                B_out_integer << t_int << " " << BValue(0) << " " << BValue(1) << " " << BValue(2) << endl;
            }
            
            mhd_solver.ComputeErrorPrimalDual(t_half, i==n_steps-1, &primal_dual_error_out);

            if (visualization)
            {
                if (CheckPlot(plot_time, t_int))
                {
                    mhd_solver.OutputParaview(paraview_dir, i_plot, t_int);
                    i_plot++;
                }
            }
        }

        energy_integer_out.close();
        energy_half_out.close();
        m_helicity1_out.close();
        m_helicity2_out.close();
        f_helicity1_out.close();
        f_helicity2_out.close();
        iter_integer_out.close();
        iter_half_out.close();
        divergence_integer_out.close();
        divergence_half_out.close();

        current_out_integer.close();
        current_out_half.close();

        A_out_integer.close();
        A_out_half.close();

        B_out_integer.close();
        B_out_half.close();
        
        primal_dual_error_out.close();
        
        current_peak_integer_out.close();
        current_peak_half_out.close();
        
        opoint_integer_out.close();
        opoint_half_out.close();

        if (pd->has_exact_solution)
        {
            mhd_solver.ComputeError(t_int, t_half, solver_info.dt);
        }

        delete pmesh;
        delete pd;

        mfemPrintf("----------------------------------------------------------\n");
        mfemPrintf("Computation done.\n");
        mfemPrintf("----------------------------------------------------------\n");
        if (Mpi::Root())
            args.PrintOptions(cout);
        mfemPrintf("----------------------------------------------------------\n");
    }
    MFEMFinalizePetsc();
    Hypre::Finalize();
    Mpi::Finalize();

    return 0;
}

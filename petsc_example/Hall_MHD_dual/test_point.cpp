#include "Hall_MHD_dual.hpp"
#include "tools.hpp"

void XYZ(const Vector &x, Vector &y)
{
    y(0) = x(0);
    y(1) = x(1);
    y(2) = x(2);
}


int main(int argc, char *argv[])
{
   
    Mpi::Init();
    Hypre::Init();

    // 1. Parse command-line options.
    int Nx = 10, Ny = 10, Nz = 10;
    real_t Sx = 1.0, Sy = 1.0, Sz = 1.0; // size
    real_t Ax = 0.0, Ay = 0.0, Az = 0.0; // left bottom corner

    OptionsParser args(argc, argv);
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
    args.Parse();
    if (!args.Good())
    {
        if(Mpi::Root())
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if(Mpi::Root())
    {
        args.PrintOptions(cout);
    }

    /* prepare the mesh */
    Mesh mesh = Mesh::MakeCartesian3D(Nx, Ny, Nz, Element::HEXAHEDRON, Sx, Sy, Sz);
    // shift the mesh
    Vector displacements(mesh.SpaceDimension()*mesh.GetNV());
    for(int i=0; i<mesh.GetNV(); i++)
    {
        displacements(0*mesh.GetNV()+i) = Ax;
        displacements(1*mesh.GetNV()+i) = Ay;
        displacements(2*mesh.GetNV()+i) = Az;
    }
    mesh.MoveVertices(displacements);
    
    // refine the parallel mesh
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
    mfemPrintf("get here \n");
    
    pmesh->PrintInfo();
    
    int order = 2;
    int dim = pmesh->Dimension();
    FiniteElementCollection *H1_coll = new H1_FECollection(order, dim);
    FiniteElementCollection *ND_coll = new ND_FECollection(order, dim);
    FiniteElementCollection *RT_coll = new RT_FECollection(order-1, dim);
    FiniteElementCollection *L2_coll = new L2_FECollection(order-1, dim);
    
    ParFiniteElementSpace *H1space = new ParFiniteElementSpace(pmesh, H1_coll);
    ParFiniteElementSpace *NDspace = new ParFiniteElementSpace(pmesh, ND_coll);
    ParFiniteElementSpace *RTspace = new ParFiniteElementSpace(pmesh, RT_coll);
    ParFiniteElementSpace *L2space = new ParFiniteElementSpace(pmesh, L2_coll);
    
    // test EvaluateGFAtPoint
    {
        ParGridFunction test_gf(NDspace);
        VectorFunctionCoefficient XYZ_coeff(3, XYZ);
        test_gf.ProjectCoefficient(XYZ_coeff);
        
        Vector pt({0.2,0.3,0.4});
        Vector val;
        
        EvaluateVectorGFAtPoint(test_gf, pt, val);
    }

    
    delete H1space;
    delete NDspace;
    delete RTspace;
    delete L2space;
    
    delete H1_coll;
    delete ND_coll;
    delete RT_coll;
    delete L2_coll;
    
    delete pmesh;    
    
    Hypre::Finalize();
    Mpi::Finalize();

    return 0;
}

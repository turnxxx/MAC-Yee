#include "Integrators.hpp"

namespace mfem
{

    void BCrossCurlCurlIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans,
                                                         DenseMatrix &elmat)
    {
        int nd = el.GetDof();
        int dim = el.GetDim();
        int dimc = el.GetCurlDim();
        real_t w;

        Vector D;
        DenseMatrix curlshape(nd, dimc), curlshape_dFt(nd, dimc), M(dim, dim);
        Vector Bvec(dim);

        elmat.SetSize(nd);

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            int order;
            if (el.Space() == FunctionSpace::Pk)
            {
                order = 2 * el.GetOrder() - 2 + B_gf->ParFESpace()->GetFE(0)->GetOrder();
            }
            else
            {
                order = 2 * el.GetOrder() + B_gf->ParFESpace()->GetFE(0)->GetOrder();
            }

            ir = &IntRules.Get(el.GetGeomType(), order);
        }

        elmat = 0.0;
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);

            Trans.SetIntPoint(&ip);

            w = ip.weight * Trans.Weight();
            el.CalcPhysCurlShape(Trans, curlshape_dFt);
            B_gf->GetVectorValue(Trans, ip, Bvec);

            M = 0.0;
            M(0, 0) = 0.0;
            M(0, 1) = Bvec(2);
            M(0, 2) = -Bvec(1);
            M(1, 0) = -Bvec(2);
            M(1, 1) = 0.0;
            M(1, 2) = Bvec(0);
            M(2, 0) = Bvec(1);
            M(2, 1) = -Bvec(0);
            M(2, 2) = 0.0;
            M *= w * alpha;
            Mult(curlshape_dFt, M, curlshape);
            AddMultABt(curlshape, curlshape_dFt, elmat);
        }
    }
    
void VectorBdryNormalDotUxVIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                const FiniteElement &test_fe1,
                                const FiniteElement &test_fe2,
                                FaceElementTransformations &Trans,
                                DenseMatrix &elmat)
{
    int ndof_trial, ndof_test, dim, order;
    
    ndof_trial = trial_face_fe.GetDof();
    ndof_test = test_fe1.GetDof();
    dim = Trans.GetSpaceDim();
    
    DenseMatrix shape_trial, shape_test;
    shape_trial.SetSize(ndof_trial, dim);
    shape_test.SetSize(ndof_test, dim);
    
    elmat.SetSize(ndof_test, ndof_trial);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        order = trial_face_fe.GetOrder() + test_fe1.GetOrder();
        ir = &IntRules.Get(Trans.GetGeometryType(), order);
    }
    
    Vector normal(dim);

    for (int p = 0; p < ir->GetNPoints(); p++)
    {
        const IntegrationPoint &ip = ir->IntPoint(p);
        
        Trans.SetAllIntPoints(&ip);
        
        CalcOrtho(Trans.Jacobian(), normal);
                
        trial_face_fe.CalcPhysVShape(Trans, shape_trial);
        test_fe1.CalcPhysVShape(Trans.GetElement1Transformation(), shape_test);
        
        DenseMatrix mat_temp;
        mat_temp.SetSize(dim,dim);
        for(int i = 0; i < ndof_test; i++)
        {
            for(int j = 0; j < ndof_trial; j++)
            {
                mat_temp(0,0) = normal(0);
                mat_temp(0,1) = normal(1);
                mat_temp(0,2) = normal(2);
                mat_temp(1,0) = shape_trial(j,0);
                mat_temp(1,1) = shape_trial(j,1);
                mat_temp(1,2) = shape_trial(j,2);
                mat_temp(2,0) = shape_test(i,0);
                mat_temp(2,1) = shape_test(i,1);
                mat_temp(2,2) = shape_test(i,2);
                
                elmat(i,j) += alpha*ip.weight*mat_temp.Det();
            }
        }                
    }

}

}
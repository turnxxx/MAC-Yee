#include "ErrorEstimator.hpp"

namespace mfem
{

    void MultipleLpEstimator::AppendEstimator(Coefficient &coef, GridFunction &sol)
    {
        estimators.push_back(new LpErrorEstimator(p, coef, sol));
    }

    void MultipleLpEstimator::AppendEstimator(VectorCoefficient &coef, GridFunction &sol)
    {
        estimators.push_back(new LpErrorEstimator(p, coef, sol));
    }

    void MultipleLpEstimator::Reset()
    {
        for (auto estimator : estimators)
        {
            estimator->Reset();
        }
    }

    const Vector &MultipleLpEstimator::GetLocalErrors()
    {
        Vector error_vec;
        error_vec = estimators[0]->GetLocalErrors();

        for (long unsigned int i = 1; i < estimators.size(); i++)
        {
            error_vec += estimators[i]->GetLocalErrors();
        }

        error_estimates.SetSize(error_vec.Size());
        error_estimates = error_vec;

        return error_estimates;
    }

    void DifferenceLpEstimator::RegisterVariable(ParGridFunction &u, ParGridFunction &v)
    {
        u_gfs.Append(&u);
        v_gfs.Append(&v);
    }

    void DifferenceLpEstimator::ComputeEstimates()
    {

        error_estimates.SetSize(pmesh->GetNE());
        error_estimates = 0.0;

        MFEM_VERIFY(u_gfs.Size() == v_gfs.Size(), "u_gfs and v_gfs must have the same size");
        MFEM_VERIFY(u_gfs.Size() > 0, "u_gfs and v_gfs must have at least one element");

        Vector error_tmp;

        for (int i = 0; i < u_gfs.Size(); i++)
        {
            ParGridFunction *u = u_gfs[i];
            ParGridFunction *v = v_gfs[i];
            
            error_tmp.SetSize(pmesh->GetNE());

            if (u->VectorDim() == 1)
            {
                GridFunctionCoefficient u_coef(u);
                v->ComputeElementLpErrors(p, u_coef, error_tmp);
            }
            else
            {
                VectorGridFunctionCoefficient u_coef(u);
                v->ComputeElementLpErrors(p, u_coef, error_tmp);
            }

            error_estimates += error_tmp;
        }
        
        current_sequence = pmesh->GetSequence();
    }
    
    const Vector & DifferenceLpEstimator::GetLocalErrors()
    {
        if (current_sequence < pmesh->GetSequence())
        {
            ComputeEstimates();
        }
        
        return error_estimates;
    }
        
}
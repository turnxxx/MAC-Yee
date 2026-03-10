#ifndef _ERROR_ESTIMATOR_HPP_
#define _ERROR_ESTIMATOR_HPP_

#include"mfem.hpp"

using namespace std;

namespace mfem
{


// Give me many error estimators, I will give you a total error
class MultipleLpEstimator : public ErrorEstimator
{
    
protected:

    int p;
    vector<ErrorEstimator*> estimators;
    
    Vector error_estimates;

public:
    
    MultipleLpEstimator(int p_): p(p_){}
    
    void AppendEstimator(Coefficient &coef, GridFunction &sol);
    
    void AppendEstimator(VectorCoefficient &coef, GridFunction &sol);
    
    ~MultipleLpEstimator()
    {
        for(auto estimator: estimators)
        {
            delete estimator;
        }
    }
    
    virtual void Reset() override;
    
    virtual const Vector &GetLocalErrors() override;
    
};


// eta_k = \sum_{i=1}^{n} |u_i - v_i|_p
class DifferenceLpEstimator : public ErrorEstimator
{
    
protected:

    int p;
    const ParMesh *pmesh;
    
    int current_sequence;
        
    // u_i, v_i
    // does not own the memory
    Array<ParGridFunction*> u_gfs;
    Array<ParGridFunction*> v_gfs;
    
    Vector error_estimates;
    
    void ComputeEstimates();

public:
    
    DifferenceLpEstimator(int p_, const ParMesh* pmesh_): p(p_), pmesh(pmesh_), current_sequence(-1){}
    
    ~DifferenceLpEstimator(){}
    
    void RegisterVariable(ParGridFunction &u, ParGridFunction &v);
    
    virtual void Reset() override {current_sequence = -1;}
    
    virtual const Vector &GetLocalErrors() override;
    
};

}


#endif
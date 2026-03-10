#ifndef _INTEGRATORS_HPP
#define _INTEGRATORS_HPP

#include "mfem.hpp"

namespace mfem 
{
    /// Integrator for $(B x \mathrm{curl}(u), \mathrm{curl}(v))$ for Nedelec elements
    class BCrossCurlCurlIntegrator : public BilinearFormIntegrator
    {
    private:

    protected:
        real_t alpha;
        ParGridFunction *B_gf;

    public:
        /// Construct a bilinear form integrator for Nedelec elements
        BCrossCurlCurlIntegrator(ParGridFunction &B_gf_, real_t a=1.0, const IntegrationRule *ir = NULL) : BilinearFormIntegrator(ir), B_gf(&B_gf_), alpha(a) {}

        /* Given a particular Finite Element, compute the
           element curl-curl matrix elmat */
        virtual void AssembleElementMatrix(const FiniteElement &el,
                                           ElementTransformation &Trans,
                                           DenseMatrix &elmat);
    };
    
    
    /*
    integrator for \int_\Gamma n dot (u x v)
    u : H(div) 
    v: H(curl)
    */
    class VectorBdryNormalDotUxVIntegrator : public BilinearFormIntegrator
    {
    protected:

    private:
        real_t alpha;

    public:
        VectorBdryNormalDotUxVIntegrator(real_t alpha_=1.0)
            : alpha(alpha_) {}
            
        virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
    };
    
}

 


#endif
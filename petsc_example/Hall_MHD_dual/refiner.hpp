#ifndef _REFINER_HPP_
#define _REFINER_HPP_

#include"mfem.hpp"

using namespace std;

namespace mfem
{
    
    // Add features to the ThresholdRefiner
    // TODO:
    // 1. If the refined mesh has more elements than max_elements, then refine the mesh to have max_elements
    // 2. Ability to get the total error [done]
    // 3. Ability to set a maximum level of refinement
    class CustomRefiner : public ThresholdRefiner
    {
        protected:
        
        real_t total_err;
        
        virtual int ApplyImpl(Mesh &mesh);
        
        public:
        
        CustomRefiner(ErrorEstimator &est): ThresholdRefiner(est){}
        
        real_t GetTotalError() const { 
            MFEM_VERIFY(current_sequence != -1, "Refiner has not been applied to a mesh yet");
            return total_err; 
        }
    };
    
    
}

#endif
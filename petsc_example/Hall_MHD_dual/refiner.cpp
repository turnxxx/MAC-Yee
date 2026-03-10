#include "refiner.hpp"

namespace mfem
{
    
int CustomRefiner::ApplyImpl(Mesh &mesh)
{
   threshold = 0.0;
   num_marked_elements = 0LL;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   const Vector &local_err = estimator.GetLocalErrors();
   MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

   total_err = GetNorm(local_err, mesh);
   if (total_err <= total_err_goal) { return STOP; }

   if (total_norm_p < infinity())
   {
      threshold = std::max((real_t) (total_err * total_fraction *
                                     std::pow(num_elements, -1.0/total_norm_p)),
                           local_err_goal);
   }
   else
   {
      threshold = std::max(total_err * total_fraction, local_err_goal);
   }

   for (int el = 0; el < NE; el++)
   {
      if (local_err(el) > threshold)
      {
         marked_elements.Append(Refinement(el));
      }
   }

   // if (aniso_estimator)
   // {
   //    const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
   //    if (aniso_flags.Size() > 0)
   //    {
   //       for (int i = 0; i < marked_elements.Size(); i++)
   //       {
   //          Refinement &ref = marked_elements[i];
   //          ref.ref_type = aniso_flags[ref.index];
   //       }
   //    }
   // }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0LL) { return STOP; }

   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}
    
}
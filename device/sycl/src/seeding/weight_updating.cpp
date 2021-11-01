/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "seeding/weight_updating.hpp"

namespace traccc {
namespace sycl {

// Forward decleration of the kernel class
class weight_update_kernel;

void weight_updating(const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q) {

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto triplet_counter_view = get_data(triplet_counter_container, resource);
    auto triplet_view = get_data(triplet_container, resource);

    // The thread-block is desinged to make each thread find triplets per
    // compatible middle-bot doublet

    // -- localRange
    // The dimension of workGroup (block) is the integer multiple of WARP_SIZE (=32)
    unsigned int localRange = 64;
    // Calculate the global number of threads to run in kernel
    unsigned int globalRange = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        globalRange += triplet_container.get_headers()[i];
    }
    // Tweak the global range so that it is exactly how it is in cuda (make it multiple of the local range)
    globalRange -= localRange * internal_sp_view.headers.size();
    // 1 dim ND Range for the kernel
    auto weightUpdateNdRange = ::sycl::nd_range<1>{::sycl::range<1>{globalRange},
                                                 ::sycl::range<1>{localRange}};    
    
    q->submit([&](::sycl::handler& h){

        // local memory initialization (equivalent to shared memory in CUDA)
        auto localMem = 
            local_accessor<scalar>(filter_config.compatSeedLimit, h);
        
        WeightUpdate kernel(filter_config, internal_sp_view, 
                            triplet_counter_view, triplet_view,
                            localMem);

        h.parallel_for<class weight_update_kernel>(weightUpdateNdRange, kernel);
    });

}

} // namespace sycl
} // namespace traccc
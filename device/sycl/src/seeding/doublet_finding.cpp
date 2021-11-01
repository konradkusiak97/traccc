/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "seeding/doublet_finding.hpp"

namespace traccc {
namespace sycl {

// Forward decleration of kernel class
class doublet_find_kernel;

void doublet_finding(const seedfinder_config& config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q) {

        auto internal_sp_view = get_data(internal_sp_container, resource);
        auto doublet_counter_view = get_data(doublet_counter_container, resource);
        auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
        auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);

        // The thread-block is desinged to make each thread find doublets per
        // compatible middle spacepoints (comptible middle spacepoint means that the
        // number of mid-bot and mid-top doublets are larger than zero)

        // -- localRange
        // The dimension of workGroup (block) is the integer multiple of WARP_SIZE (=32)
        unsigned int localRange = 64;
        // Calculate the global number of threads to run in kernel
        unsigned int globalRange = 0;
        for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
            globalRange += doublet_counter_container.get_headers()[i];
        }
        // Tweak the global range so that it is exactly how it is in cuda (make it multiple of the local range)
        globalRange -= localRange * internal_sp_view.headers.size();

        // 1 dim ND Range for the kernel
        auto doubletFindNdRange = ::sycl::nd_range<1>{::sycl::range<1>{globalRange},
                                                            ::sycl::range<1>{localRange}};
         q->submit([&](::sycl::handler& h) {

            // local memory initialization (equivalent to shared memory in CUDA)
            auto localMem = 
                local_accessor<int>(localRange*2, h);
            
            DupletFind kernel(config, internal_sp_view, doublet_counter_view, 
                            mid_bot_doublet_view,mid_top_doublet_view,localMem);

            h.parallel_for<class doublet_find_kernel>(doubletFindNdRange, kernel);
        });                                                            

    }

} // namespace sycl
} // namespace traccc
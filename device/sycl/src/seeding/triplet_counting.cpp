/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "seeding/triplet_counting.hpp"

namespace traccc {
namespace sycl {

// Forward decleration of the kernel class
class triplet_count_kernel;

void triplet_counting(const seedfinder_config& config,
                    host_internal_spacepoint_container& internal_sp_container,
                    host_doublet_counter_container& doublet_counter_container,
                    host_doublet_container& mid_bot_doublet_container,
                    host_doublet_container& mid_top_doublet_container,
                    host_triplet_counter_container& triplet_counter_container,
                    vecmem::memory_resource* resource,
                    ::sycl::queue* q) {
    
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);
    auto triplet_counter_container_view =
        get_data(triplet_counter_container, resource);

    // The thread-block is desinged to make each thread count triplets per
    // middle-bot doublet

    // -- localRange
    // The dimension of workGroup (block) is the integer multiple of WARP_SIZE (=32)
    unsigned int localRange = 64;
    // Calculate the global number of threads to run in kernel
    unsigned int globalRange = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
            globalRange += mid_bot_doublet_container.get_headers()[i];
    }
    // Tweak the global range so that it is exactly how it is in cuda (make it multiple of the local range)
    globalRange -= localRange * internal_sp_view.headers.size();

    // 1 dim ND Range for the kernel
    auto tripletCountNdRange = ::sycl::nd_range<1>{::sycl::range<1>{globalRange},
                                                        ::sycl::range<1>{localRange}};
    q->submit([](::sycl::handler& h){
        TripletCount kernel(config, internal_sp_view, doublet_counter_container_view,
                            mid_bot_doublet_view, mid_top_doublet_view,
                            triplet_counter_container_view);
        h.parallel_for<class triplet_count_kernel>(tripletCountNdRange, kernel);
    });                                                      
}

} // namespace sycl
} // namespace traccc
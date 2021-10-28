/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "seeding/doublet_counting.hpp"

namespace traccc {
namespace sycl {
    
// Forward decleration of kernel class
class doublet_count_kernel;

void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource,
                      ::sycl::queue* q) {

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);

    // The local number of threads per work-group (block in cuda) (number taken from cuda implementation, warp size *2)
    unsigned int localRange = 64;
    // Calculate the global number of threads to run in kernel
    unsigned int globalRange = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        globalRange += internal_sp_view.items.m_ptr[i].size();
    }
    // Tweak the global range so that it is exactly how it is in cuda (make it multiple of the local range)
    globalRange -= localRange * internal_sp_view.headers.size();

    // 1 dim ND Range for the kernel
    auto doubletCountNdRange = ::sycl::nd_range<1>{::sycl::range<1>{globalRange},
                                                        ::sycl::range<1>{localRange}};
    
    q->submit([](::sycl::handler& h){
        DupletCount kernel( config, internal_sp_view, 
                            doublet_counter_container_view);
        h.parallel_for<class doublet_count_kernel>(doubletCountNdRange, kernel);
    });

}

} // namespace sycl
} // namespace traccc
/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <sycl/seeding/doublet_counting.hpp>
#include <CL/sycl.hpp>

namespace traccc {
namespace sycl {

void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource,
                      cl::sycl::queue* q) {

    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);
    
    

    q->submit([](cl::sycl::handler& h){
        DupletCount kernel(/*TODO*/);
        h.parallel_for<class doublet_count_kernel>(/*NDrange, TODO*/ kernel);
    });

}

// Kernel class for doublet counting
class DupletCount {
public:
    DupletCount() : {} // TODO

    void operator()(cl::sycl::nd_item</*TODO*/>) const {
            //TODO
    }


}

} // namespace sycl
} // namespace traccc
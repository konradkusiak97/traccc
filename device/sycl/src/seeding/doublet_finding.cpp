/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <sycl/seeding/doublet_finding.hpp>

namespace traccc {
namespace sycl {

void doublet_finding(const seedfinder_config& config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     vecmem::memory_resource* resource,
                     cl::sycl::queue* q) {

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

    }
}
}
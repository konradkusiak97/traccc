/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <sycl/seeding/doublet_counting.hpp>

namespace traccc {
namespace sycl {
    
// Forward decleration of kernel class
class doublet_count_kernel;

// Define shorthand alias for the type of atomics needed by this kernel 
template <typename T>
using global_atomic_ref = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource,
                      sycl::queue* q) {

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
    auto doubletCountNdRange = sycl::nd_range<1>{sycl::range<1>{globalRange},
                                                        sycl::range<1>{localRange}};
    
    q->submit([](sycl::handler& h){
        DupletCount kernel( config, internal_sp_view, 
                            doublet_counter_container_view);
        h.parallel_for<class doublet_count_kernel>(doubletCountNdRange, kernel);
    });

}

// Kernel class for doublet counting
class DupletCount {
public:
    DupletCount(const seedfinder_config config,
               internal_spacepoint_container_view internal_sp_view, 
                doublet_counter_container_view doublet_counter_view)
    : m_config(config),
      m_internal_sp_view(internal_sp_view),
      m_doublet_counter_view(doublet_counter_view) {} 

    void operator()(sycl::nd_item<1> item) {
        
        // Mapping cuda indexing to dpc++
        auto workGroup = item.get_group();
        
        // Equivalent to blockIdx.x in cuda
        auto groupIdx = workGroup.get_linear_id()
        // Equivalent to blockDim.x in cuda
        auto groupDim = workGroup.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_linear_id();
        
        // Get device container for input parameters
        device_internal_spacepoint_container internal_sp_device(
            {m_internal_sp_view.headers, m_internal_sp_view.items});
        device_doublet_counter_container doublet_counter_device(
            {m_doublet_counter_view.headers, m_doublet_counter_view.items});
        
        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

        /////////////// TAken from CUDA helper function ///////////////////////
        // the item jagged vector of edm
        auto jag_vec = internal_sp_device.get_items();

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;

        // taken from cuda helper functions - get_header_idx()
        for (unsigned int i = 0; i < jag_vec.size(); ++i) {
            nblocks_per_header = jag_vec[i].size() / groupDim + 1;
            nblocks_accum += nblocks_per_header;

            if (groupIdx < nblocks_accum) {
                header_idx = i;
                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        /////////////////// End of the helper funciton /////////////////////////   

    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects per
    // bin
    const auto& bin_info = internal_sp_device.get_headers().at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.get_items().at(bin_idx);

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    auto& num_compat_spM_per_bin =
        doublet_counter_device.get_headers().at(bin_idx);
    auto doublet_counter_per_bin =
        doublet_counter_device.get_items().at(bin_idx);

    // index of internal spacepoint in the item vector
    auto sp_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;

    if (sp_idx >= doublet_counter_per_bin.size()) return;

    // zero initialization for the number of doublets per thread (or middle sp)
    unsigned int n_mid_bot = 0;
    unsigned int n_mid_top = 0;

    // zero initialization for the number of doublets per bin
    doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
    doublet_counter_per_bin[sp_idx].n_mid_top = 0;

    // middle spacepoint index
    auto spM_loc = sp_location({bin_idx, sp_idx});
    // middle spacepoint
    const auto& isp = internal_sp_per_bin[sp_idx];

    // Loop over (bottom and top) internal spacepoints in the neighbor bins
    for (size_t i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
        const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
        const auto& neigh_internal_sp_per_bin =
            internal_sp_device.get_items().at(neigh_bin);

        for (size_t spB_idx = 0; spB_idx < neigh_internal_sp_per_bin.size();
            ++spB_idx) {
            const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];

            // Check if middle and bottom sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                    true)) {
                n_mid_bot++;
            }

            // Check if middle and top sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                    false)) {
                n_mid_top++;
            }
        }
    }
    // if number of mid-bot and mid-top doublet for a middle spacepoint is
    // larger than 0, the entry is added to the doublet counter
    if (n_mid_bot > 0 && n_mid_top > 0) {
        auto pos = global_atomic_ref<int>(num_compat_spM_per_bin);
        pos += 1;
        doublet_counter_per_bin[pos] = {spM_loc, n_mid_bot, n_mid_top};
    }        
}
private: 
    const seedfinder_config m_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
}
} // namespace sycl
} // namespace traccc
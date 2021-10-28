/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <algorithm>
#include "seeding/seed_selecting.hpp"

namespace traccc {
namespace sycl {

// Thrust comparator function for triplet weight (in descending order)
struct triplet_weight_descending
    : public thrust::binary_function<triplet, triplet, bool> {
        bool operator()(const triplet& lhs, const triplet& rhs) const {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        } else {
            return fabs(lhs.z_vertex) < fabs(rhs.z_vertex);
        }
    }
};

// comparator function for triplet weight (in ascending order)
static bool triplet_weight_compare(const triplet& lhs,
                                              const triplet& rhs) {
    if (lhs.weight != rhs.weight) {
        return lhs.weight < rhs.weight;
    } else {
        return fabs(lhs.z_vertex) > fabs(rhs.z_vertex);
    }
}

// Forward decleration of kernel class
class seed_select_kernel;

// Define shorthand alias for the type of atomics needed by this kernel 
template <typename T>
using global_atomic_ref = sycl::atomic_ref<
    T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::system,
    sycl::access::address_space::global_space>;

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = sycl::accessor<
    T,
    1,
    sycl::access::mode::read_write,
    sycl::access::target::local>;

void seed_selecting(const seedfilter_config& filter_config,
                    host_internal_spacepoint_container& internal_sp_container,
                    host_doublet_counter_container& doublet_counter_container,
                    host_triplet_counter_container& triplet_counter_container,
                    host_triplet_container& triplet_container,
                    host_seed_container& seed_container,
                    vecmem::memory_resource* resource,
                    sycl::queue* q) {
    
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);
    auto triplet_counter_container_view =
        get_data(triplet_counter_container, resource);
    auto triplet_container_view = get_data(triplet_container, resource);
    auto seed_container_view = get_data(seed_container, resource);

    // The thread-block is desinged to make each thread find triplets per
    // compatible middle-bot doublet

    // -- localRange
    // The dimension of workGroup (block) is the integer multiple of WARP_SIZE (=32)
    unsigned int localRange = 64;
    // Calculate the global number of threads to run in kernel
    unsigned int globalRange = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
            globalRange += triplet_counter_container.get_headers()[i];
    }
    // Tweak the global range so that it is exactly how it is in cuda (make it multiple of the local range)
    globalRange -= localRange * internal_sp_view.headers.size();

    // 1 dim ND Range for the kernel
    auto seedSelectNdRange = sycl::nd_range<1>{sycl::range<1>{globalRange},
                                                sycl::range<1>{localRange}};
    q->submit([](sycl::handler& h){

        // local memory initialization (equivalent to shared memory in CUDA)
        auto localMem = 
            local_accessor<triplet>(localRange * filter_config.max_triplets_per_spM, 
                                    h); 
        SeedSelect kernel(filter_config, internal_sp_view, doublet_counter_container_view,
                          triplet_counter_container_view, triplet_container_view,
                          seed_container_view, localMem);

        h.parallel_for<class seed_select_kernel>(seedSelectNdRange, kernel);
    });
}
class SeedSelect {
public:
    SeedSelect(const seedfilter_config filter_config,
               internal_spacepoint_container_view internal_sp_view,
               doublet_counter_container_view doublet_counter_view,
               triplet_counter_container_view triplet_counter_view,
               triplet_container_view triplet_view, 
               seed_container_view seed_view,
               local_accessor<triplet> localMem)
    : m_filter_config(filter_config),
      m_doublet_counter_view(doublet_counter_view),
      m_triplet_counter_view(triplet_counter_view),
      m_triplet_view(triplet_view),
      m_seed_view(seed_view),
      m_localMem(localMem) {}

      void operator()(sycl::nd_item<1> item) {

          // Mapping cuda indexing to dpc++
        auto workGroup = item.get_group();
        
        // Equivalent to blockIdx.x in cuda
        auto groupIdx = workGroup.get_linear_id();
        // Equivalent to blockDim.x in cuda
        auto groupDim = workGroup.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_linear_id();

        device_internal_spacepoint_container internal_sp_device(
        {m_internal_sp_view.headers, m_internal_sp_view.items});
        device_doublet_counter_container doublet_counter_device(
            {m_doublet_counter_view.headers, m_doublet_counter_view.items});
        device_triplet_counter_container triplet_counter_device(
            {m_triplet_counter_view.headers, m_triplet_counter_view.items});
        device_triplet_container triplet_device(
            {m_triplet_view.headers, m_triplet_view.items});
        device_seed_container seed_device({m_seed_view.headers, m_seed_view.items});

        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

        /////////////// TAken from CUDA helper function ///////////////////////
        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < triplet_counter_device.size(); ++i) {
            nblocks_per_header = triplet_counter_device.get_headers()[i] / groupDim + 1;
            nblocks_accum += nblocks_per_header;

            if (groupIdx < nblocks_accum) {
                header_idx = i;
                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        /////////////////// End of the helper funciton ////////////////////

        // Header of internal spacepoint container : spacepoint bin information
        // Item of internal spacepoint container : internal spacepoint objects per
        // bin
        auto internal_sp_per_bin = internal_sp_device.get_items().at(bin_idx);
        auto& num_compat_spM_per_bin =
            doublet_counter_device.get_headers().at(bin_idx);

        // Header of doublet counter : number of compatible middle sp per bin
        // Item of doublet counter : doublet counter objects per bin
        auto doublet_counter_per_bin =
            doublet_counter_device.get_items().at(bin_idx);

        // Header of triplet counter: number of compatible mid_top doublets per bin
        // Item of triplet counter: triplet counter objects per bin
        auto& num_compat_mb_per_bin =
            triplet_counter_device.get_headers().at(bin_idx);

        // Header of triplet: number of triplets per bin
        // Item of triplet: triplet objects per bin
        auto& num_triplets_per_bin = triplet_device.get_headers().at(bin_idx);
        auto triplets_per_bin = triplet_device.get_items().at(bin_idx);

        // Header of seed: number of seeds per event
        // Item of seed: seed objects per event
        auto& num_seeds = seed_device.get_headers().at(0);
        auto seeds = seed_device.get_items().at(0);

        auto triplets_per_spM = m_localMem;

        // index of doublet counter in the item vector
        auto gid = (groupIdx - ref_block_idx) * groupDim + workItemIdx;

        // prevent overflow
        if (gid >= num_compat_spM_per_bin) {
            return;
        }
        
        // middle spacepoint index
        auto& spM_loc = doublet_counter_per_bin[gid].spM;
        auto& spM_idx = spM_loc.sp_idx;
        // middle spacepoint
        auto& spM = internal_sp_per_bin[spM_idx];

        // number of triplets per compatible middle spacepoint
        unsigned int n_triplets_per_spM = 0;

        // the start index of triplets_per_spM
        unsigned int stride = workItemIdx * m_filter_config.max_triplets_per_spM;

        // iterate over the triplets in the bin
        for (unsigned int i = 0; i < num_triplets_per_bin; ++i) {
            auto& aTriplet = triplets_per_bin[i];
            auto& spB_loc = aTriplet.sp1;
            auto& spT_loc = aTriplet.sp3;
            auto& spB =
                internal_sp_device.get_items()[spB_loc.bin_idx][spB_loc.sp_idx];
            auto& spT =
                internal_sp_device.get_items()[spT_loc.bin_idx][spT_loc.sp_idx];

            // consider only the triplets with the same middle spacepoint
            if (spM_loc == aTriplet.sp2) {

                // update weight of triplet
                seed_selecting_helper::seed_weight(m_filter_config, spM, spB, spT,
                                                aTriplet.weight);

                // check if it is a good triplet
                if (!seed_selecting_helper::single_seed_cut(m_filter_config, spM, spB,
                                                            spT, aTriplet.weight)) {
                    continue;
                }

                // if the number of good triplets is larger than the threshold, the
                // triplet with the lowest weight is removed
                if (n_triplets_per_spM >= m_filter_config.max_triplets_per_spM) {
                    int begin_idx = stride;
                    int end_idx = stride + m_filter_config.max_triplets_per_spM;

                    // Note: min_index method gives a result different
                    //       from sorting method when there are the cases where
                    //       weight & z_vertex are same.
                    //
                    //       So min_index method reduces seed matching ratio
                    //       since the cpu version is using sorting method.
                    //
                    //       But that doesn't mean min_index method
                    //       is wrong of course
                    //
                    //       Let's not be so obsessed about achieving
                    //       perfectly same result :))))))))

                    int min_index = std::min_element(triplets_per_spM + begin_idx,
                                                    triplets_per_spM + end_idx,
                                                    triplet_weight_compare) -
                                    triplets_per_spM;

                    auto& min_weight = triplets_per_spM[min_index].weight;

                    if (aTriplet.weight > min_weight) {
                        triplets_per_spM[min_index] = aTriplet;
                    }
                }

                else if (n_triplets_per_spM < m_filter_config.max_triplets_per_spM) {
                    triplets_per_spM[stride + n_triplets_per_spM] = aTriplet;
                    n_triplets_per_spM++;
                }
            }       
        }

        // sort the triplets per spM
        // sequential version of thrust sorting algorithm is used
        thrust::sort(thrust::seq, triplets_per_spM + stride,
                    triplets_per_spM + stride + n_triplets_per_spM,
                    triplet_weight_descending());

        // the number of good seed per compatible middle spacepoint
        unsigned int n_seeds_per_spM = 0;

        // iterate over the good triplets for final selection of seeds
        for (unsigned int i = stride; i < stride + n_triplets_per_spM; ++i) {
            auto& aTriplet = triplets_per_spM[i];
            auto& spB_loc = aTriplet.sp1;
            auto& spT_loc = aTriplet.sp3;
            auto& spB =
                internal_sp_device.get_items()[spB_loc.bin_idx][spB_loc.sp_idx];
            auto& spT =
                internal_sp_device.get_items()[spT_loc.bin_idx][spT_loc.sp_idx];

            // if the number of seeds reaches the threshold, break
            if (n_seeds_per_spM >= m_filter_config.maxSeedsPerSpM + 1) {
                break;
            }

            // check if it is a good triplet
            if (seed_selecting_helper::cut_per_middle_sp(
                    m_filter_config, spM.sp(), spB.sp(), spT.sp(), aTriplet.weight) ||
                n_seeds_per_spM == 0) {
                auto pos = global_atomic_ref<int>(num_seeds);
                pos += 1;

                // prevent overflow
                if (pos >= seeds.size()) {
                    break;
                }
                n_seeds_per_spM++;

                seeds[pos] = seed({spB.m_sp, spM.m_sp, spT.m_sp, aTriplet.weight,
                                aTriplet.z_vertex});
            }
        }

      }
private:
    const seedfilter_config m_filter_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
    triplet_counter_container_view m_triplet_counter_view;
    triplet_container_view m_triplet_view;
    seed_container_view m_seed_view;
    local_accessor<triplet> m_localMem;
};                                  

} // namespace sycl
} // namesapce traccc
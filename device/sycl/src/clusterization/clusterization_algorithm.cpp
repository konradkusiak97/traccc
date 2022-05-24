/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"

#include "traccc/device/get_prefix_sum.hpp"

// SYCL library include(s).
#include "cluster_counting.hpp"
#include "clusters_sum.hpp"
#include "component_connection.hpp"
#include "measurement_creation.hpp"
#include "spacepoint_formation.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::sycl {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource &mr,
                                                   queue_wrapper queue)
    : m_mr(mr), m_queue(queue) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Get the view of the cells container
    auto cells_data = get_data(cells_per_event, &m_mr.get());
    cell_container_types::const_view cells_view(cells_data);

    // Get the sizes of the cells in each module
    auto cell_sizes = copy.get_sizes(cells_view.items);

    // Get the cell sizes with +1 in each entry for sparse ccl indices buffer
    // The +1 is needed to store the number of found clusters at the end of
    // the vector in each module
    std::vector<std::size_t> cell_sizes_plus(num_modules);
    std::transform(cell_sizes.begin(), cell_sizes.end(),
                   cell_sizes_plus.begin(),
                   [](std::size_t x) { return x + 1; });

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices(
        cell_sizes_plus, m_mr.get());
    copy.setup(sparse_ccl_indices);

    // Vector with numbers of found clusters in each module, later trandformed into prefix sum vector
    vecmem::data::vector_buffer<std::size_t> clusters_per_module(num_modules,
                                                                 m_mr.get());
    copy.setup(clusters_per_module);

    // Get the prefix sum of the cells
    const device::prefix_sum_t cells_prefix_sum =
        device::get_prefix_sum(cell_sizes, m_mr.get());

    // Clusters sum kernel
    traccc::sycl::clusters_sum(cells_view, sparse_ccl_indices, 
                               clusters_per_module,
                               m_queue);

    // Copy the sizes of clusters per each module to the host 
    vecmem::vector<std::size_t> clusters_per_module_host(&m_mr.get());
    copy(clusters_per_module, clusters_per_module_host);

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), 
         std::vector<std::size_t>(clusters_per_module_host.begin(), clusters_per_module_host.end()),
         m_mr.get()}};
    copy.setup(measurements_buffer.headers);
    copy.setup(measurements_buffer.items);

    // Spacepoint container buffer to fill in spacepoint formation
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), 
         std::vector<std::size_t>(clusters_per_module_host.begin(), clusters_per_module_host.end()),
         m_mr.get()}};
    copy.setup(spacepoints_buffer.headers);
    copy.setup(spacepoints_buffer.items);

    // Perform the exclusive scan operation. It will results with a vector of prefix sums, starting with 0, until the second to last sum.
    // The total numbewr of clusters in this case will be the last prefix sum + the clusters found in the last idx 
    unsigned int total_clusters = clusters_per_module_host.back();
    std::exclusive_scan(clusters_per_module_host.begin(), clusters_per_module_host.end(), clusters_per_module_host.begin(), 0);
    total_clusters += clusters_per_module_host.back();

    // Copy the prefix sum back to its device container
    copy(vecmem::get_data(clusters_per_module_host), clusters_per_module);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.get());
    copy.setup(cluster_sizes_buffer);
    copy.memset(cluster_sizes_buffer, 0);

    // Cluster counting kernel
    traccc::sycl::cluster_counting(sparse_ccl_indices, cluster_sizes_buffer,
                                   clusters_per_module,
                                   vecmem::get_data(cells_prefix_sum), m_queue);

    std::vector<unsigned int> cluster_sizes;
    copy(cluster_sizes_buffer, cluster_sizes);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.get()},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.get()}};
    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    // Component connection kernel
    traccc::sycl::component_connection(
        clusters_buffer, cells_view, sparse_ccl_indices, clusters_per_module,
        vecmem::get_data(cells_prefix_sum), m_queue);


    // Measurement creation kernel
    traccc::sycl::measurement_creation(measurements_buffer, clusters_buffer,
                                       cells_view, m_queue);


    // Get the prefix sum of the measurements.
    const device::prefix_sum_t measurements_prefix_sum = device::get_prefix_sum(
        copy.get_sizes(measurements_buffer.items), m_mr.get());

    // Spacepoint formation kernel
    traccc::sycl::spacepoint_formation(
        spacepoints_buffer, measurements_buffer,
        vecmem::get_data(measurements_prefix_sum), m_queue);

    return spacepoints_buffer;
}

}  // namespace traccc::sycl
/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Clusterization include(s)
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"

// SYCL library include(s).
#include "cluster_counting.hpp"
#include "clusters_sum.hpp"
#include "component_connection.hpp"
#include "measurement_creation.hpp"
#include "spacepoint_formation.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::sycl {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource &mr,
                                                   queue_wrapper queue)
    : m_mr(mr), m_queue(queue) {}

host_spacepoint_container clusterization_algorithm::operator()(
    const cell_container_types::host &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Get the sizes of the cells in each module (+1 is an extra space for
    // storing the n_clusters at the end of the indices vector) and the maximum
    // size of cells in module
    std::size_t cells_max = 0;
    std::vector<std::size_t> cell_sizes(num_modules, 0);
    for (std::size_t j = 0; j < num_modules; ++j) {
        cell_sizes[j] = cells_per_event.get_items().at(j).size() + 1;
        if (cell_sizes[j] - 1 > cells_max)
            cells_max = cell_sizes[j] - 1;
    }

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices(
        cell_sizes, m_mr.get());
    copy.setup(sparse_ccl_indices);

    // Vector buffer for prefix sums for proper indexing, used only on device.
    // Vector with "clusters per module" is needed for measurement creation
    // buffer
    vecmem::data::vector_buffer<std::size_t> cluster_prefix_sum(num_modules,
                                                                m_mr.get());
    vecmem::data::vector_buffer<std::size_t> clusters_per_module(num_modules,
                                                                 m_mr.get());
    copy.setup(cluster_prefix_sum);
    copy.setup(clusters_per_module);

    // Invoke the reduction kernel that gives the total number of clusters which
    // will be found (and also computes the prefix sums and clusters per module)
    auto total_clusters = vecmem::make_unique_alloc<unsigned int>(m_mr.get());
    *total_clusters = 0;
    traccc::sycl::clusters_sum(cells_per_event, sparse_ccl_indices,
                               *total_clusters, cluster_prefix_sum,
                               clusters_per_module, m_mr.get(), m_queue);

    // Vector of the exact cluster sizes, will be filled in cluster_counting
    // kernel
    vecmem::vector<unsigned int> cluster_sizes(*total_clusters, 0, &m_mr.get());
    traccc::sycl::cluster_counting(
        num_modules, sparse_ccl_indices, vecmem::get_data(cluster_sizes),
        cluster_prefix_sum, cells_max, m_mr.get(), m_queue);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {*total_clusters, m_mr.get()},
        {std::vector<std::size_t>(*total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.get()}};

    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    // Component connection kernel
    traccc::sycl::component_connection(clusters_buffer, cells_per_event,
                                       sparse_ccl_indices, cluster_prefix_sum,
                                       cells_max, m_mr.get(), m_queue);

    // Copy the sizes of clusters per each module to the std vector for
    // measurement buffer initialization
    std::vector<std::size_t> clusters_per_module_host;
    copy(clusters_per_module, clusters_per_module_host);

    // Resizable buffer for the measurements
    vecmem::data::vector_buffer<measurement> measurements_buffer(
        *total_clusters, 0, m_mr.get());
    copy.setup(measurements_buffer);

    // Measurement creation kernel
    traccc::sycl::measurement_creation(measurements_buffer, clusters_buffer,
                                       m_queue);

    spacepoint_container_buffer spacepoints_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.get()}};
    copy.setup(spacepoints_buffer.headers);
    copy.setup(spacepoints_buffer.items);

    // Do the spacepoint formation
    traccc::sycl::spacepoint_formation(spacepoints_buffer, measurements_buffer,
                                       cells_per_event, m_mr.get(), m_queue);

    host_spacepoint_container spacepoints_per_event(&m_mr.get());
    copy(spacepoints_buffer.headers, spacepoints_per_event.get_headers());
    copy(spacepoints_buffer.items, spacepoints_per_event.get_items());

    return spacepoints_per_event;
}

}  // namespace traccc::sycl
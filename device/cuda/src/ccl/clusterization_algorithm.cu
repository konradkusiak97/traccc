/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/ccl/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s)
#include "traccc/clusterization/device/connect_components.hpp"
#include "traccc/clusterization/device/count_cluster_cells.hpp"
#include "traccc/clusterization/device/create_measurements.hpp"
#include "traccc/clusterization/device/find_clusters.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/device/get_prefix_sum.hpp"

// Vecmem include(s).
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::cuda {
namespace kernels {

/// Class identifying the kernel running @c traccc::device::find_clusters
__global__ void find_clusters(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    device::find_clusters(threadIdx.x + blockIdx.x * blockDim.x, cells_view,
                          sparse_ccl_indices_view, clusters_per_module_view);
}

/// Class identifying the kernel running @c traccc::device::count_cluster_cells
__global__ void count_cluster_cells(
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view) {

    device::count_cluster_cells(
        threadIdx.x + blockIdx.x * blockDim.x, sparse_ccl_indices_view,
        cluster_prefix_sum_view, cells_prefix_sum_view, cluster_sizes_view);
}

/// Class identifying the kernel running @c traccc::device::connect_components
__global__ void connect_components(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    cluster_container_types::view clusters_view) {

    device::connect_components(threadIdx.x + blockIdx.x * blockDim.x,
                               cells_view, sparse_ccl_indices_view,
                               cluster_prefix_sum_view, cells_prefix_sum_view,
                               clusters_view);
}

/// Class identifying the kernel running @c traccc::device::create_measurements
__global__ void create_measurements(
    cluster_container_types::const_view clusters_view,
    const cell_container_types::const_view cells_view,
    measurement_container_types::view measurements_view) {

    device::create_measurements(threadIdx.x + blockIdx.x * blockDim.x,
                                clusters_view, cells_view, measurements_view);
}

__global__ void form_spacepoints(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, measurements_prefix_sum_view,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr)
    : m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host& cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Get the view of the cells container
    auto cells_data =
        get_data(cells_per_event, (m_mr.host ? m_mr.host : &(m_mr.main)));

    // Get the sizes of the cells in each module
    auto cell_sizes = m_copy->get_sizes(cells_data.items);

    // Move the cells to device buffer
    // cell_container_types::buffer cells_buffer{
    //     {num_modules, m_mr.main}, {cells_my_view.items, m_mr.main,
    //     m_mr.host}};
    // m_copy->setup(cells_buffer.headers);
    // m_copy->setup(cells_buffer.items);
    // (*m_copy)(cells_my_view.headers, cells_buffer.headers,
    // vecmem::copy::type::copy_type::host_to_device);
    // (*m_copy)(cells_my_view.items, cells_buffer.items,
    // vecmem::copy::type::copy_type::host_to_device);

    // Get the cell sizes with +1 in each entry for sparse ccl indices buffer
    // The +1 is needed to store the number of found clusters at the end of
    // the vector in each module
    std::vector<std::size_t> cell_sizes_plus(num_modules);
    std::transform(cell_sizes.begin(), cell_sizes.end(),
                   cell_sizes_plus.begin(),
                   [](std::size_t x) { return x + 1; });

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices_buff(
        cell_sizes_plus, m_mr.main, m_mr.host);
    m_copy->setup(sparse_ccl_indices_buff);

    // Vector with numbers of found clusters in each module, later transformed
    // into prefix sum vector
    vecmem::data::vector_buffer<std::size_t> cl_per_module_prefix_buff(
        num_modules, m_mr.main);
    m_copy->setup(cl_per_module_prefix_buff);

    // Calculate nd_range to run cluster finding
    std::size_t localSize = 64;
    std::size_t num_blocks = (num_modules + localSize - 1) / localSize;

    // Run cluster finding kernel
    kernels::find_clusters<<<num_blocks, localSize>>>(
        cells_data, sparse_ccl_indices_buff, cl_per_module_prefix_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Get the prefix sum of the cells and copy it to the device buffer
    const device::prefix_sum_t cells_prefix_sum = device::get_prefix_sum(
        cell_sizes, (m_mr.host ? *(m_mr.host) : m_mr.main));
    vecmem::data::vector_buffer<device::prefix_sum_element_t>
        cells_prefix_sum_buff(cells_prefix_sum.size(), m_mr.main);
    m_copy->setup(cells_prefix_sum_buff);
    (*m_copy)(vecmem::get_data(cells_prefix_sum), cells_prefix_sum_buff,
              vecmem::copy::type::copy_type::host_to_device);

    // Wait here for the cluster_finding kernel to finish

    // Copy the sizes of clusters per each module to the host
    // and create a copy of this "clusters per module" vector for other
    // operations in the future
    vecmem::vector<std::size_t> cl_per_module_prefix_host(
        m_mr.host ? m_mr.host : &(m_mr.main));
    (*m_copy)(cl_per_module_prefix_buff, cl_per_module_prefix_host,
              vecmem::copy::type::copy_type::device_to_host);
    std::vector<std::size_t> clusters_per_module_host(
        cl_per_module_prefix_host.begin(), cl_per_module_prefix_host.end());

    // Perform the exclusive scan operation. It will results with a vector of
    // prefix sums, starting with 0, until the second to last sum. The total
    // number of clusters in this case will be the last prefix sum + the
    // clusters in the last idx
    unsigned int total_clusters = cl_per_module_prefix_host.back();
    std::exclusive_scan(cl_per_module_prefix_host.begin(),
                        cl_per_module_prefix_host.end(),
                        cl_per_module_prefix_host.begin(), 0);
    total_clusters += cl_per_module_prefix_host.back();

    // Copy the prefix sum back to its device container
    (*m_copy)(vecmem::get_data(cl_per_module_prefix_host),
              cl_per_module_prefix_buff,
              vecmem::copy::type::copy_type::host_to_device);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.main);
    m_copy->setup(cluster_sizes_buffer);
    m_copy->memset(cluster_sizes_buffer, 0);

    // Run cluster counting kernel
    num_blocks = (cells_prefix_sum.size() + localSize - 1) / localSize;
    kernels::count_cluster_cells<<<num_blocks, localSize>>>(
        sparse_ccl_indices_buff, cl_per_module_prefix_buff,
        cells_prefix_sum_buff, cluster_sizes_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy cluster sizes back to the host
    std::vector<unsigned int> cluster_sizes;
    (*m_copy)(cluster_sizes_buffer, cluster_sizes,
              vecmem::copy::type::copy_type::device_to_host);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.main},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.main, m_mr.host}};
    m_copy->setup(clusters_buffer.headers);
    m_copy->setup(clusters_buffer.items);

    // Run component connection kernel (nd_range same as before)
    kernels::connect_components<<<num_blocks, localSize>>>(
        cells_data, sparse_ccl_indices_buff, cl_per_module_prefix_buff,
        cells_prefix_sum_buff, clusters_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(measurements_buffer.headers);
    m_copy->setup(measurements_buffer.items);

    // Spacepoint container buffer to fill in spacepoint formation
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(spacepoints_buffer.headers);
    m_copy->setup(spacepoints_buffer.items);

    // Wait here for the component_connection kernel to finish

    // Run measurement_creation kernel
    num_blocks = (total_clusters + localSize - 1) / localSize;
    kernels::create_measurements<<<num_blocks, localSize>>>(
        clusters_buffer, cells_data, measurements_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Get the prefix sum of the measurements and copy it to the device buffer
    const device::prefix_sum_t meas_prefix_sum =
        device::get_prefix_sum(m_copy->get_sizes(measurements_buffer.items),
                               (m_mr.host ? *(m_mr.host) : m_mr.main));
    vecmem::data::vector_buffer<device::prefix_sum_element_t>
        meas_prefix_sum_buff(meas_prefix_sum.size(), m_mr.main);
    m_copy->setup(meas_prefix_sum_buff);
    (*m_copy)(vecmem::get_data(meas_prefix_sum), meas_prefix_sum_buff,
              vecmem::copy::type::copy_type::host_to_device);

    // Run spacepoint formation kernel (ndrange same as before)
    kernels::form_spacepoints<<<num_blocks, localSize>>>(
        measurements_buffer, meas_prefix_sum_buff, spacepoints_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return spacepoints_buffer;
}

}  // namespace traccc::cuda
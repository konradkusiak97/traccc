/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE
void measurement_creation(std::size_t globalIndex,
                          cluster_container_types::const_view clusters_view,
                          const cell_container_types::const_view& cells_view,
                          measurement_container_types::view measurements_view) {

        // Initialize device vector that gives us the execution range
        const cluster_container_types::const_device clusters_device(
            clusters_view);

        // Ignore if idx is out of range
        if (globalIndex >= clusters_device.size())
            return;

        // Create other device containers
        measurement_container_types::device measurements_device(
            measurements_view);
        cell_container_types::const_device cells_device(cells_view);

        // items: cluster of cells at current idx
        // header: cluster_id object with the information about the
        // cell module
        const auto& cluster = clusters_device[globalIndex].items;
        const cluster_id& cl_id = clusters_device[globalIndex].header;

        const vector2 pitch = detail::get_pitch(cl_id);
        const auto module_idx = cl_id.module_idx;

        scalar totalWeight = 0.;

        // Get the cell module for this module idx
        const auto& module = cells_device[module_idx].header;

        // To calculate the mean and variance with high numerical
        // stability we use a weighted variant of Welford's
        // algorithm. This is a single-pass online algorithm that
        // works well for large numbers of samples, as well as
        // samples with very high values.
        //
        // To learn more about this algorithm please refer to:
        // [1] https://doi.org/10.1080/00401706.1962.10490022
        // [2] The Art of Computer Programming, Donald E. Knuth,
        // second
        //     edition, chapter 4.2.2.
        point2 mean = {0., 0.}, var = {0., 0.};

        // Should not happen
        assert(cluster.empty() == false);

        detail::calc_cluster_properties(cluster, cl_id, mean, var,
                                        totalWeight);

        if (totalWeight > 0.) {
            measurement m;
            // normalize the cell position
            m.local = mean;
            // normalize the variance
            m.variance[0] = var[0] / totalWeight;
            m.variance[1] = var[1] / totalWeight;
            // plus pitch^2 / 12
            m.variance =
                m.variance + point2{pitch[0] * pitch[0] / 12,
                                    pitch[1] * pitch[1] / 12};
            // @todo add variance estimation
            measurements_device[module_idx].header = module;
            measurements_device[module_idx].items.push_back(m);
        }
    }

} // namespace traccc::device
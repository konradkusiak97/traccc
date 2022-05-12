/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward decleration of spacepoint formation kernel
///
void spacepoint_formation(
    spacepoint_container_view spacepoints_view,
    vecmem::data::vector_view<measurement> measurements_view,
    const cell_container_types::host& cells_per_event,
    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl

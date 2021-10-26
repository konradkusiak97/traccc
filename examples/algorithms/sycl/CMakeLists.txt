add_library(traccc_algorithms_cuda SHARED
${CMAKE_CURRENT_SOURCE_DIR}/include/sycl/track_finding/seeding_algorithm.hpp)

target_link_libraries(traccc_algorithms_sycl
  PUBLIC traccc::io traccc::core traccc::sycl)

add_library(traccc::algorithms_sycl ALIAS traccc_algorithms_sycl)
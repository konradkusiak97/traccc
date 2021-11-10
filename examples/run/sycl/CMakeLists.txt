add_executable (seeding_example_sycl seeding_example_sycl.sycl)

target_link_libraries (seeding_example_sycl LINK_PUBLIC
  traccc::io
  traccc::algorithms
  traccc::algorithms_sycl)
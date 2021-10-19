add_library(traccc_sycl SHARED 
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/doublet_counting.cpp)

target_compile_options(
	traccc_sycl
	PRIVATE -fsycl)

target_link_libraries(traccc_sycl PUBLIC traccc::core vecmem::cuda)

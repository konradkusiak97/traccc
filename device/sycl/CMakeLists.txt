add_library(traccc_sycl SHARED 
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/doublet_counting.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/doublet_finding.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/seed_selecting.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/triplet_counting.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/triplet_finding.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/seeding/weight_updating.cpp)

target_compile_options(
	traccc_sycl
	PRIVATE -fsycl)

target_link_libraries(traccc_sycl PUBLIC traccc::core vecmem::cuda)

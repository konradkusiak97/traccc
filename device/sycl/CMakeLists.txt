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

target_include_directories(traccc_sycl PRIVATE "/atlas/software/cuda/11.4.1/x86_64-ubuntu1804/include")
target_link_libraries(traccc_sycl PUBLIC traccc::core vecmem::sycl)
add_library(traccc::sycl ALIAS traccc_sycl)

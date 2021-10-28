add_library(traccc_sycl SHARED 
# header files
include/seeding/detail/doublet_counter.hpp
include/seeding/detail/triplet_counter.hpp
include/seeding/detail/multiplet_estimator.hpp
include/seeding/doublet_counting.hpp
include/seeding/doublet_finding.hpp
include/seeding/seed_finding.hpp
include/seeding/seed_selecting.hpp
include/seeding/triplet_counting.hpp
include/seeding/triplet_finding.hpp
include/seeding/weight_updating.hpp
# implementation files
src/seeding/doublet_counting.cpp
src/seeding/doublet_finding.cpp
src/seeding/seed_selecting.cpp
src/seeding/triplet_counting.cpp
src/seeding/triplet_finding.cpp
src/seeding/weight_updating.cpp)

target_compile_options(
	traccc_sycl
	PRIVATE -fsycl)

target_include_directories(traccc_sycl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )
	
target_include_directories(traccc_sycl PRIVATE "/atlas/software/cuda/11.4.1/x86_64-ubuntu1804/include")
target_link_libraries(traccc_sycl PUBLIC traccc::core vecmem::sycl)

add_library(traccc::sycl ALIAS traccc_sycl)

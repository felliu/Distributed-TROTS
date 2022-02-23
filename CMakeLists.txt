cmake_minimum_required(VERSION 3.13)

project(trots_benchmark LANGUAGES CXX C)
find_package(HDF5 MODULE REQUIRED)

set(MKL_INTERFACE lp64)
find_package(MKL CONFIG REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
add_subdirectory(external/matio)

add_executable(trots_test
    trots.cpp
    trots.h
    TROTSEntry.cpp
    TROTSEntry.h
    SparseMat.h
    OptimizationProblem.h
    util.h
    main.cpp)

target_compile_features(trots_test PUBLIC cxx_std_17)
set_target_properties(trots_test
    PROPERTIES
        CXX_EXTENSIONS off)

target_compile_options(trots_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_compile_definitions(trots_test PUBLIC MKL_LP64)
target_include_directories(trots_test PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(trots_test PUBLIC matio::matio $<LINK_ONLY:MKL::MKL>)




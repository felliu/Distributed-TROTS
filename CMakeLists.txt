cmake_minimum_required(VERSION 3.13)

project(trots_benchmark LANGUAGES CXX C)
find_package(HDF5 MODULE REQUIRED)
set(MKL_INTERFACE lp64)

find_package(MKL CONFIG)

#In case we cannot find MKL through the cmake config, try to find it manually in LD_LIBRARY_PATH
if (NOT ${MKL_FOUND})
    if (DEFINED ENV{MKLROOT})
        set(MKL_LINK_FLAGS
                "-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64\
                 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
        set(MKL_INCLUDE_DIRS
                "-I${MKLROOT}/include")
        set(MKL_COMPILE_OPTIONS
                "-m64")
    endif()
else ()
    set(MKL_LINK_FLAGS $<LINK_ONLY:MKL::MKL>)
    set(MKL_INCLUDE_DIRS $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    set(MKL_COMPILE_OPTIONS $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
endif()

add_subdirectory(external/matio)

find_library(IPOPT
    ipopt
    HINTS ENV LD_LIBRARY_PATH
    REQUIRED)

add_executable(trots_test
    trots.cpp
    trots.h
    TROTSEntry.cpp
    TROTSEntry.h
    trots_ipopt.h
    trots_ipopt.cpp
    trots_matfile_data.cpp
    trots_matfile_data.h
    SparseMat.h
    OptimizationProblem.h
    util.h
    main.cpp)

target_compile_features(trots_test PUBLIC cxx_std_17)
set_target_properties(trots_test
    PROPERTIES
        CXX_EXTENSIONS off)

target_compile_options(trots_test PUBLIC ${MKL_COMPILE_OPTIONS})
target_compile_definitions(trots_test PUBLIC MKL_LP64)
if (${USE_MKL})
    target_compile_definitions(trots_test PUBLIC USE_MKL)
endif()
target_include_directories(trots_test PUBLIC ${MKL_INCLUDE_DIRS})
target_link_libraries(trots_test PUBLIC matio::matio ${MKL_LINK_FLAGS} ${IPOPT})




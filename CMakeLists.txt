cmake_minimum_required(VERSION 3.13)

project(trots_benchmark LANGUAGES CXX C)
set(MKL_INTERFACE lp64)

find_package(MKL CONFIG)
find_package(OpenMP REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
#In case we cannot find MKL through the cmake config
#assume we still can find it on the linker's search path and just
#specify link flags manually.
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
add_subdirectory(trots_lib)

add_subdirectory(ipopt_driver)
add_subdirectory(test)

if (${MPI})
    add_subdirectory(ipopt_mpi_driver)
endif()

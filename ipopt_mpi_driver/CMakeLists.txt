find_library(IPOPT
    ipopt
    HINTS ENV LD_LIBRARY_PATH
    REQUIRED
)

find_package(MPI REQUIRED)

add_executable(ipopt_mpi_main
    main.cpp
    data_distribution.h
    data_distribution.cpp
    globals.h
    rank_local_data.h
    sparse_matrix_transfers.cpp
    sparse_matrix_transfers.h
)

target_compile_features(ipopt_mpi_main PUBLIC cxx_std_17)
set_target_properties(ipopt_mpi_main
    PROPERTIES
        CXX_EXTENSIONS off)

target_include_directories(ipopt_mpi_main
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_link_libraries(ipopt_mpi_main
    PRIVATE
        ${MPI_CXX_LIBRARIES}
        trots_lib
        ${IPOPT})

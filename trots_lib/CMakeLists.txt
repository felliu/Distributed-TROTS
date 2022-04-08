set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)

find_package(Boost REQUIRED COMPONENTS serialization)

if(NOT TARGET Boost::serialization)
    add_library(Boost::serialization IMPORTED INTERFACE)
    set_property(TARGET Boost::serialization PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::serialization PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
endif()


add_library(trots_lib STATIC
    trots.cpp
    trots.h
    trots_matfile_data.cpp
    trots_matfile_data.h
    TROTSEntry.cpp
    TROTSEntry.h
    SparseMat.h
    MKL_sparse_matrix.h
    EigenSparseMat.h
    util.cpp
    util.h
)

target_compile_features(trots_lib PUBLIC cxx_std_17)
set_target_properties(trots_lib
    PROPERTIES
        CXX_EXTENSIONS off)

target_include_directories(trots_lib PUBLIC
    ../external/matio/src/
    .
)

target_link_libraries(trots_lib PUBLIC matio::matio Boost::serialization)

if (${USE_MKL})
    target_compile_options(trots_lib PUBLIC ${MKL_COMPILE_OPTIONS})
    target_compile_definitions(trots_lib PUBLIC MKL_LP64 USE_MKL)
    target_include_directories(trots_lib PUBLIC ${MKL_INCLUDE_DIRS})
    target_link_libraries(trots_lib PUBLIC ${MKL_LINK_FLAGS})
else ()
    find_package(OpenMP REQUIRED)
    target_link_libraries(trots_lib PUBLIC OpenMP::OpenMPI_CXX)
endif()

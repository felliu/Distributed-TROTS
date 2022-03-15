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

target_link_libraries(trots_lib PUBLIC matio::matio)

if (${USE_MKL})
    target_compile_options(trots_lib PUBLIC ${MKL_COMPILE_OPTIONS})
    target_compile_definitions(trots_lib PUBLIC MKL_LP64 USE_MKL)
    target_include_directories(trots_lib PUBLIC ${MKL_INCLUDE_DIRS})
    target_link_libraries(trots_lib PUBLIC ${MKL_LINK_FLAGS})
endif()

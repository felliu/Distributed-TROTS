add_executable(trots_test
    main.cpp
)

target_compile_features(trots_test PRIVATE cxx_std_17)
set_target_properties(trots_test
    PROPERTIES
        CXX_EXTENSIONS off)

target_link_libraries(trots_test PRIVATE trots_lib)


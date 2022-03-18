find_library(IPOPT
    ipopt
    HINTS ENV LD_LIBRARY_PATH
    REQUIRED
)

add_executable(ipopt_main
    main.cpp
    trots_ipopt.cpp
    trots_ipopt.h
)

target_compile_features(ipopt_main PUBLIC cxx_std_17)
set_target_properties(ipopt_main
    PROPERTIES
        CXX_EXTENSIONS off)

target_link_libraries(ipopt_main PUBLIC trots_lib ${IPOPT})

#ifndef UTIL_H
#define UTIL_H
#include <string>

struct matvar_t;

template <typename T>
void check_null(T* ptr, const std::string& msg) {
    if (ptr == NULL)
        throw std::runtime_error(msg);
}

template <typename DestType>
DestType cast_from_double(matvar_t* var) {
    assert(var->class_type == MAT_C_DOUBLE);
    double* data = static_cast<double*>(var->data);
    return static_cast<DestType>(*data);
}

#endif

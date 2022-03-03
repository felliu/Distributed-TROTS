#ifndef UTIL_H
#define UTIL_H
#include <string>
#include <fstream>

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

template <typename T>
void dump_vector_to_file(const std::vector<T>& vec, const std::string& path, bool append=false)
{
    std::ofstream outfile;
    if (append)
        outfile.open(path, std::ios::binary | std::ios::app);
    else
        outfile.open(path, std::ios::binary | std::ios::out);

    size_t sz = vec.size();
    outfile.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    outfile.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * sz);
}
#endif

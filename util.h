#ifndef UTIL_H
#define UTIL_H
#include <string>

template <typename T>
void check_null(T* ptr, const std::string& msg) {
    if (ptr == NULL)
        throw std::runtime_error(msg);
}

#endif
#ifndef UTILS_H
#define UTILS_H

#include <vector>

template <typename T>
void print_vector(const std::vector<T>& vec) {
    if (vec.empty()) {
        std::cout << "{}" << std::endl;
        return;
    }
    std::cout << "{";
    for (auto it = vec.cbegin(); it != (vec.cend() - 1); ++it)
        std::cout << *it << ", ";

    std::cout << vec[vec.size() - 1];
    std::cout << "}" << std::endl;
}

#endif
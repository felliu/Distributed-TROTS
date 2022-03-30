#ifndef UTIL_H
#define UTIL_H

#include <cassert>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <tuple>

#include "matio.h"

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

//Takes a CSC-matrix triplet and returns a CSR-matrix triplet
//Implementation basically same as the one used in SciPy.
template <typename ValueType>
std::tuple<ValueType*, int*, int*>
csc_to_csr(int rows, int cols,
           const ValueType* data,
           const int* row_idxs,
           const int* col_ptrs) {

    //Allocate arrays for the new CSR-matrix
    const int nnz = static_cast<int>(col_ptrs[cols]);
    ValueType* data_csr = new ValueType[nnz];
    int* col_idxs_csr = new int[nnz];
    int* row_ptrs_csr = new int[rows + 1];

    std::fill(row_ptrs_csr, row_ptrs_csr + rows, 0);

    //Compute the number of non-zeros per row, and store in row_ptrs_csr for now
    for (int i = 0; i < nnz; ++i) {
        row_ptrs_csr[row_idxs[i]]++;
    }

    int cumulative_sum = 0;
    //Compute the cumulative sum to get the actual row_ptr values
    for (int row = 0; row < rows; ++row) {
        const int tmp = row_ptrs_csr[row];
        row_ptrs_csr[row] = cumulative_sum;
        cumulative_sum += tmp;
    }
    row_ptrs_csr[rows] = nnz;

    for (int col = 0; col < cols; ++col) {
        for (int i = col_ptrs[col]; i < col_ptrs[col + 1]; ++i) {
            const int row = row_idxs[i];
            const int dest_idx = row_ptrs_csr[row];

            col_idxs_csr[dest_idx] = col;
            data_csr[dest_idx] = data[i];

            row_ptrs_csr[row]++;
        }
    }

    int last = 0;
    for (int row = 0; row <= rows; ++row) {
        const int tmp = row_ptrs_csr[row];
        row_ptrs_csr[row] = last;
        last = tmp;
    }

    return std::make_tuple(data_csr, col_idxs_csr, row_ptrs_csr);
}

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

std::string get_name_str(const matvar_t* name_var);

#endif

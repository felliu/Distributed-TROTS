#ifndef MATLAB_SPARSE_MAT_H
#define MATLAB_SPARSE_MAT_H

#include <algorithm>
#include <cassert>
#include <cstring> //for std::memcpy
#include <iostream>
#include <utility>
#include <mkl.h>

namespace {
    bool check_MKL_status(sparse_status_t status) {
        switch (status) {
            case SPARSE_STATUS_SUCCESS: 
                return true;
            case SPARSE_STATUS_NOT_INITIALIZED:
                std::cerr << "Not initialized error encountered in sparse MKL.\n";
                return false;
            case SPARSE_STATUS_ALLOC_FAILED:
                std::cerr << "Allocation failure encountered in sparse MKL.\n";
                return false;
            case SPARSE_STATUS_INVALID_VALUE:
                std::cerr << "Input parameter invalid in sparse MKL.\n";
                return false;
            case SPARSE_STATUS_EXECUTION_FAILED:
                std::cerr << "Execution failure in sparse MKL.\n";
                return false;
            case SPARSE_STATUS_INTERNAL_ERROR:
                std::cerr << "Internal error in sparse MKL.\n";
                return false;
            case SPARSE_STATUS_NOT_SUPPORTED:
                std::cerr << "Operaton not supported error in sparse MKL.\n";
                return false;
        }

        return false;
    }
}

//Internally, we might store the data in CSC format, since we read them from matlab.
enum class StorageTypeInternal {
    CSR,
    CSC
};

template <typename T>
class MKL_sparse_matrix {
using MKL_mat_handle = sparse_matrix_t;
public:
    template <typename IdxType>
    static MKL_sparse_matrix from_CSC_mat(int nnz, int rows, int cols, 
                                          const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs);
    MKL_sparse_matrix() = default;

    MKL_sparse_matrix(const MKL_sparse_matrix& rhs);
    MKL_sparse_matrix& operator=(MKL_sparse_matrix rhs);

    MKL_sparse_matrix(MKL_sparse_matrix&& rhs);

    friend void swap(MKL_sparse_matrix& m1, MKL_sparse_matrix& m2) {
        std::swap(m1.data, m2.data);
        std::swap(m1.indices, m2.indices);
        std::swap(m1.indptrs, m2.indptrs);
        std::swap(m1.sp_type, m2.sp_type);
        std::swap(m1.mkl_handle, m2.mkl_handle);
        std::swap(m1.nnz, m2.nnz);
        std::swap(m1.rows, m2.rows);
        std::swap(m1.cols, m2.cols);
    }

    ~MKL_sparse_matrix();
private:
    void init_mkl_handle();
    StorageTypeInternal sp_type;
    MKL_mat_handle mkl_handle;
    size_t nnz, rows, cols;
    T* data;
    int* indices; //Col indices if CSR and row indices if CSC.
    int* indptrs; 
};

template <typename T>
MKL_sparse_matrix<T>::MKL_sparse_matrix(const MKL_sparse_matrix<T>& rhs) {
    static_assert(std::is_floating_point_v<T>);
    std::cerr << "Copy constructing sparse matrix.\n";
    this->nnz = rhs.nnz;
    this->rows = rhs.rows;
    this->cols = rhs.cols;
    this->sp_type = rhs.sp_type;

    this->data = new T[this->nnz];
    this->indices = new int[this->nnz];
    this->indptrs = new int[this->cols + 1];

    std::memcpy(this->data, rhs.data, sizeof(T) * this->nnz);
    std::memcpy(this->indices, rhs.indices, sizeof(int) * this->nnz);
    std::memcpy(this->indptrs, rhs.indptrs, sizeof(int) * (this->cols + 1));

    this->init_mkl_handle();    
}

template <typename T>
MKL_sparse_matrix<T>& MKL_sparse_matrix<T>::operator=(MKL_sparse_matrix rhs) {
    swap(*this, rhs);
    return *this;
}

template <typename T>
MKL_sparse_matrix<T>::MKL_sparse_matrix(MKL_sparse_matrix&& other) {
    std::cerr << "Move constructing sparse matrix.\n";
    this->nnz = other.nnz;
    this->rows = other.rows;
    this->cols = other.cols;
    this->sp_type = other.sp_type;
    this->mkl_handle = other.mkl_handle;

    this->data = other.data;
    this->indices = other.indices;
    this->indptrs = other.indptrs;

    other.data = nullptr;
    other.indices = nullptr;
    other.indptrs = nullptr;
    other.mkl_handle = nullptr;
}

template <typename T>
MKL_sparse_matrix<T>::~MKL_sparse_matrix() {
    delete[] this->data;
    delete[] this->indices;
    delete[] this->indptrs;
    mkl_sparse_destroy(this->mkl_handle);
}

template <typename T>
template <typename IdxType>
MKL_sparse_matrix<T>
MKL_sparse_matrix<T>::from_CSC_mat(int nnz, int rows, int cols, const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    MKL_sparse_matrix<T> mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.nnz = nnz;
    mat.sp_type = StorageTypeInternal::CSC;
    mat.data = new T[nnz];
    mat.indices = new int[nnz];
    mat.indptrs = new int[cols + 1];

    std::memcpy(mat.data, vals, sizeof(T) * nnz);
    //Assuming here that the narrowing cast to int32_t from things like uint32_t will fit.
    std::transform(row_idxs, row_idxs + nnz, mat.indices, [](auto x) { return static_cast<int>(x); });
    std::transform(col_ptrs, col_ptrs + cols + 1, mat.indptrs, [](auto x) { return static_cast<int>(x); });

    mat.init_mkl_handle();
    return mat;
}

template <typename T>
void MKL_sparse_matrix<T>::init_mkl_handle() {
static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if (sp_type == StorageTypeInternal::CSC) {
        sparse_matrix_t csc_tmp_handle = nullptr;
        if constexpr (std::is_same_v<T, double>) {
            const auto status = mkl_sparse_d_create_csc(&csc_tmp_handle,
                                                        SPARSE_INDEX_BASE_ZERO,
                                                        this->rows, this->cols,
                                                        this->indptrs, this->indptrs + 1,
                                                        this->indices, this->data);
            assert(check_MKL_status(status));
        }
        else if constexpr (std::is_same_v<T, float>) {
            const auto status = mkl_sparse_s_create_csc(&csc_tmp_handle,
                                                        SPARSE_INDEX_BASE_ZERO,
                                                        this->rows, this->cols,
                                                        this->indptrs, this->indptrs + 1,
                                                        this->indices, this->data);
            assert(check_MKL_status(status));
        }

        this->mkl_handle = nullptr;
        mkl_sparse_convert_csr(csc_tmp_handle, SPARSE_OPERATION_NON_TRANSPOSE, &this->mkl_handle);
        mkl_sparse_destroy(csc_tmp_handle);
    } else {
        throw "Internal storage type CSR not implemented yet!\n";
    }
}



#endif
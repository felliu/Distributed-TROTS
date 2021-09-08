#ifndef MATLAB_SPARSE_MAT_H
#define MATLAB_SPARSE_MAT_H

#include <algorithm>
#include <cstring> //for std::memcpy
#include <utility>
#include <mkl.h>

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
    MKL_sparse_matrix& operator=(MKL_sparse_matrix&& rhs);

    friend void swap(MKL_sparse_matrix& m1, MKL_sparse_matrix& m2) {
        std::swap(m1.data, m2.data);
        std::swap(m1.indices, m2.indices);
        std::swap(m1.indptrs, m2.indptrs);
        std::swap(m1.sp_type, m2.sp_type);
        std::swap(m1.mkl_handle, m2.mkl_handle);
    }

    ~MKL_sparse_matrix();
private:
    StorageTypeInternal sp_type;
    MKL_mat_handle mkl_handle;
    size_t nnz, rows, cols;
    T* data;
    int* indices; //Col indices if CSR and row indices if CSC.
    int* indptrs; 
};

template <typename T>
MKL_sparse_matrix<T>::MKL_sparse_matrix(const MKL_sparse_matrix<T>& rhs) {
    this->nnz = rhs.nnz;
    this->rows = rhs.rows;
    this->cols = rhs.cols;
    this->sp_type = rhs.sp_type;
    this->mkl_handle = rhs.mkl_handle;
    std::memcpy(this->data, rhs.data, sizeof(T) * rhs.nnz);
    std::memcpy(this->indices, rhs.indices, sizeof(int) * rhs.nnz);
    std::memcpy(this->indptrs, rhs.indptrs, sizeof(int) * (rhs.cols + 1));
}

template <typename T>
MKL_sparse_matrix<T>& MKL_sparse_matrix<T>::operator=(MKL_sparse_matrix rhs) {
    swap(*this, rhs);
    return *this;
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
    static_assert(std::is_floating_point_v<T>);
    MKL_sparse_matrix<T> mat;
    mat.sp_type = StorageTypeInternal::CSC;
    mat.data = new T[nnz];
    mat.indices = new int[nnz];
    mat.indptrs = new int[rows + 1];

    std::memcpy(mat.data, vals, sizeof(T) * nnz);
    //Assuming here that the narrowing cast to int32_t from things like uint32_t will fit.
    std::transform(row_idxs, row_idxs + nnz, mat.indices, [](auto x) { return static_cast<int>(x); });
    std::transform(col_ptrs, col_ptrs + rows + 1, mat.indptrs, [](auto x) { return static_cast<int>(x); });

    //We have sparse matrix data in csc format, but want csr format for better parallel performance in MKL.
    //Create a temporary handle for the csc format matrix, which we will then convert to csr using MKL functions.
    sparse_matrix_t csc_tmp_handle;
    if constexpr (std::is_same_v<T, double>)
        mkl_sparse_d_create_csc(&csc_tmp_handle, SPARSE_INDEX_BASE_ZERO, rows, cols, mat.indptrs, mat.indptrs + 1, mat.indices, mat.data);
    
    if constexpr (std::is_same_v<T, float>)
        mkl_sparse_s_create_csc(&csc_tmp_handle, SPARSE_INDEX_BASE_ZERO, rows, cols, mat.indptrs, mat.indptrs + 1, mat.indices, mat.data);
    
    mkl_sparse_convert_csr(csc_tmp_handle, SPARSE_OPERATION_NON_TRANSPOSE, &mat.mkl_handle);
    mkl_sparse_destroy(csc_tmp_handle);

    return mat;
}



#endif
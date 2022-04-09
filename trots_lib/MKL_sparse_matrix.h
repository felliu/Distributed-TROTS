
#ifndef MKL_SPARSE_MAT_H
#define MKL_SPARSE_MAT_H

#include <algorithm>
#include <cassert>
#include <cstring> //for std::memcpy
#include <iostream>
#include <numeric>
#include <memory>

#include <mkl.h>

#include "SparseMat.h"
#include "util.h"

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


    matrix_descr desc{ SPARSE_MATRIX_TYPE_GENERAL,
                       SPARSE_FILL_MODE_LOWER,
                       SPARSE_DIAG_NON_UNIT};
}

template <typename T>
class MKL_sparse_matrix : public SparseMatrix<T> {
using MKL_mat_handle = sparse_matrix_t;
public:
    template <typename IdxType>
    static std::unique_ptr<SparseMatrix<T>>
    from_CSC_mat(int nnz, int rows, int cols,
                 const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs);

    static std::unique_ptr<SparseMatrix<T>>
    from_CSR_mat(int nnz, int rows, int cols,
                 const T* vals, const int* col_idxs, const int* row_ptrs);

    MKL_sparse_matrix() = default;

    MKL_sparse_matrix(const MKL_sparse_matrix& rhs);
    MKL_sparse_matrix& operator=(MKL_sparse_matrix rhs);

    MKL_sparse_matrix(MKL_sparse_matrix&& rhs);

    int get_rows() const noexcept override {
        return this->rows;
    }

    int get_cols() const noexcept override {
        return this->cols;
    }

    int get_nnz() const noexcept override {
        return this->nnz;
    }

    const int* get_col_inds() const noexcept override {
        return this->indices;
    }

    const int* get_row_ptrs() const noexcept override {
        return this->indptrs;
    }

    const T* get_data_ptr() const noexcept override {
        return this->data;
    }

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

    //Computes A * x and stores result in y.
    void vec_mul(const T* x, T* y) const override;
    //Computes A^T * x and stores result in y.
    void vec_mul_transpose(const T* x, T* y) const override;
    //Computes the value of the quadratic form x^T * A * x and returns the value.
    //The value A * x is stored in y.
    T quad_mul(const T* x, T* y) const;

    void check_consistency() const;

    ~MKL_sparse_matrix();

private:
    void vec_mul_impl(const T* x, T* y, bool transpose = false) const;
    void init_mkl_handle();
    MKL_mat_handle mkl_handle;
    int nnz, rows, cols;
    T* data;
    int* indices;
    int* indptrs;
};

template <typename T>
MKL_sparse_matrix<T>::MKL_sparse_matrix(const MKL_sparse_matrix<T>& rhs) {
    static_assert(std::is_floating_point_v<T>);
    this->nnz = rhs.nnz;
    this->rows = rhs.rows;
    this->cols = rhs.cols;

    this->data = new T[this->nnz];
    this->indices = new int[this->nnz];
    this->indptrs = new int[this->rows + 1];

    std::memcpy(this->data, rhs.data, sizeof(T) * this->nnz);
    std::memcpy(this->indices, rhs.indices, sizeof(int) * this->nnz);
    std::memcpy(this->indptrs, rhs.indptrs, sizeof(int) * (this->rows + 1));

    this->init_mkl_handle();
}

template <typename T>
MKL_sparse_matrix<T>& MKL_sparse_matrix<T>::operator=(MKL_sparse_matrix rhs) {
    swap(*this, rhs);
    return *this;
}

template <typename T>
MKL_sparse_matrix<T>::MKL_sparse_matrix(MKL_sparse_matrix&& other) {
    this->nnz = other.nnz;
    this->rows = other.rows;
    this->cols = other.cols;
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
std::unique_ptr<SparseMatrix<T>>
MKL_sparse_matrix<T>::from_CSC_mat(int nnz, int rows, int cols, const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    mkl_set_num_threads(28);
    MKL_sparse_matrix<T>* mat = new MKL_sparse_matrix<T>();

    std::vector<int> row_idxs_int;
    std::vector<int> col_ptrs_int;
    row_idxs_int.reserve(nnz);
    col_ptrs_int.reserve(cols + 1);
    //Assuming here that the narrowing cast to int32_t from things like uint32_t will fit.
    std::transform(row_idxs, row_idxs + nnz, std::back_inserter(row_idxs_int), [](auto x) { return static_cast<int>(x); });
    std::transform(col_ptrs, col_ptrs + cols + 1, std::back_inserter(col_ptrs_int), [](auto x) { return static_cast<int>(x); });

    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = nnz;
    std::tie(mat->data, mat->indices, mat->indptrs) = csc_to_csr(rows, cols, vals, &row_idxs_int[0], &col_ptrs_int[0]);

    mat->init_mkl_handle();
    return std::unique_ptr<MKL_sparse_matrix<T>>(mat);
}

template <typename T>
std::unique_ptr<SparseMatrix<T>>
MKL_sparse_matrix<T>::from_CSR_mat(int nnz, int rows, int cols,
                                   const T* vals, const int* col_idxs, const int* row_ptrs) {
    MKL_sparse_matrix<T>* mat = new MKL_sparse_matrix<T>();
    mat->rows = rows;
    mat->cols = cols;
    mat->nnz = nnz;

    mat->data = new T[nnz];
    mat->indices = new int[nnz];
    mat->indptrs = new int[rows + 1];

    std::memcpy(static_cast<void*>(mat->data), static_cast<const void*>(vals), sizeof(T) * nnz);
    std::memcpy(static_cast<void*>(mat->indices), static_cast<const void*>(col_idxs), sizeof(int) * nnz);
    std::memcpy(static_cast<void*>(mat->indptrs), static_cast<const void*>(row_ptrs), sizeof(int) * (rows + 1));

    mat->init_mkl_handle();
    return std::unique_ptr<MKL_sparse_matrix<T>>(mat);
}

template <typename T>
void MKL_sparse_matrix<T>::init_mkl_handle() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    sparse_status_t status = SPARSE_STATUS_SUCCESS;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_create_csr(&this->mkl_handle,
                                         SPARSE_INDEX_BASE_ZERO,
                                         this->rows, this->cols,
                                         this->indptrs, this->indptrs + 1,
                                         this->indices, this->data);
        assert(check_MKL_status(status));
    }
    else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_create_csr(&this->mkl_handle,
                                         SPARSE_INDEX_BASE_ZERO,
                                         this->rows, this->cols,
                                         this->indptrs, this->indptrs + 1,
                                         this->indices, this->data);
        assert(check_MKL_status(status));
    }
    status = mkl_sparse_set_mv_hint(this->mkl_handle,
                                    SPARSE_OPERATION_NON_TRANSPOSE,
                                    desc, 3000);
    assert(check_MKL_status(status));
    status = mkl_sparse_set_mv_hint(this->mkl_handle,
                                    SPARSE_OPERATION_TRANSPOSE,
                                    desc, 3000);
    assert(check_MKL_status(status));
    status = mkl_sparse_optimize(this->mkl_handle);

    assert(check_MKL_status(status));
}

template <typename T>
void MKL_sparse_matrix<T>::vec_mul_transpose(const T* x, T* y) const {
    this->vec_mul_impl(x, y, true);
}

template <typename T>
void MKL_sparse_matrix<T>::vec_mul(const T* x, T* y) const {
    this->vec_mul_impl(x, y, false);
}

template <typename T>
void MKL_sparse_matrix<T>::vec_mul_impl(const T* x, T* y, bool transpose) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    sparse_status_t status = SPARSE_STATUS_SUCCESS;
    const sparse_operation_t trans_op = transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;

    if constexpr (std::is_same_v<T, double>)
        status = mkl_sparse_d_mv(trans_op, 1.0, this->mkl_handle, desc, x, 0.0, y);
    else
        status = mkl_sparse_s_mv(trans_op, 1.0, this->mkl_handle, desc, x, 0.0, y);

    assert(check_MKL_status(status));
}

template <typename T>
T MKL_sparse_matrix<T>::quad_mul(const T* x, T* y) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    sparse_status_t status = SPARSE_STATUS_SUCCESS;
    T val = 0.0;

    if constexpr (std::is_same_v<T, double>)
        status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, this->mkl_handle, desc, x, 0.0, y, &val);
    else
        status = mkl_sparse_s_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, this->mkl_handle, desc, x, 0.0, y, &val);

    assert(check_MKL_status(status));

    return val;
}

//This is intended to be a debug function, to check if some aspects of MKL's internal representation is consistent with
//our representation. Checks that MKL and we think that the matrix has the same number of rows, columns and non-zeros.
template <typename T>
void MKL_sparse_matrix<T>::check_consistency() const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    int rows_mkl, cols_mkl;

    sparse_index_base_t idx_base;
    int* rows_end;
    int* rows_start;
    int* col_idxs;
    sparse_status_t status = SPARSE_STATUS_SUCCESS;
    if constexpr (std::is_same_v<T, double>) {
        double* data;
        status = mkl_sparse_d_export_csr(this->mkl_handle, &idx_base, &rows_mkl, &cols_mkl, &rows_start, &rows_end, &col_idxs, &data);
    } else {
        float* data;
        status = mkl_sparse_s_export_csr(this->mkl_handle, &idx_base, &rows_mkl, &cols_mkl, &rows_start, &rows_end, &col_idxs, &data);
    }

    const int nnz_mkl = rows_end[this->rows-1];
    assert(this->rows == rows_mkl);
    assert(this->cols == cols_mkl);
    assert(this->nnz == nnz_mkl);
    assert(idx_base == SPARSE_INDEX_BASE_ZERO);
    assert(check_MKL_status(status));
}


#endif
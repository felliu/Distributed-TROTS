#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

template <typename T>
struct SparseMatrix {
    virtual void vec_mul(const T* x, T* y) const = 0;
    virtual void vec_mul_transpose(const T* x, T* y) const = 0;
    virtual T quad_mul(const T* x, T* y) const = 0;

    virtual int get_rows() const = 0;
    virtual int get_cols() const = 0;
    virtual int get_nnz() const = 0;
    virtual const int* get_col_inds() const = 0;
    virtual const int* get_row_ptrs() const = 0;
    virtual const T* get_data_ptr() const = 0;

    virtual ~SparseMatrix() = default;
};




#endif
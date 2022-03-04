#ifndef EIGEN_SPARSE_MAT_H
#define EIGEN_SPARSE_MAT_H

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <tuple>
#include <vector>
#include <type_traits>

#include "SparseMat.h"

namespace {
    template <typename T>
    std::vector<Eigen::Triplet<T>>
    csc_to_triplet(int rows, int cols,
                   const T* vals, const int* row_idxs, const int* col_ptrs) {
        const int nnz = col_ptrs[cols];
        std::vector<Eigen::Triplet<T>> triplets;
        triplets.reserve(nnz);

        for (int col = 0; col <= cols; ++cols)
            for (int idx = col_ptrs[col]; idx < col_ptrs[col + 1]; ++idx)
                triplets.push_back(Eigen::Triplet<double>(row_idxs[idx], col, vals[idx]));

        return triplets;
    }
}

template <typename T>
class EigenSparseMat : public SparseMatrix<T> {
public:
    template <typename IdxType>
    static EigenSparseMat<T> from_CSC_mat(int nnz, int rows, int cols,
                                          const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs);

    explicit EigenSparseMat(int rows, int cols) : mat(rows, cols) {}

    int get_rows() const override { return this->mat.rows(); }
    int get_cols() const override { return this->mat.cols(); }
    int get_nnz() const override { return this->mat.nonZeros(); }
    int* get_col_inds() const override { return this->mat.innerIndexPtr(); }

    void vec_mul(const T* x, T* y) const override;
    void vec_mul_transpose(const T* x, T* y) const override;
    T quad_mul(const T* x, T* y) const override;
private:
    Eigen::SparseMatrix<T, Eigen::RowMajor, int> mat;
};

template <typename T>
template <typename IdxType>
EigenSparseMat<T>
EigenSparseMat<T>::from_CSC_mat(
    int nnz, int rows, int cols,
    const T* vals, const IdxType* row_idxs, const IdxType* col_ptrs) {

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    EigenSparseMat<T> mat(rows, cols);

    std::vector<Eigen::Triplet<T>> triplets =
        csc_to_triplet(rows, cols, vals, row_idxs, col_ptrs);

    mat.mat.setFromTriplets(triplets.cbegin(), triplets.cend());
}

template <typename T>
void EigenSparseMat<T>::vec_mul(const T* x, T* y) const {
    Eigen::Map<Eigen::RowVectorXd> x_mp(x, this->get_cols());
    Eigen::Map<Eigen::RowVectorXd> y_mp(y, this->get_rows());

    y_mp = this->mat * x_mp;
}

template <typename T>
void EigenSparseMat<T>::vec_mul_transpose(const T* x, T* y) const {
    Eigen::Map<Eigen::RowVectorXd> x_mp(x, this->get_rows());
    Eigen::Map<Eigen::RowVectorXd> y_mp(y, this->get_cols());

    y_mp = this->mat.transpose() * x_mp;
}

template <typename T>
T EigenSparseMat<T>::quad_mul(const T* x, T* y) const {
}

#endif
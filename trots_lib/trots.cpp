#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>

#include "trots.h"
#include "util.h"

#ifdef USE_MKL
#include "MKL_sparse_matrix.h"
#else
#include "EigenSparseMat.h"
#endif


namespace {
    std::vector<double> get_mean_vector(matvar_t* matrix_entry) {
        matvar_t* matrix_data_var = Mat_VarGetStructFieldByName(matrix_entry, "A", 0);
        check_null(matrix_data_var, "Could not read matrix A from struct.\n");
        assert(matrix_data_var->dims[0] == 1); //Should be a pure column vector
        size_t num_elems = matrix_data_var->dims[1];

        //The mean vectors are stored in single precision, for whatever reason.
        assert(matrix_data_var->data_type == MAT_T_SINGLE);
        std::vector<double> A;
        A.reserve(num_elems);
        float* data = static_cast<float*>(matrix_data_var->data);
        std::transform(data, data + num_elems, std::back_inserter(A), [](const float x) { return static_cast<double>(x); });

        return A;
    }

    //Reads the Matlab sparse matrix stored at idx dataID - 1 and converts it to a CSR format and MKL sparse type.
    std::unique_ptr<SparseMatrix<double>> read_and_cvt_sparse_mat(matvar_t* matrix_entry) {
        matvar_t* matrix_data_var = Mat_VarGetStructFieldByName(matrix_entry, "A", 0);
        check_null(matrix_data_var, "Could not read matrix A from struct.\n");
        assert(matrix_data_var->class_type == MAT_C_SPARSE);
        assert(matrix_data_var->data_type == MAT_T_DOUBLE);

        mat_sparse_t* matlab_sparse_m = static_cast<mat_sparse_t*>(matrix_data_var->data);
        const int nnz = static_cast<int>(matlab_sparse_m->ndata);
        const int rows = static_cast<int>(matrix_data_var->dims[0]);
        const int cols = static_cast<int>(matrix_data_var->dims[1]);

#ifdef USE_MKL
        return MKL_sparse_matrix<double>::from_CSC_mat(nnz, rows, cols,
                                                       static_cast<double*>(matlab_sparse_m->data),
                                                       matlab_sparse_m->ir,
                                                       matlab_sparse_m->jc);
#else
        return EigenSparseMat<double>::from_CSC_mat(nnz, rows, cols,
                                                    static_cast<double*>(matlab_sparse_m->data),
                                                    matlab_sparse_m->ir,
                                                    matlab_sparse_m->jc);
#endif
    }
}


TROTSProblem::TROTSProblem(TROTSMatFileData&& trots_data_) :
    trots_data{std::move(trots_data_)}
{
    matvar_t* problem_struct = this->trots_data.problem_struct;
    size_t num_entries = problem_struct->dims[1];

    this->read_dose_matrices();

    int stride[] = {0, 0};
    int edge[] = {1, 1};
    this->nnz_jac_cons = 0;
    for (int i = 0; i < num_entries; ++i) {
        std::cerr << "Reading trots entry " << i << " of " << num_entries << "...\n";
        int start[] =  {0, i};
        matvar_t* struct_elem = Mat_VarGetStructs(problem_struct, start, stride, edge, 0);
        const TROTSEntry entry{struct_elem, this->trots_data.matrix_struct, this->matrices};
        std::cerr << "TROTSEntry read!\n\n";

        if (!entry.is_active()) {
            std::cout << "Entry: " << entry.get_roi_name() << " skipped\n";
            continue;
        }

        if (entry.is_constraint()) {
            /*if (entry.function_type() != FunctionType::Mean)
                continue;*/
            this->constraint_entries.push_back(entry);
            auto vec = entry.get_grad_nonzero_idxs();
            this->nnz_jac_cons += vec.size();
        } else {
            /*if (entry.function_type() != FunctionType::Mean)
                continue;*/
            this->objective_entries.push_back(entry);
        }
    }

    matvar_t* misc_struct = Mat_VarGetStructFieldByName(this->trots_data.data_struct, "misc", 0);
    matvar_t* size_var = Mat_VarGetStructFieldByName(misc_struct, "size", 0);
    this->num_vars = cast_from_double<int>(size_var);
}

void TROTSProblem::read_dose_matrices() {
    int stride[] = {0, 0};
    int edge[] = {1, 1};

    size_t num_matrices = this->trots_data.matrix_struct->dims[1];
    this->matrices.reserve(num_matrices);
    for (int i = 0; i < num_matrices; ++i) {
        std::cerr << "Reading dose matrix " << i + 1 << " of " << num_matrices << "...\n";
        int start[] = {0, i};
        matvar_t* matrix_entry = Mat_VarGetStructs(this->trots_data.matrix_struct, start, stride, edge, 1);
        check_null(matrix_entry, "Failed to read entry " + std::to_string(i) + " from matrix.data\n");
        matvar_t* A = Mat_VarGetStructFieldByName(matrix_entry, "A", 0);
        check_null(A, "Failed to read A from entry " + std::to_string(i) + " in matrix.data\n");
        auto& new_variant = this->matrices.emplace_back();

        //For the mean functions, the "A"-matrix is reduced to a dense vector.
        //Check if we have a sparse matrix or dense vector
        if (A->class_type == MAT_C_SPARSE) {
            new_variant.emplace<std::unique_ptr<SparseMatrix<double>>>(
                read_and_cvt_sparse_mat(matrix_entry)
            );
        }
        else {
            new_variant.emplace<std::vector<double>>(
                get_mean_vector(matrix_entry)
            );
        }

        //Avoid storing the matrix data twice.
        Mat_VarFree(matrix_entry);
    }
}


double TROTSProblem::calc_objective(const double* x) const {
    double sum = 0.0;
    for (const auto& entry : this->objective_entries) {
        sum += entry.get_weight() * entry.calc_value(x);
    }
    return sum;
}

void TROTSProblem::calc_obj_gradient(const double* x, double* y) const {
    std::fill(y, y + this->num_vars, 0.0);
    std::vector<double> grad_tmp(this->num_vars);

    for (const auto& entry : this->objective_entries) {
        entry.calc_gradient(x, &grad_tmp[0]);
        for (int i = 0; i < grad_tmp.size(); ++i) {
            double weight = entry.get_weight();
            /*if (entry.function_type() == FunctionType::Mean) {
                weight *= entry.get_weight();
            }*/
            y[i] += weight * grad_tmp[i];
        }
    }
}
void TROTSProblem::calc_jacobian_vals(const double* x, double* jacobian_vals, bool cached_dose) const {
    int idx = 0;
    for (const auto& constraint_entry : constraint_entries) {
        std::vector<double> grad_vals =
            constraint_entry.calc_sparse_grad(x, cached_dose);
        for (double v : grad_vals) {
            jacobian_vals[idx] = v;
            ++idx;
        }
    }
}

void TROTSProblem::calc_constraints(const double* x, double* cons_vals, bool cached_dose) const {
    for (int i = 0; i < this->constraint_entries.size(); ++i) {
        cons_vals[i] = this->constraint_entries[i].calc_value(x, cached_dose);
    }
}





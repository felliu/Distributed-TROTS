#ifndef TROTS_H
#define TROTS_H

#include <cassert>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <matio.h>
#include <mkl.h>

#include "SparseMat.h"
#include "TROTSEntry.h"

struct TROTSMatFileData {
public:
    TROTSMatFileData(const std::filesystem::path& file_path);

    TROTSMatFileData(const TROTSMatFileData& other) = delete;
    TROTSMatFileData& operator=(const TROTSMatFileData& rhs) = delete;

    TROTSMatFileData(TROTSMatFileData&& data);
    ~TROTSMatFileData();

    mat_t* file_fp;
    matvar_t* problem_struct;
    matvar_t* data_struct;
    matvar_t* matrix_struct;
private:
    void init_problem_data_structs();
};

class TROTSProblem {
public:
    TROTSProblem(TROTSMatFileData&& trots_data);

    //TODO: Make these private
    std::vector<TROTSEntry> objective_entries;
    std::vector<TROTSEntry> constraint_entries;
    int get_num_vars() const noexcept { return this->num_vars; }
    int get_nnz_jac_cons() const noexcept { return this->nnz_jac_cons; }
    int get_num_constraints() const noexcept {
        return this->constraint_entries.size();
    }
    double calc_objective(const double* x) const;
    void calc_obj_gradient(const double* x, double* y) const;
    void calc_constraints(const double* x, double* cons_vals) const;
    void calc_jacobian_vals(const double* x, double* jacobian_vals) const;


private:
    void read_dose_matrices();

    int num_vars;
    int nnz_jac_cons;
    TROTSMatFileData trots_data;
    //List of matrix entries, indexed by dataID.
    //If the FunctionType is mean, the value is computed using a dot product with a dense vector,
    //In other cases, the dose is calculated using a dose deposition matrix.
    std::vector<std::variant<std::unique_ptr<SparseMatrix<double>>, std::vector<double>>> matrices;
};

#endif

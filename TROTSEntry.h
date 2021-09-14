#ifndef TROTS_ENTRY_H
#define TROTS_ENTRY_H

#include <variant>

enum class FunctionType {
    Min, Max, Mean, Quadratic,
    gEUD, LTCP, DVH, Chain
};

class TROTSEntry {
public:
    TROTSEntry(matvar_t* problem_struct_entry, matvar_t* data_struct,
               const std::vector<std::variant<MKL_sparse_matrix<double>,
                                              std::vector<double>>
                                >& mat_refs);
    bool is_constraint() const noexcept { return this->is_cons; }
    double calc_value(const double* x) const;
    FunctionType function_type() const noexcept { return this->type; }
private:
    double calc_quadratic(const double* x) const;
    int id;
    std::string roi_name;
    std::vector<double> func_params;

    bool minimise;
    bool is_cons;

    FunctionType type;
    double rhs;
    double weight;
    //Multiple objectives / constraints can use the same dose deposition matrix. To avoid storing duplicates,
    //all the matrices are stored in TROTSProblem, and the TROTSEntries each have a reference to their matrix instead.
    const std::variant<MKL_sparse_matrix<double>, std::vector<double>>* matrix_ref;
    double c; //Scalar factor used in quadratic cost functions.

    //When the objective function is quadratic, the MKL sparse BLAS function mkl_sparse_d_dotmv
    //needs an input array for the matrix-vector product result, even though we are not interested in it in our case.
    mutable std::vector<double> dot_product_tmp;
};


#endif

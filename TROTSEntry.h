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
    bool is_constraint() const { return this->is_cons; }
private:
    int id;
    std::string roi_name;
    std::vector<double> func_params;

    bool minimise;
    bool is_cons;

    FunctionType type;
    double rhs;
    double weight;
    const std::variant<MKL_sparse_matrix<double>, std::vector<double>>* matrix_ref;
    double c; //Scalar factor used in quadratic cost functions.
};


#endif
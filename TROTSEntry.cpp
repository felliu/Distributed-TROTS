#include <cassert>
#include <vector>

#include <matio.h>

#include "SparseMat.h"
#include "TROTSEntry.h"
#include "util.h"

namespace {
    const char* func_type_names[] = {"Min", "Max", "Mean", "Quadratic", "gEUD", "LTCP", "DVH", "Chain"};

    FunctionType get_linear_function_type(int dataID, bool minimise, const std::string& roi_name, matvar_t* matrix_struct) {
        const int zero_indexed_dataID = dataID - 1;
        std::cerr << "Reading name field\n";
        //The times when the function type is mean, the data.matrix struct should have an entry at dataID with the name "<ROI_name> + (mean)"...
        matvar_t* matrix_entry_name_var = Mat_VarGetStructFieldByName(matrix_struct, "Name", zero_indexed_dataID);
        check_null(matrix_entry_name_var, "Failed to read name field of matrix entry\n.");

        std::string matrix_entry_name(static_cast<char*>(matrix_entry_name_var->data));
        std::cerr << "Matrix name: " << matrix_entry_name << "\n";

        auto n = matrix_entry_name.find("(mean)");
        if (n != std::string::npos)
            return FunctionType::Mean;

        return minimise ? FunctionType::Max : FunctionType::Min;
    }

    FunctionType get_nonlinear_function_type(int type_id) {
        assert(type_id >= 2);
        //The type ID is one-indexed, so need to subtract one for that.
        //Then add two since type_id one maps to three different possible function types.
        return static_cast<FunctionType>(type_id + 1);
    }
}

TROTSEntry::TROTSEntry(matvar_t* problem_struct_entry, matvar_t* matrix_struct,
                       const std::vector<std::variant<MKL_sparse_matrix<double>,
                                         std::vector<double>>
                                        >& mat_refs) :
    rhs{0}, weight{0}, c{0}
{
    assert(problem_struct_entry->class_type == MAT_C_STRUCT);
    //Try to ensure that the structure is a scalar (1x1) struct.
    assert(problem_struct_entry->rank == 2);
    assert(problem_struct_entry->dims[0] == 1 && problem_struct_entry->dims[1] == 1);

    matvar_t* name_var = Mat_VarGetStructFieldByName(problem_struct_entry, "Name", 0);
    check_null(name_var, "Cannot find name variable in problem entry.");
    assert(name_var->class_type == MAT_C_CHAR);
    this->roi_name = std::string(static_cast<char*>(name_var->data));

    std::cerr << "Reading dataID\n";

    //Matlab stores numeric values as doubles by default, which seems to be the type
    //used by TROTS as well even for integral values. Do some casting here to convert the data to
    //more intuitive types.
    matvar_t* id_var = Mat_VarGetStructFieldByName(problem_struct_entry, "dataID", 0);
    check_null(id_var, "Could not read id field from struct\n");
    this->id = cast_from_double<int>(id_var);

    this->matrix_ref = &mat_refs[this->id - 1];

    std::cerr << "Reading Minimise\n";

    matvar_t* minimise_var = Mat_VarGetStructFieldByName(problem_struct_entry, "Minimise", 0);
    check_null(minimise_var, "Could not read Minimise field from struct\n");
    this->minimise = cast_from_double<bool>(minimise_var);

    std::cerr << "Reading IsConstraint\n";

    //The IsConstraint field goes against the trend of using doubles, and is actually a MATLAB logical val,
    //which matio has as a MAT_C_UINT8
    matvar_t* is_cons_var = Mat_VarGetStructFieldByName(problem_struct_entry, "IsConstraint", 0);
    check_null(is_cons_var, "Could not read IsConstraint field from struct\n");
    this->is_cons = *static_cast<bool*>(is_cons_var->data);

    std::cerr << "Reading Objective\n";

    matvar_t* objective_var = Mat_VarGetStructFieldByName(problem_struct_entry, "Objective", 0);
    check_null(objective_var, "Could not read Objective field from struct\n");
    this->rhs = *static_cast<double*>(objective_var->data);

    std::cerr << "Reading FunctionType\n";

    matvar_t* function_type = Mat_VarGetStructFieldByName(problem_struct_entry, "Type", 0);
    check_null(function_type, "Could not read the \"Type\" field from the problem struct\n");
    int TROTS_type = cast_from_double<int>(function_type);
    std::cerr << "dataID: " << this->id << "\n";
    //An index of 1 means a "linear" function, which in reality can be one of three possibilities. Min, max and mean.
    //Determine which one it is.
    if (TROTS_type == 1)
        this->type = get_linear_function_type(this->id, this->minimise, this->roi_name, matrix_struct);
    else
        this->type = get_nonlinear_function_type(TROTS_type);

    std::cerr << "Function type: " << func_type_names[static_cast<int>(this->type)] << "\n";

    matvar_t* weight_var = Mat_VarGetStructFieldByName(problem_struct_entry, "Weight", 0);
    check_null(weight_var, "Could not read the \"Weight\" field from the problem struct.\n");
    this->weight = *static_cast<double*>(weight_var->data);
    //A lot of the objective functions require a temporary y vector to hold the dose. To avoid allocating that space for each
    //call to compute the objective value, we pre-allocate storage for it in this->y_vec.
    if (this->type != FunctionType::Mean) {
        auto num_rows = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref)).get_rows();
        this->y_vec.resize(num_rows);
    }

    if (this->type == FunctionType::Quadratic) {
        matvar_t* c_var = Mat_VarGetStructFieldByName(matrix_struct, "c", this->id - 1);
        assert(c_var->data_type == MAT_T_SINGLE);
        this->c = static_cast<double>(*static_cast<float*>(c_var->data));
    }
}

double TROTSEntry::calc_value(const double* x) const {
    switch (this->type) {
        case FunctionType::Quadratic:
            return this->calc_quadratic(x);
        case FunctionType::Max:
            return this->calc_max(x);
        case FunctionType::Min:
            return this->calc_min(x);
        case FunctionType::Mean:
            return this->calc_mean(x);
        default:
            throw "Not implemented yet!\n";
    }
}

double TROTSEntry::calc_quadratic(const double* x) const {
    const auto& matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    return matrix.quad_mul(x, &this->y_vec[0]) + this->c;
}

double TROTSEntry::calc_max(const double* x) const {
    const auto& matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    matrix.vec_mul(x, &this->y_vec[0]);

    const double max_elem = *std::max_element(this->y_vec.cbegin(), this->y_vec.cend());
    return max_elem;
}

double TROTSEntry::calc_min(const double* x) const {
    const auto& matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    matrix.vec_mul(x, &this->y_vec[0]);

    const double min_elem = *std::min_element(this->y_vec.cbegin(), this->y_vec.cend());
    return min_elem;
}

double TROTSEntry::calc_mean(const double* x) const {
    const auto& vector = std::get<std::vector<double>>(*(this->matrix_ref));
    return cblas_ddot(vector.size(), x, 1, &vector[0], 1);
}

double TROTSEntry::quadratic_penalty_min(const double* x) const {
    const auto& matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    matrix.vec_mul(x, &this->y_vec[0]);

    double sq_diff = 0.0;
    const size_t num_voxels = this->y_vec.size();
    for (int i = 0; i < num_voxels; ++i) {
        double clamped_diff = std::min(this->y_vec[i] - this->rhs, 0.0);
        double sq = clamped_diff * clamped_diff;
        sq_diff += sq;
    }

    return sq_diff / static_cast<double>(num_voxels);
}

double TROTSEntry::quadratic_penalty_max(const double* x) const {
    const auto& matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    matrix.vec_mul(x, &this->y_vec[0]);

    double sq_diff = 0.0;
    const size_t num_voxels = this->y_vec.size();
    for (int i = 0; num_voxels; ++i) {
        double clamped_diff = std::max(this->y_vec[i] - this->rhs, 0.0);
        double sq = clamped_diff * clamped_diff;
        sq_diff += sq;
    }

    return sq_diff / static_cast<double>(num_voxels);
}
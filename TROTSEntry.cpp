#include <cassert>
#include <vector>

#include <matio.h>

#include "SparseMat.h"
#include "TROTSEntry.h"
#include "util.h"

namespace {
    matrix_descr desc{.type = SPARSE_MATRIX_TYPE_GENERAL,
                      .mode = SPARSE_FILL_MODE_LOWER,
                      .diag = SPARSE_DIAG_NON_UNIT};

    FunctionType get_linear_function_type(int dataID, bool minimise, const std::string& roi_name, matvar_t* matrix_struct) {
        const int zero_indexed_dataID = dataID - 1;
        std::cerr << "Reading name field\n";
        //The times when the function type is mean, the data.matrix struct should have an entry at dataID with the name "<ROI_name> + (mean)"...
        matvar_t* matrix_entry_name_var = Mat_VarGetStructFieldByName(matrix_struct, "Name", zero_indexed_dataID);
        check_null(matrix_entry_name_var, "Failed to read name field of matrix entry\n.");

        std::string matrix_entry_name(static_cast<char*>(matrix_entry_name_var->data));
        
        if (matrix_entry_name == (roi_name + " (mean)"))
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
                                        >& mat_refs)
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

    matrix_ref = &mat_refs[this->id - 1];

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

    //In the case of quadratic cost function, we allocate space in dot_product_tmp to store the intermediate result when computing the quadratic
    //cost function using MKL.
    if (this->type == FunctionType::Quadratic) {
        auto num_rows = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref)).get_rows();
        this->dot_product_tmp.resize(num_rows);
    }
}

double TROTSEntry::calc_value(const double* x) const {
    switch (this->type) {
        case FunctionType::Quadratic:
            return this->calc_quadratic(x);
        default:
            throw "Not implemented yet!\n";
    }
}

double TROTSEntry::calc_quadratic(const double* x) const {
    double val = 0.0;
    const auto matrix = std::get<MKL_sparse_matrix<double>>(*(this->matrix_ref));
    const auto status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.mkl_handle, desc, x, 0.0, &this->dot_product_tmp[0], &val);
    assert(check_MKL_status(status));
    val += this->c;
    return val;
}

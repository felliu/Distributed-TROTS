#ifndef TROTS_H
#define TROTS_H

#include <filesystem>
#include <map>
#include <string>
#include <cassert>
#include <variant>


#include <hdf5/serial/hdf5.h>
#include <matio.h>
#include <mkl.h>

#include "OptimizationProblem.h"
#include "SparseMat.h"

enum class FunctionType {
    Min, Max, Mean, Quadratic,
    gEUD, LTCP, DVH, Chain
};

class TROTSEntry {
public:
    TROTSEntry(matvar_t* problem_struct_entry, matvar_t* data_struct);
private:
    int id;
    std::string roi_name;
    std::vector<double> func_params;

    bool minimise;
    bool is_cons;

    FunctionType type;
    double rhs;
    double weight;
    //If the FunctionType is mean, the value is computed using a dot product with a dense vector,
    //In other cases, the dose is calculated using a dose deposition matrix.
    std::variant<MKL_sparse_matrix<double>, std::vector<double>> matrix;
    double c; //Scalar factor used in quadratic cost functions.
};

class TROTSMatFileData {
public:
    TROTSMatFileData(const std::filesystem::path& file_path);
private:
    void init_problem_data_structs();
    void read_dose_matrices();

    mat_t* file_fp;
    matvar_t* problem_struct;
    matvar_t* data_struct;
    std::map<std::string, sparse_matrix_t> dose_matrices;
};

class TROTSProblem : public OptimizationProblem {
public:
    TROTSProblem(const TROTSMatFileData& trots_data);
private:

};

#endif
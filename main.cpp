#include "trots.h"

#include <filesystem>
#include <iostream>
#include <random>

std::vector<double> init_rand_vector(int size) {
    std::vector<double> x;
    x.reserve(size);
    for (int i = 0; i < size; ++i)
        x.push_back(std::rand() / static_cast<double>(RAND_MAX));

    return x;
}

void test_value_calc(const std::vector<TROTSEntry>& entries, const std::vector<double>& x) {
    int idx = 0;
    for (const auto& entry : entries) {
        std::cerr << "Checking entry number: " << idx << "\n";
        std::cerr << "Name: " << entry.get_roi_name() << "\n";
        if (entry.function_type() == FunctionType::Quadratic) {
            double val = entry.calc_value(x.data());
            std::cerr << "Type: Quadratic\tVal: " << val << "\n";
        }

        else if (entry.function_type() == FunctionType::Max) {
            std::cerr << "x sz: " << x.size() << "\n";
            double val = entry.calc_value(x.data());
            std::cerr << "Type: Max\tVal: " << val << "\n";
        }

        else if (entry.function_type() == FunctionType::Min) {
            double val = entry.calc_value(x.data());
            std::cerr << "Type: Min\tVal: " << val << "\n";
        }

        else if (entry.function_type() == FunctionType::Mean) {
            double val = entry.calc_value(x.data());
            std::cerr << "Type: Mean\tVal: " << val << "\n";
        }


        ++idx;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)  {
        std::cerr << "Incorrect number of arguments\n";
        std::cerr << "Usage: ./program <mat_file_path>\n";
        return -1;
    }

    std::string path_str{argv[1]};
    std::filesystem::path path{path_str};

    TROTSProblem trots_problem{TROTSMatFileData{path}};

    std::cerr << "Calculating quadratic objectives...\n";
    std::vector<double> x(trots_problem.get_num_vars(), 1.0); // = init_rand_vector(trots_problem.get_num_vars());
    test_value_calc(trots_problem.objective_entries, x);
    test_value_calc(trots_problem.constraint_entries, x);
    return 0;
    /*
    std::cerr << "Reading matfile...";
    mat_t* mat_file_handle = Mat_Open(argv[1], MAT_ACC_RDONLY);
    if (mat_file_handle == NULL) {
        std::cerr << "Mat file could not be opened, check file name spelling\n.";
        return -1;
    }
    std::cerr << "Done!\n";

    std::cerr << "Reading problem struct...";
    matvar_t* problem_struct = Mat_VarRead(mat_file_handle, "problem");
    if (problem_struct == NULL) {
        std::cerr << "Problem struct could not be read from matfile\n.";
        return -1;
    }
    std::cerr << "Done!\n";

    std::cerr << "Reading data struct...";
    matvar_t* data_struct = Mat_VarRead(mat_file_handle, "data");
    if (data_struct == NULL) {
        std::cerr << "Data struct could not be read from matfile\n.";
        return -1;
    }
    std::cerr << "Done!\n";

    std::cerr << "Reading matrix struct from data...\n";
    matvar_t* matrix_struct = Mat_VarGetStructFieldByName(data_struct, "matrix", 0);
    if (matrix_struct == NULL) {
        std::cerr << "Matrix struct could not be read from matfile\n";
        return -1;
    }
    */

    return 0;
}

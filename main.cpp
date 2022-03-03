#include "trots.h"
#include "util.h"

#include "trots_ipopt.h"

#include <filesystem>
#include <iostream>
#include <fstream>
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
        std::cout << "Checking entry number: " << idx << "\n";
        std::cout << "Name: " << entry.get_roi_name() << "\n";
        double val = entry.calc_value(x.data());
        switch (entry.function_type()) {
            case FunctionType::Quadratic:
                std::cout << "Type: Quadratic\tVal: " << val << "\n";
                break;
            case FunctionType::Max:
                std::cout << "Type: Max\tVal: " << val << "\n";
                break;
            case FunctionType::Min:
                std::cout << "Type: Min\tVal: " << val << "\n";
                break;
            case FunctionType::Mean:
                std::cout << "Type: Mean\tVal: " << val << "\n";
                break;
            case FunctionType::gEUD:
                std::cout << "Type: gEUD\tVal: " << val << "\n";
                break;
            case FunctionType::LTCP:
                std::cout << "Type: LTCP\tVal: " << val << "\n";
                break;
        }

        ++idx;
    }
}

void test_gradient_calc(const std::vector<TROTSEntry>& entries, const std::vector<double>& x) {
    int idx = 0;
    std::vector<double> grad(x.size());
    for (const auto& entry : entries) {
        std::cout << "Computing gradient for entry number: " << idx << "\n";
        std::cout << "Name: " << entry.get_roi_name() << "\n";
        entry.calc_gradient(&x[0], &grad[0]);
        idx++;
    }

}

void calc_jacobian_sparsity(const TROTSProblem& problem) {
    std::vector<int> col_nonzero_hist(problem.get_num_vars(), 0);
    int row = 0;
    for (const auto& cons_entry : problem.constraint_entries) {
        std::vector<int> col_idxs = cons_entry.get_grad_nonzero_idxs();
        for (int idx : col_idxs) {
            std::cerr << "Row: " << row << ", col: " << idx << "\n";
            col_nonzero_hist[idx]++;
        }
        row++;
    }
    dump_vector_to_file(col_nonzero_hist, "col_idx_hist_hn_01.bin");
}

int main(int argc, char* argv[])
{
    ipopt_main_func(argc, argv);
    return 0;
    if (argc != 2)  {
        std::cerr << "Incorrect number of arguments\n";
        std::cerr << "Usage: ./program <mat_file_path>\n";
        return -1;
    }

    std::string path_str{argv[1]};
    std::filesystem::path path{path_str};

    TROTSProblem trots_problem{TROTSMatFileData{path}};
    //calc_jacobian_sparsity(trots_problem);
    return 0;

    std::vector<double> x = init_rand_vector(trots_problem.get_num_vars());
    test_value_calc(trots_problem.objective_entries, x);
    test_value_calc(trots_problem.constraint_entries, x);
    test_gradient_calc(trots_problem.objective_entries, x);
    test_gradient_calc(trots_problem.constraint_entries, x);
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

#include "trots.h"

#include <chrono>
#include <iostream>

std::vector<double> init_x(const TROTSProblem& problem) {
    std::vector<double> x(problem.get_num_vars());

    std::fill(x.begin(), x.end(), 100.0);
    return x;
}

void time_obj_func(const TROTSProblem& problem, const std::vector<double>& x) {
    using namespace std::chrono;

    std::vector<double> tmp(problem.get_num_constraints());
    auto start_obj = high_resolution_clock::now();
    problem.calc_objective(&x[0]);
    auto end_obj = high_resolution_clock::now();
    problem.calc_constraints(&x[0], &tmp[0]);
    auto end_cons = high_resolution_clock::now();

    std::cout << "Objective time: " << duration_cast<milliseconds>(end_obj - start_obj).count() << " ms\n";
    std::cout << "Constraints time: " << duration_cast<milliseconds>(end_cons - end_obj).count() << " ms\n";
    std::cout << "Total time: " << duration_cast<milliseconds>(end_cons - start_obj).count() << " ms\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./<program> <TROTS_mat_file>\n";
        return -1;
    }

    std::string path_str{argv[1]};
    std::filesystem::path path{path_str};

    TROTSProblem trots_problem{TROTSMatFileData{path}};
    std::vector<double> x = init_x(trots_problem);
    time_obj_func(trots_problem, x);
    return 0;
}
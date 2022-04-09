#include "trots.h"

#include <chrono>
#include <iostream>

constexpr int N = 10;

std::vector<double> init_x(const TROTSProblem& problem) {
    std::vector<double> x(problem.get_num_vars());

    std::fill(x.begin(), x.end(), 100.0);
    return x;
}

void time_obj_func(const TROTSProblem& problem, const std::vector<double>& x) {
    using namespace std::chrono;

    std::vector<double> tmp(problem.get_num_constraints());
    std::vector<microseconds> obj_durations;
    std::vector<microseconds> cons_durations;


    for (int i = 0; i < N; ++i) {
        auto start_obj = high_resolution_clock::now();
        problem.calc_objective(&x[0]);
        auto end_obj = high_resolution_clock::now();
        problem.calc_constraints(&x[0], &tmp[0]);
        auto end_cons = high_resolution_clock::now();
        obj_durations.push_back(duration_cast<microseconds>(end_obj - start_obj));
        cons_durations.push_back(duration_cast<microseconds>(end_cons - end_obj));
    }

    microseconds total_time_obj(0);
    microseconds total_time_cons(0);
    for (int i = 0; i < N; ++i) {
        total_time_obj += obj_durations[i];
        total_time_cons += cons_durations[i];
    }
    double avg_time_obj_ms = duration_cast<milliseconds>(total_time_obj).count() / static_cast<double>(N);
    double avg_time_cons_ms = duration_cast<milliseconds>(total_time_cons).count() / static_cast<double>(N);
    double avg_time_total_ms = avg_time_obj_ms + avg_time_cons_ms;

    std::cout << "Obj time avg: " << avg_time_obj_ms << " ms\n";
    std::cout << "cons time avg: " << avg_time_cons_ms << " ms\n";
    std::cout << "total time avg: " << avg_time_total_ms << " ms\n";
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

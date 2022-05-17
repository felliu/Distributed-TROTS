#include "trots.h"

#include <chrono>
#include <filesystem>
#include <fstream>
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

void dump_nnz_counts(const TROTSProblem& trots_problem,
                     const std::filesystem::path& cons_nnz_path,
                     const std::filesystem::path& obj_nnz_path) {
    std::vector<int> nnz_cons;
    std::vector<int> nnz_obj;
    for (const TROTSEntry& cons_entry : trots_problem.constraint_entries) {
        nnz_cons.push_back(cons_entry.get_nnz());
    }
    for (const TROTSEntry& obj_entry : trots_problem.constraint_entries) {
        nnz_obj.push_back(obj_entry.get_nnz());
    }

    std::ofstream cons_stream{cons_nnz_path};
    for (int i  = 0; i < nnz_cons.size() - 1; ++i) {
        cons_stream << nnz_cons[i] << ", ";
    }
    cons_stream << nnz_cons[nnz_cons.size() - 1] << "\n";

    std::ofstream obj_stream{obj_nnz_path};
    for (int i  = 0; i < nnz_obj.size() - 1; ++i) {
        obj_stream << nnz_obj[i] << ", ";
    }
    obj_stream << nnz_obj[nnz_obj.size() - 1] << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: ./<program> <TROTS_mat_file>\n";
        return -1;
    }

    std::string path_str{argv[1]};
    std::filesystem::path path{path_str};

    TROTSProblem trots_problem{TROTSMatFileData{path}};
    dump_nnz_counts(trots_problem, "cons_nnz.csv", "obj_nnz.csv");
    std::vector<double> x = init_x(trots_problem);
    time_obj_func(trots_problem, x);
    return 0;
}

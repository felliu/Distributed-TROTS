#include <mpi.h>

#include <iostream>
#include <filesystem>
#include "trots.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (rank == 0) {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: ./program <mat_file>\n"
                      << "\t./program <mat_file> <max_iters>\n";
        }

        std::string path_str{argv[1]};
        std::filesystem::path path{path_str};

        TROTSProblem trots_problem{TROTSMatFileData{path}};
    }
    MPI_Finalize();
    return 0;
}
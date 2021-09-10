#include "trots.h"

#include <filesystem>
#include <iostream>

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
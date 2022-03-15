#include "trots_matfile_data.h"

#include <cassert>

#include "util.h"

namespace fs = std::filesystem;

TROTSMatFileData::TROTSMatFileData(const fs::path& path) {
    this->file_fp = Mat_Open(path.c_str(), MAT_ACC_RDONLY);
    assert(this->file_fp != nullptr);
    this->init_problem_data_structs();
}

TROTSMatFileData::TROTSMatFileData(TROTSMatFileData&& other) {
    this->file_fp = other.file_fp;
    this->data_struct = other.data_struct;
    this->problem_struct = other.problem_struct;
    this->matrix_struct = other.matrix_struct;

    other.file_fp = NULL;
    other.data_struct = NULL;
    other.problem_struct = NULL;
    other.matrix_struct = NULL;
}

TROTSMatFileData::~TROTSMatFileData() {
    Mat_VarFree(this->data_struct);
    Mat_VarFree(this->problem_struct);
    Mat_Close(this->file_fp);
}

void TROTSMatFileData::init_problem_data_structs() {
    this->problem_struct = Mat_VarRead(this->file_fp, "problem");
    check_null(this->problem_struct, "Unable to read problem struct from matfile\n.");

    this->data_struct = Mat_VarRead(this->file_fp, "data");
    check_null(this->data_struct, "Unable to read data variable from matfile\n");

    this->matrix_struct = Mat_VarGetStructFieldByName(this->data_struct, "matrix", 0);
    check_null(this->matrix_struct, "Unable to read matrix field from matfile\n");
}
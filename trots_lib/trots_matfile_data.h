#ifndef TROTS_MATFILE_DATA_H
#define TROTS_MATFILE_DATA_H

#include <matio.h>
#include <filesystem>

struct TROTSMatFileData {
public:
    TROTSMatFileData(const std::filesystem::path& file_path);

    TROTSMatFileData(const TROTSMatFileData& other) = delete;
    TROTSMatFileData& operator=(const TROTSMatFileData& rhs) = delete;

    TROTSMatFileData(TROTSMatFileData&& data);
    ~TROTSMatFileData();

    mat_t* file_fp;
    matvar_t* problem_struct;
    matvar_t* data_struct;
    matvar_t* matrix_struct;
private:
    void init_problem_data_structs();
};

#endif
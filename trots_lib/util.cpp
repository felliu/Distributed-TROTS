#include "util.h"

#include <cassert>
#include <cstring>

std::string get_name_str(const matvar_t* name_var) {
    assert(name_var->rank == 2 && name_var->dims[0] == 1);
    const size_t name_len = name_var->dims[1];
    char null_terminated_buffer[name_len + 1];
    std::memcpy(static_cast<void*>(&null_terminated_buffer[0]), name_var->data, name_len * sizeof(char));
    null_terminated_buffer[name_len] = '\0';
    return std::string(null_terminated_buffer);
}
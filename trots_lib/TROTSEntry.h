#ifndef TROTS_ENTRY_H
#define TROTS_ENTRY_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <variant>

#include "SparseMat.h"

enum class FunctionType {
    Min, Max, Mean, Quadratic,
    gEUD, LTCP, DVH, Chain
};

struct matvar_t;

class TROTSEntry {
public:
    //The default constructor is provided to enable serialization & de-serialization to
    //transfer between MPI ranks
    TROTSEntry() = default;
    TROTSEntry(matvar_t* problem_struct_entry, matvar_t* data_struct,
               const std::vector<std::variant<std::unique_ptr<SparseMatrix<double>>,
                                              std::vector<double>>
                                >& mat_refs);

    bool is_constraint() const noexcept { return this->is_cons; }
    bool is_active() const noexcept { return this->active; }
    bool is_minimisation() const noexcept { return this->minimise; }
    double calc_value(const double* x, bool cached_dose=false) const;
    double get_weight() const noexcept { return this->weight; }
    double get_rhs() const noexcept { return this->rhs; }
    int get_id() const noexcept { return this->id; }
    void calc_gradient(const double* x, double* grad, bool cached_dose=false) const;
    int get_nnz() const {
        assert (this->matrix_ref != nullptr || this->mean_vec_ref != nullptr);
        if (this->matrix_ref != nullptr) {
            return this->matrix_ref->get_nnz();
        } else {
            return static_cast<int>(this->mean_vec_ref->size());
        }
    }

    void set_matrix_ptr(SparseMatrix<double>* ptr) { this->matrix_ref = ptr; }
    void set_mean_vec_ptr(std::vector<double>* ptr) { this->mean_vec_ref = ptr; }

    std::vector<double> calc_sparse_grad(const double* x, bool cached_dose=false) const;
    FunctionType function_type() const noexcept { return this->type; }
    std::string get_roi_name() const { return this->roi_name; }

    //Returns the indexes of the non-zero elements in the gradient of the entry.
    std::vector<int> get_grad_nonzero_idxs() const { return this->grad_nonzero_idxs; }
    int get_grad_nnz() const { return this->grad_nonzero_idxs.size(); }
private:
    double calc_quadratic(const double* x) const;
    double calc_max(const double* x) const;
    double calc_min(const double* x) const;
    double calc_mean(const double* x) const;
    double calc_LTCP(const double* x, bool cached_dose=false) const;
    double calc_gEUD(const double* x, bool cached_dose=false) const;
    double quadratic_penalty_min(const double* x, bool cached_dose=false) const;
    double quadratic_penalty_max(const double* x, bool cached_dose=false) const;
    double quadratic_penalty_mean(const double* x) const;

    void mean_grad(const double* x, double* grad) const;
    void quad_mean_grad(const double* x, double* grad) const;
    void LTCP_grad(const double* x, double* grad, bool cached_dose) const;
    void gEUD_grad(const double* x, double* grad, bool cached_dose) const;
    void quad_min_grad(const double* x, double* grad, bool cached_dose) const;
    void quad_max_grad(const double* x, double* grad, bool cached_dose) const;
    void quad_grad(const double* x, double* grad) const;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version);

    std::vector<int> calc_grad_nonzero_idxs() const;

    int num_vars;
    int id;
    std::string roi_name;
    std::vector<double> func_params;

    std::vector<int> grad_nonzero_idxs;

    bool active;
    bool minimise;
    bool is_cons;

    FunctionType type;
    double rhs;
    double weight;

    double c; //Scalar factor used in quadratic cost functions.
    //const MKL_sparse_matrix<double>* matrix_ref;
    SparseMatrix<double> const* matrix_ref;
    std::vector<double> const* mean_vec_ref;

    //When calculating many objective values, a temporary store for the A*x is needed. Provide it here once so it does not
    //need to be allocated every time.
    mutable std::vector<double> y_vec;
    //Gradient calculation can require more temporaries
    mutable std::vector<double> grad_tmp;
};

template <typename Archive>
void TROTSEntry::serialize(Archive& ar, const unsigned int version) {
    //this->matrix_ref = nullptr;
    //this->mean_vec_ref = nullptr;
    ar & num_vars;
    ar & id;
    ar & roi_name;
    ar & func_params;
    ar & grad_nonzero_idxs;
    ar & active;
    ar & minimise;
    ar & is_cons;
    ar & type;
    ar & rhs;
    ar & weight;
    ar & c;
    //These will lose their meaning when sent, but we will reset them on the other side
    //ar & matrix_ref;
    //ar & mean_vec_ref;

    ar & y_vec;
    ar & grad_tmp;
}


#endif

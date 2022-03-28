#ifndef TROTS_IPOPT_MPI_H
#define TROTS_IPOPT_MPI_H

#include <memory>
#include <optional>

#include "coin-or/IpTNLP.hpp"

#include "rank_local_data.h"
#include "trots.h"

//Structure with data on the distribution of the constraints over the different ranks
//This is used to be able to gather the values on rank 0 after they are computed
struct ConsDistributionData {
    std::vector<int> recv_counts_g;
    std::vector<int> recv_displacements_g;
    std::vector<int> recv_counts_jac;
    std::vector<int> recv_displacements_jac;
};

class TROTS_ipopt_mpi : public Ipopt::TNLP {
public:
    TROTS_ipopt_mpi(TROTSProblem&& problem,
                    const std::vector<std::vector<int>>& cons_term_distribution,
                    LocalData&& data);
    bool get_nlp_info(
        int& n, int& m, int& nnz_jac_g, int& nnz_h_lag,
        Ipopt::TNLP::IndexStyleEnum& index_style) override;

    bool get_bounds_info(int n, double* x_l, double* x_u, int m, double* g_l, double* g_u) override;
    bool get_starting_point(int n, bool init_x, double* x, bool init_z, double* z_l, double* z_u,
                            int m, bool init_lambda, double* lambda) override;

    bool eval_f(int n, const double* x, bool new_x, double& obj_val) override;
    bool eval_grad_f(int n, const double* x, bool new_x, double* grad_f) override;
    bool eval_g(int n, const double* x, bool new_x, int m, double* g) override;
    bool eval_jac_g(int n, const double* x, bool new_x,
                    int m, int nnz_jac, int* irow, int* icol, double* vals) override;
    void finalize_solution(Ipopt::SolverReturn status, int n,
                           const double* x, const double* z_l, const double* z_u,
                           int m, const double* g, const double* lambda, double obj,
                           const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq) override;
private:
    //The bulk of the data associated with the problem will be distributed across MPI ranks
    //in this version. We keep this reference to a TROTSProblem instance here for
    //some of the basic info about the problem, such as number of variables and constraints.
    std::unique_ptr<TROTSProblem> trots_problem;
    //A vector with information about how the constraint terms are distributed over ranks
    std::vector<std::vector<int>> cons_term_distribution;
    LocalData local_data;
    ConsDistributionData distrib_data;
};

double compute_obj_vals_mpi(const double* x, bool calc_grad, double* grad, LocalData& local_data);
void compute_cons_vals_mpi(const double* x, double* cons_vals,
                           bool calc_grad, double* grad, LocalData& local_data,
                           std::optional<ConsDistributionData> distrib_data);

#endif
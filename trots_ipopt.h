#ifndef TROTS_IPOPT_H
#define TROTS_IPOPT_H

#include <memory>

#include "coin-or/IpTNLP.hpp"

#include "trots.h"

class TROTS_ipopt : public Ipopt::TNLP {
public:
    TROTS_ipopt(TROTSProblem&& prob);

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
    std::unique_ptr<TROTSProblem> problem;
};

int ipopt_main_func(int argc, char* argv[]);

#endif
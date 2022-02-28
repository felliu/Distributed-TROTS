#include "trots_ipopt.h"

#include "coin-or/IpIpoptApplication.hpp"

TROTS_ipopt::TROTS_ipopt(TROTSProblem&& problem) {
    this->problem = std::make_unique<TROTSProblem>(std::move(problem));
}

bool TROTS_ipopt::get_nlp_info(
    int& n, int& m, int& nnz_jac_g, int& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style) {

    n = this->problem->get_num_vars();
    m = this->problem->get_num_constraints();

    nnz_jac_g = this->problem->get_nnz_jac_cons();
    nnz_h_lag = n * n / 2; //Dense but symmetric
    index_style = C_STYLE;

    return true;
}


bool TROTS_ipopt::get_bounds_info(int n, double* x_l, double* x_u, int m, double* g_l, double* g_u) {
    //By default, IPOPT considers values greater than 1e19 in magnitude as infinite.
    constexpr double negative_inf = -1e20;
    constexpr double pos_inf = 1e20;
    for (int i = 0; i < n; ++i) {
        x_l[i] = negative_inf;
        x_u[i] = pos_inf;
    }

    for (int i = 0; i < m; ++i) {
        g_l[i] = 0.0;
        g_u[i] = pos_inf;
    }

    return true;
}


bool TROTS_ipopt::get_starting_point(
    int n, bool init_x, double* x, bool init_z, double* z_l, double* z_u,
    int m, bool init_lambda, double* lambda) {

    if (init_x) {
        //TODO: smarter initialization
        for (int i = 0; i < n; ++i) {
            x[i] = 1.0;
        }
    }

    //We have no bounds, so these probably don't matter, set them to zero in case.
    if (init_z) {
        for (int i = 0; i < n; ++i) {
            z_l[i] = 0.0;
            z_u[i] = 0.0;
        }
    }

    if (init_lambda) {
        //TODO: smarter initalization
        for (int i = 0; i < m; ++i) {
            lambda[i] = 1.0;
        }
    }

    return true;
}

bool TROTS_ipopt::eval_f(int n, const double* x, bool new_x, double& obj_val) {
    obj_val = this->problem->calc_objective(x);
    return true;
}

bool TROTS_ipopt::eval_grad_f(int n, const double* x, bool new_x, double* grad_f) {
    this->problem->calc_obj_gradient(x, grad_f);
    return true;
}

bool TROTS_ipopt::eval_g(int n, const double* x, bool new_x, int m, double* g) {
    this->problem->calc_constraints(x, g);
    return true;
}


bool TROTS_ipopt::eval_jac_g(int n, const double* x, bool new_x,
                             int m, int nnz_jac, int* irow, int* icol, double* vals) {

    if (vals == nullptr) {
        assert(irow != nullptr && icol != nullptr);
        int i = 0;
        for (int row = 0; row < this->problem->get_num_constraints(); ++row) {
            std::vector<int> cols = this->problem->constraint_entries[row].get_grad_nonzero_idxs();
            for (int col : cols) {
                irow[i] = row;
                icol[i] = col;
                ++i;
            }
        }
        return true;
    }

    this->problem->calc_jacobian_vals(x, vals);
    return true;
}


void TROTS_ipopt::finalize_solution(Ipopt::SolverReturn status, int n,
    const double* x, const double* z_l, const double* z_u,
    int m, const double* g, const double* lambda, double obj,
    const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq) {

    std::cout << "IPOPT finalize_solution called\n";
    std::cout << "Exit status: " << status << "\n";
}

int ipopt_main_func(int argc, char* argv[]) {
    if (argc != 2)  {
        std::cerr << "Incorrect number of arguments\n";
        std::cerr << "Usage: ./program <mat_file_path>\n";
        return -1;
    }

    std::string path_str{argv[1]};
    std::filesystem::path path{path_str};

    TROTSProblem trots_problem{TROTSMatFileData{path}};
    Ipopt::SmartPtr<Ipopt::TNLP> trots_nlp = new TROTS_ipopt(std::move(trots_problem));
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    //app->Options()->SetStringValue("derivative_test", "first-order");

    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    status = app->OptimizeTNLP(trots_nlp);

    return status;
}
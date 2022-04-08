#include "trots_ipopt_mpi.h"

#include "globals.h"
#include "util.h"

#include <iostream>
#include <mpi.h>

TROTS_ipopt_mpi::TROTS_ipopt_mpi(
        TROTSProblem&& problem,
        const std::vector<std::vector<int>>& cons_term_distribution,
        LocalData&& data) {
    this->trots_problem = std::make_unique<TROTSProblem>(std::move(problem));
    this->cons_term_distribution = cons_term_distribution;
    this->local_data = std::move(data);

    int cumulative_sum_g = 0;
    int cumulative_sum_jac_g = 0;

    for (int i = 0; i < this->cons_term_distribution.size(); ++i) {
        const std::vector<int>& cons_idxs = this->cons_term_distribution[i];
        int nnz_total = 0;
        int num_cons = cons_idxs.size();
        for (int idx : cons_idxs) {
            nnz_total += this->trots_problem->constraint_entries[idx].get_grad_nnz();
        }
        this->distrib_data.recv_counts_g.push_back(num_cons);
        this->distrib_data.recv_displacements_g.push_back(cumulative_sum_g);
        this->distrib_data.recv_counts_jac.push_back(nnz_total);
        this->distrib_data.recv_displacements_jac.push_back(cumulative_sum_jac_g);

        cumulative_sum_g += num_cons;
        cumulative_sum_jac_g += nnz_total;
    }

    print_vector(this->distrib_data.recv_counts_g);
    print_vector(this->distrib_data.recv_displacements_g);
    print_vector(this->distrib_data.recv_counts_jac);
    print_vector(this->distrib_data.recv_displacements_jac);
}

bool TROTS_ipopt_mpi::get_nlp_info(
    int& n, int& m, int& nnz_jac_g, int& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style) {

    n = this->trots_problem->get_num_vars();
    m = this->trots_problem->get_num_constraints();
    std::cout << "n: " << n << std::endl;
    std::cout << "m: " << m << std::endl;
    nnz_jac_g = this->trots_problem->get_nnz_jac_cons();
    std::cout << "nnz_jac_g: " << nnz_jac_g << std::endl;
    //Hessian is dense, but symmetric
    nnz_h_lag = n * n / 2;
    index_style = C_STYLE;
    return true;
}

bool TROTS_ipopt_mpi::get_bounds_info(int n, double* x_l, double* x_u, int m, double* g_l, double* g_u) {
    //By default, IPOPT considers values greater than 1e19 in magnitude as infinite.
    constexpr double neg_inf = -1e20;
    constexpr double pos_inf = 1e20;
    for (int i = 0; i < n; ++i) {
        x_l[i] = 0.0;
        x_u[i] = pos_inf;
    }

    for (int i = 0; i < m; ++i) {
        //Right now this only works for problems with only minimisation constraints.
        //TODO: maybe add support for other types of constraints
        g_l[i] = neg_inf;
        g_u[i] = 0.0;
    }

    return true;
}

bool TROTS_ipopt_mpi::get_starting_point(
    int n, bool init_x, double* x, bool init_z, double* z_l,
    double* z_u, int m, bool init_lambda, double* lambda) {
    if (init_x) {
        std::vector<TROTSEntry> LTCP_entries;
        std::copy_if(this->trots_problem->objective_entries.cbegin(),
                     this->trots_problem->objective_entries.cend(),
                     std::back_inserter(LTCP_entries),
                     [](const TROTSEntry& e){ return e.function_type() == FunctionType::LTCP; });

        for (int i = 0; i < n; ++i) {
            x[i] = 100.0;
        }

        const auto compute_obj_val = [x](const TROTSEntry& e) -> double {
            return e.calc_value(x);
        };
        std::vector<double> LTCP_vals(LTCP_entries.size());
        std::transform(LTCP_entries.cbegin(), LTCP_entries.cend(),
                       LTCP_vals.begin(), compute_obj_val);
        while (std::any_of(LTCP_vals.cbegin(), LTCP_vals.cend(), [](double v){ return v > 1500.0; })) {
            for (int i = 0; i < n; ++i) {
                x[i] *= 1.5;
            }
            std::transform(LTCP_entries.cbegin(), LTCP_entries.cend(),
                           LTCP_vals.begin(), compute_obj_val);
        }
    }

    if (init_z) {
        for (int i = 0; i < n; ++i) {
            z_l[i] = 0.0;
            z_u[i] = 0.0;
        }
    }

    if (init_lambda) {
        for (int i = 0; i < m; ++i) {
            lambda[i] = 1.0;
        }
    }

    //this->trots_problem->clear_mat_data();
    return true;
}

bool TROTS_ipopt_mpi::eval_f(int n, const double* x, bool new_x, double& obj_val) {
    //std::cout << "Calculating f\n";
    obj_val = compute_vals_mpi(true, x, nullptr, false, nullptr,
                               this->local_data, std::nullopt, false);
    //std::cout << "Done" << std::endl;
    return true;
}

bool TROTS_ipopt_mpi::eval_grad_f(int n, const double* x, bool new_x, double* grad_f) {
    //std::cout << "Calculating grad f\n";
    compute_vals_mpi(true, x, nullptr, true, grad_f, this->local_data, std::nullopt, false);
    //std::cout << "Done" << std::endl;
    return true;
}

bool TROTS_ipopt_mpi::eval_g(int n, const double* x, bool new_x, int m, double* g) {
    //std::cout << "Calculating g\n";
    compute_vals_mpi(false, x, g, false, nullptr, this->local_data,
                     std::make_optional(this->distrib_data), false);
    //std::cout << "Done" << std::endl;
    return true;
}

bool TROTS_ipopt_mpi::eval_jac_g(
    int n, const double* x, bool new_x,
    int m, int nnz_jac, int* irow, int* icol, double* vals) {
    if (vals == nullptr) {
        assert(irow != nullptr && icol != nullptr);
        //std::cout << "Setting jacobian indexes\n";
        //Flatten the array with constraint indices over ranks.
        std::vector<int> flattened_cons_idxs;
        for (const std::vector<int>& rank_idxs : this->cons_term_distribution) {
            flattened_cons_idxs.insert(flattened_cons_idxs.end(),
                                       rank_idxs.cbegin(), rank_idxs.cend());
        }
        int idx = 0;
        int i = 0;
        for (int row = 0; row < flattened_cons_idxs.size(); ++row) {
            const int cons_idx = flattened_cons_idxs[row];
            std::vector<int> col_idxs =
                this->trots_problem->constraint_entries[cons_idx].get_grad_nonzero_idxs();
            for (int col : col_idxs) {
                irow[idx] = row;
                icol[idx] = col;
                ++idx;
            }
        }
        //std::cout << "Done" << std::endl;
    }

    else {
        //std::cout << "Calculating jac g\n";
        compute_vals_mpi(false, x, nullptr, true, vals,
                         this->local_data, std::make_optional(this->distrib_data), false);
        //std::cout << "Done" << std::endl;
    }

    return true;
}

void TROTS_ipopt_mpi::finalize_solution(
    Ipopt::SolverReturn status, int n,
    const double* x, const double* z_l, const double* z_u,
    int m, const double* g, const double* lambda, double obj,
    const Ipopt::IpoptData* ip_data, Ipopt::IpoptCalculatedQuantities* ip_cq) {

    //Output to file: reuse code for dumping std::vector arrays to file
    std::vector<double> x_vec;
    x_vec.reserve(n);
    for (int i = 0; i < n; ++i) {
        x_vec.push_back(x[i]);
    }

    dump_vector_to_file(x_vec, "out_mpi.bin");

    std::cout << "IPOPT finalize_solution called\n";
    std::cout << "Exit status: " << status << "\n";
}

double compute_vals_mpi(bool calc_obj, const double* x, double* cons_vals, bool calc_grad, double* grad,
                        LocalData& local_data, std::optional<ConsDistributionData> distrib_data, bool done) {
    while (true) {
        MPI_Barrier(MPI_COMM_WORLD);
        int done_flag = static_cast<int>(done);
        MPI_Bcast(&done_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (done_flag)
            return 0.0;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        double obj_val = 0;
        int calc_obj_flag = static_cast<int>(calc_obj);
        MPI_Bcast(&calc_obj, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (calc_obj) {
            obj_val = compute_obj_vals_mpi(x, calc_grad, grad, local_data);
        } else {
            compute_cons_vals_mpi(x, cons_vals, calc_grad, grad, local_data, distrib_data);
        }

        if (rank == 0)
            return obj_val;
    }
}

double compute_obj_vals_mpi(const double* x, bool calc_grad, double* grad, LocalData& local_data) {
    MPI_Barrier(MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::copy(x, x + local_data.num_vars, local_data.x_buffer.begin());
        assert(local_data.obj_entries.empty());
    }
    int grad_flag_local = static_cast<int>(calc_grad);
    MPI_Bcast(&grad_flag_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_data.x_buffer[0], local_data.num_vars, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double obj_val_local = 0.0;
    double obj_val = 0.0;
    if (!grad_flag_local) {
        for (const TROTSEntry& entry : local_data.obj_entries) {
            obj_val_local += entry.calc_value(&local_data.x_buffer[0]) * entry.get_weight();
        }
        MPI_Reduce(&obj_val_local, &obj_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        double* grad_buf = new double[local_data.num_vars];
        std::fill(grad_buf, grad_buf + local_data.num_vars, 0.0);
        for (const TROTSEntry& entry : local_data.obj_entries) {
            entry.calc_gradient(&local_data.x_buffer[0], &local_data.grad_tmp[0]);
            for (int i = 0; i < local_data.num_vars; ++i) {
                grad_buf[i] += local_data.grad_tmp[i] * entry.get_weight();
            }
        }
        MPI_Reduce(grad_buf, grad, local_data.num_vars, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        delete[] grad_buf;
    }
    return obj_val;
}


void compute_cons_vals_mpi(const double* x, double* cons_vals,
                           bool calc_grad, double* grad, LocalData& local_data,
                           std::optional<ConsDistributionData> distrib_data) {
    MPI_Barrier(MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        assert(distrib_data.has_value());
        std::copy(x, x + local_data.num_vars, local_data.x_buffer.begin());
    }

    int grad_flag_local = static_cast<int>(calc_grad);
    MPI_Bcast(&grad_flag_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_data.x_buffer[0], local_data.num_vars, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (!grad_flag_local) {
        std::vector<double> local_vals(local_data.cons_entries.size());
        int i = 0;
        for (const TROTSEntry& entry : local_data.cons_entries) {
            local_vals[i] = entry.calc_value(&local_data.x_buffer[0]);
            ++i;
        }
        if (rank == 0) {
            ConsDistributionData& dist_data = distrib_data.value();
            MPI_Gatherv(&local_vals[0], 0, MPI_DOUBLE,
                        cons_vals, &dist_data.recv_counts_g[0], &dist_data.recv_displacements_g[0],
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(&local_vals[0], local_vals.size(), MPI_DOUBLE,
                        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

    } else {
        double* local_buf = new double[local_data.local_jac_nnz];
        int start_idx = 0;
        for (const TROTSEntry& entry : local_data.cons_entries) {
            std::vector<double> vals = entry.calc_sparse_grad(&local_data.x_buffer[0]);
            std::copy(vals.cbegin(), vals.cend(), local_buf + start_idx);
            start_idx += vals.size();
        }
        if (rank == 0) {
            ConsDistributionData& dist_data = distrib_data.value();
            MPI_Gatherv(local_buf, 0, MPI_DOUBLE, grad,
                        &dist_data.recv_counts_jac[0], &dist_data.recv_displacements_jac[0],
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(local_buf, local_data.local_jac_nnz, MPI_DOUBLE, nullptr,
                        nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        delete[] local_buf;
    }
}


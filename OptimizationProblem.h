#ifndef OPTIMIZATION_PROBLEM_H
#define OPTIMIZATION_PROBLEM_H

#include <vector>

class OptimizationProblem {
public:
    virtual double eval_objective(int num_vars, const double* x) = 0;
    virtual void eval_obj_grad(int num_vars, const double* x, double* grad) = 0;
    virtual void get_lower_x_bounds(int num_vars, const double* bounds) = 0;
    virtual void get_upper_x_bounds(int num_vars, const double* bounds) = 0;
    virtual void eval_cons(int num_vars, int num_cons, const double* x, double* cons_vals) = 0;
    virtual void eval_ineq_cons(int num_vars, int num_ineq_cons, const double* x, double* ineq_cons_vals) = 0;
    virtual void eval_eq_cons(int num_vars, int num_eq_cons, const double* x, double* eq_cons_vals) = 0;

    virtual void eval_jacobian(int num_vars, int num_cons, const double* x, int nnz, int* row_inds, int* col_inds, double* vals) = 0;
    virtual void eval_ineq_jacobian(int num_vars, int num_ineq_cons, const double* x, int nnz, int* row_inds, int* col_inds, double* vals) = 0;
    virtual void eval_eq_jacobian(int num_vars, int num_eq_cons, const double* x, int nnz, int* row_inds, int* col_inds, double* vals) = 0;
    virtual void eval_hessian(int num_vars, int num_cons, const double* x, const double* multipliers,
                              int nnz, int* row_inds, int* col_inds, double* vals) = 0;
    virtual int get_num_vars() const = 0;
    virtual int get_num_cons() const = 0;
    virtual int get_nnz_hess() const = 0;
    virtual int get_nnz_jac() const = 0;

    virtual ~OptimizationProblem() = default;
};


#endif
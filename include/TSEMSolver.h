/*
 * p-multigrid solver for tsem derived linear system
 * criterion of coarse/fine dof is determined by the sum of polynomial order on each dof
 */
#ifndef __TSEMSOLVER_
#define __TSEMSOLVER_

#include <fstream>
#include "lac/sparsity_pattern.h"
#include "lac/sparse_matrix.h"
#include "lac/vector.h"
// #include "AFEPack/AMGSolver.h"

#include "lapacke.h"


template <typename valuetype> class TSEMSolver
{
protected:
    const bool flag_output_err = false;
    const bool flag_output_intermedia = false;
    const double tol_zero = 1.0e-8;

public:
    TSEMSolver(){};
    void ite_Jacobi(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, unsigned int ite_step) const;
    void ite_GaussSeidel(const SparseMatrix<valuetype> &M, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const;
    void ite_BackwardGaussSeidel(const SparseMatrix<valuetype> &M, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const;
    void ite_SymmetricGaussSeidel(const SparseMatrix<valuetype> &M, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const;
    void ite_SOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega, unsigned int ite_step) const;
    void ite_SSOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega, unsigned int ite_step) const;
    void ite_SSOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega,
		  unsigned int max_step, double tol) const;
    void solve_PCG(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r,
    		   double tol = 1.0e-8, unsigned int max_step = 100000, unsigned int flag_preconditioner = 1, unsigned int ite_step = 1, double var = 1) const;
    void solve_InversePower(Vector<valuetype> &x, const SparseMatrix<valuetype> &A, const SparseMatrix<valuetype> &B,
			    bool Flag_Output_Intermediate, double tol = 1.0e-8, unsigned int max_step = 10000) const; // Ax = \lambda_min Bx, x^TBx = I
    void solve_LOBPCG(TSEM<double> &tsem, Vector<valuetype> &x, const SparseMatrix<valuetype> &A, const SparseMatrix<valuetype> &B,
		      bool Flag_Output_Intermediate, double tol = 1.0e-8, unsigned int max_step = 10000) const; // Ax = \lambda_min Bx, x^TBx = I
};

#define TEMPLATE_TSEMSOLVER template<typename valuetype>
#define THIS_TSEMSOLVER TSEMSolver<valuetype>

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_GaussSeidel(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const
{
    const SparsityPattern& sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    for (unsigned int n_ite = 0; n_ite < step; ++n_ite){
        for (unsigned int row = 0; row < sp_matrix.m(); ++row){
            valuetype r0 = r(row);
            for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
                r0 -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
            x(row) = r0 / sp_matrix.global_entry(row_start[row]);
        }
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_BackwardGaussSeidel(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const
{
    const SparsityPattern& sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    for (unsigned int n_ite = 0; n_ite < step; ++n_ite){
        for (int row = sp_matrix.m()-1; row >= 0; --row){
            valuetype r0 = r(row);
            for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
                r0 -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
            x(row) = r0 / sp_matrix.global_entry(row_start[row]);
        }
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_SymmetricGaussSeidel(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, const int &step) const
{
    const SparsityPattern& sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    for (unsigned int n_ite = 0; n_ite < step; ++n_ite){
        for (unsigned int row = 0; row < sp_matrix.m(); ++row){
            valuetype r0 = r(row);
            for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
                r0 -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
            x(row) = r0 / sp_matrix.global_entry(row_start[row]);
        }
	for (int row = sp_matrix.m()-1; row >= 0; --row){
	    valuetype r0 = r(row);
            for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
                r0 -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
            x(row) = r0 / sp_matrix.global_entry(row_start[row]);
	}
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_Jacobi(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, unsigned int ite_step) const
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    Vector<valuetype> y(x.size());
    
    for (int n_ite = 0; n_ite < ite_step; ++n_ite){
    	y = x;
    	for (unsigned int row = 0; row < r.size(); ++row){
    	    valuetype count = (valuetype) r(row);
    	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    		count -= y(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    	    x(row) = count / sp_matrix.global_entry(row_start[row]);
    	}
    }

    // for (unsigned int row = 0; row < r.size(); ++row)
    // 	x(row) = r(row) / sp_matrix.global_entry(row_start[row]);
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_SOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega, unsigned int ite_step) const
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    double omega_ = 1 - omega;


    for (int n_ite = 0; n_ite < ite_step; ++n_ite){
	for (unsigned int row = 0; row < r.size(); ++row){
	    valuetype count = (valuetype) r(row);
	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
	    x(row) = omega_ * x(row) + omega * count / sp_matrix.global_entry(row_start[row]);
	}
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_SSOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega, unsigned int ite_step) const
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    double omega_ = 1 - omega, omega__ = 2 - omega;


    for (int n_ite = 0; n_ite < ite_step; ++n_ite){
    	for (unsigned int row = 0; row < r.size(); ++row){
    	    valuetype count = (valuetype) r(row);
    	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    	    x(row) = omega_ * x(row) + omega * count / sp_matrix.global_entry(row_start[row]);
    	}
    	for (int row = r.size()-1; row >= 0; --row){
    	    valuetype count = (valuetype) r(row);
    	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    	    x(row) = omega_ * x(row) + omega * count / sp_matrix.global_entry(row_start[row]);
    	}
    }


    // for (unsigned int row = 0; row < r.size(); ++row){
    // 	valuetype count = (valuetype) r(row) * omega__;
    // 	for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    // 	    if (col_nums[pos_whole] < row)
    // 		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    // 	x(row) = omega * count / sp_matrix.global_entry(row_start[row]);
    // }
    // for (int row = r.size()-1; row >= 0; --row){
    // 	valuetype count = x(row) * sp_matrix.global_entry(row_start[row]) / omega;
    // 	for (unsigned int pos_whole = row_start[row+1]; pos_whole < row_start[row+1]; ++pos_whole)
    // 	    if (col_nums[pos_whole] > row)
    // 		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    // 	x(row) = omega * count / sp_matrix.global_entry(row_start[row]);
    // }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::ite_SSOR(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r, double omega,
			       unsigned int max_step, double tol) const
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    double omega_ = 1 - omega;
    Vector<valuetype> res(r), tmp(r.size());
    sp_matrix.vmult(tmp, x);
    res.add(-1, tmp);
    double residual_init = res.l1_norm();
    double residual = residual_init;


    for (int n_ite = 0; n_ite < max_step; ++n_ite){
    	for (unsigned int row = 0; row < r.size(); ++row){
    	    valuetype count = (valuetype) r(row);
    	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    	    x(row) = omega_ * x(row) + omega * count / sp_matrix.global_entry(row_start[row]);
    	}
    	for (int row = r.size()-1; row >= 0; --row){
    	    valuetype count = (valuetype) r(row);
    	    for (unsigned int pos_whole = row_start[row]+1; pos_whole < row_start[row+1]; ++pos_whole)
    		count -= x(col_nums[pos_whole]) * sp_matrix.global_entry(pos_whole);
    	    x(row) = omega_ * x(row) + omega * count / sp_matrix.global_entry(row_start[row]);
    	}
	
	res = r;
	sp_matrix.vmult(tmp, x);
	res.add(-1, tmp);
	residual = res.l1_norm();
	if (residual < tol * residual_init) break;
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::solve_PCG(const SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &x, const Vector<valuetype> &r,
				double tol, unsigned int max_step, unsigned int flag_preconditioner, unsigned int ite_step, double var) const
{/* preconditioned conjugate gradient method
  * flag_preconditioner: 1 - diagonal matrix (Jacobi preconditioner); 2 - sor; 3 - ssor
  * var1, var2: extra variable for PCG, ssor: var1 - omega
  */
    // setup
    double omega = var;
    unsigned int smooth_step;

    
    Vector<valuetype> tmp(r.size()), rk(r.size()), yk(x), pk(r);
    // calculate initial residual
    sp_matrix.vmult(rk, x);
    rk.add(-1.0, r); // r0 = sp_matrix * x0 - rhs
    switch (flag_preconditioner){
    case 0:
	yk = rk;
	break;
    case 1:
	yk = rk;
	for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
	    yk(ind_dof) /= sp_matrix.diag_element(ind_dof);
	break;
    case 2:
	if (omega <= 0 || omega >= 2){
	    std::cerr << "invalid sor parameter!\n";
	    return;
	}
	ite_SOR(sp_matrix, yk, rk, omega, ite_step);
	break;
    case 3:
	if (omega <= 0 || omega >= 2){
	    std::cerr << "invalid ssor parameter!\n";
	    return;
	}
	ite_SSOR(sp_matrix, yk, rk, omega, ite_step);
	// ite_SSOR(sp_matrix, yk, rk, omega, 1000, 1.0e-1);
	break;
    default:
	std::cerr << "undefined preconditioner type!\n";
	return;
    }
    pk = yk; pk *= -1;
    valuetype residual = rk.l1_norm();
    valuetype init_residual = residual;
    std::cerr << "\tinitial residual " << init_residual << '\n';
    unsigned int iteration_step = 0;


    std::vector<valuetype> err(max_step+2);
    err[0] = residual;


    valuetype rkTyk = (valuetype) 0, pkTApk, alphak, rkTyk_next, betak;
    for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
	rkTyk += rk(ind_dof) * yk(ind_dof);
    // while (residual >= tol*init_residual){
    while (residual >= tol){
	// calculate step size
        pkTApk = (valuetype) 0;
	sp_matrix.vmult(tmp, pk);
	for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
	    pkTApk += pk(ind_dof) * tmp(ind_dof);
	alphak = rkTyk / pkTApk;

	// update xk
	x.add(alphak, pk);

	// update rk
	rk.add(alphak, tmp);

	// calculate yk+1
	// yk = rk;
	yk *= 0;
	switch (flag_preconditioner){
	case 0:
	    break;
	case 1:
	    yk = rk;
	    for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
		yk(ind_dof) /= sp_matrix.diag_element(ind_dof);
	    break;
	case 2:
	    ite_SOR(sp_matrix, yk, rk, omega, ite_step);
	    break;
	case 3:
	    // yk = rk;
	    // for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
	    // 	yk(ind_dof) /= sp_matrix.diag_element(ind_dof);
	    ite_SSOR(sp_matrix, yk, rk, omega, ite_step);
	    // ite_SSOR(sp_matrix, yk, rk, omega, 1000, 1.0e-1);
	    break;
	}


	// calcualte step size for updating direction pk
	rkTyk_next = (valuetype) 0;
	for (int ind_dof = 0; ind_dof < r.size(); ++ind_dof)
	    rkTyk_next += rk(ind_dof) *  yk(ind_dof);
	betak = rkTyk_next / rkTyk;

	// update direction pk+1
	pk *= betak;
	pk.add(-1, yk);

	// record residual on this step
	residual = rk.l1_norm();

	// update iteration step and rkTrk
	iteration_step++;
	err[iteration_step] = residual;
	if (iteration_step > max_step) break;
	rkTyk = rkTyk_next;

	if (flag_output_intermedia)
	    std::cout << "iterationr_step = " << iteration_step << ", residual = " << residual << '\n';
    }


    if (flag_output_err){
	std::ofstream output("./PCG_Data.m");
	for (int i = 0; i <= iteration_step; ++i)
	    output << "err(" << i+1 << ") = " << err[i] << ";\n";
	output.close();
    }
    

    // output residual and iteration info
    // if (residual < tol*init_residual){
    if (residual < tol){
        std::cerr << "\r\tconverge with residual " << residual
                  << " at step " << iteration_step << "." << std::endl;
    }
    else {
        std::cerr << "\r\tfailed to converge with residual " << residual
                  << " at step " << iteration_step << "." << std::endl;
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::solve_InversePower(Vector<valuetype> &x, const SparseMatrix<valuetype> &A, const SparseMatrix<valuetype> &B,
					 bool Flag_Output_Intermediate, double tol, unsigned int max_step) const
{// Ax = \lambda_min Bx, x^TBx = I, break criterion: e = x^TAx, |e_last-e_now| < tol
    Vector<valuetype> rhs(x.size()), tmp(x.size());
    valuetype e_last, e_now, cnt;
    B.vmult(rhs, x); // preparation
    for (unsigned int n_ite = 0; n_ite < max_step; ++n_ite){
	// iterate
	solve_PCG(A, x, rhs, tol*1.0e-2, max_step*100);
	// normalize
	cnt = 0;
	B.vmult(rhs, x);
	for (unsigned int ind_dof = 0; ind_dof < x.size(); ++ind_dof)
	    cnt += rhs(ind_dof) * x(ind_dof);
	x /= sqrt(cnt);
	// calculate e_now
	e_now = 0;
	A.vmult(tmp, x);
	for (unsigned int ind_dof = 0; ind_dof < x.size(); ++ind_dof)
	    e_now += tmp(ind_dof) * x(ind_dof);
	// determine whether break, update e_last
	if (n_ite > 0 && fabs(e_now-e_last) < tol) break;
	if (Flag_Output_Intermediate && n_ite > 0)
	    std::cout << "\tn_ite = " << n_ite << ", e_now = " << e_now << ", dif = " << fabs(e_now-e_last) << '\n';
	e_last = e_now;
    }
}

TEMPLATE_TSEMSOLVER
void THIS_TSEMSOLVER::solve_LOBPCG(TSEM<double> &tsem, Vector<valuetype> &x, const SparseMatrix<valuetype> &A, const SparseMatrix<valuetype> &B,
				   bool Flag_Output_Intermediate, double tol, unsigned int max_step) const
{// Ax = \lambda_min Bx, x^TBx = I, break criterion: e = x^TAx, |e_last-e_now| < tol
     // lapack parameters
    double a[9], b[9];
    double vl, vu;
    int m;
    double w[3];
    int ldz = 3;
    double z[3];
    int ifail[3], info;

    
    // initialize
    // vector consist of [x, p, w]
    unsigned int n_dof_total = x.size();
    std::vector<Vector<double> >  xpw(3, Vector<double> (n_dof_total));
    std::vector<Vector<double> > Axpw(3, Vector<double> (n_dof_total));
    std::vector<Vector<double> > Bxpw(3, Vector<double> (n_dof_total));
    double eig, eig_last;
    double tmp_n = 0, tmp_d = 0; // n for numerator, d for denominator
    xpw[0] = x;
    A.vmult(Axpw[0], xpw[0]);
    B.vmult(Bxpw[0], xpw[0]);
    for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof){
	tmp_n += xpw[0](ind_dof) * Axpw[0](ind_dof);
	tmp_d += xpw[0](ind_dof) * Bxpw[0](ind_dof);
    }
    eig_last = eig = tmp_n / tmp_d;
    // the first search direction is set to be residual
    xpw[1] = Bxpw[0];    xpw[1] *= eig;    xpw[1].add(-1.0, Axpw[0]);
    A.vmult(Axpw[1], xpw[1]);
    B.vmult(Bxpw[1], xpw[1]);
    // vec = [xpw, Axpw, Bxpw] on last step
    std::vector<Vector<double> > vec_last(9, Vector<double> (n_dof_total));
    // x, Ax, Bx
    vec_last[0] = xpw[0];    vec_last[3] = Axpw[0];    vec_last[6] = Bxpw[0];
    // p, Ap, Bp
    vec_last[1] = xpw[1];    vec_last[4] = Axpw[1];    vec_last[7] = Bxpw[1];
    // TSEMSolver<double> solver;
    Vector<double> resi(n_dof_total);
    for (int n_ite = 0; n_ite < max_step; ++n_ite){
	// get residual
	if (n_ite == 0)
	    resi = xpw[1];
	else{
	    resi = Bxpw[0];
	    resi *= eig;
	    resi.add(-1.0, Axpw[0]);
	}

	// assign sparse matrix for preconditioner
	std::vector<unsigned int> nnz(n_dof_total, 0);
	SparsityPattern spp;
	SparseMatrix<double> spm;
        SparseMatrix<double>::iterator spm_ite = tsem.stiff_matrix.begin(0);
	SparseMatrix<double>::iterator spm_end = tsem.stiff_matrix.end(tsem.stiff_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) nnz[spm_ite->row()]++;
	spm_ite = tsem.mass_matrix.begin(0);
	spm_end = tsem.mass_matrix.end(tsem.mass_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) nnz[spm_ite->row()]++;    
	spp.reinit(n_dof_total, n_dof_total, nnz);
	spm_ite = tsem.stiff_matrix.begin(0);
	spm_end = tsem.stiff_matrix.end(tsem.stiff_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) spp.add(spm_ite->row(), spm_ite->column());
	spm_ite = tsem.mass_matrix.begin(0);
	spm_end = tsem.mass_matrix.end(tsem.mass_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) spp.add(spm_ite->row(), spm_ite->column());
	spp.compress();
	spm.reinit(spp);
	spm_ite = tsem.stiff_matrix.begin(0);
	spm_end = tsem.stiff_matrix.end(tsem.stiff_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) spm.add(spm_ite->row(), spm_ite->column(), spm_ite->value() * 0.5);
	spm_ite = tsem.mass_matrix.begin(0);
	spm_end = tsem.mass_matrix.end(tsem.mass_matrix.m()-1);
	for (; spm_ite != spm_end; ++spm_ite) spm.add(spm_ite->row(), spm_ite->column(), spm_ite->value() * -eig);

	// solve
	tsem.impose_zero_boundary_condition(spm);
	solve_PCG(spm, xpw[2], resi, tol*1.0e-2, max_step*100);


	// assign Axpw, Bxpw for residual
	A.vmult(Axpw[2], xpw[2]);
	B.vmult(Bxpw[2], xpw[2]);

	// copy info about w to vec_last
	vec_last[2] =  xpw[2];
	vec_last[5] = Axpw[2];
	vec_last[8] = Bxpw[2];
    
	
	// assign local matrix, calculate its n_pair eigenvectors for corresponding n_pair smallest eigenvalue
	for (unsigned int row = 0; row < 3; ++row)
	    for (unsigned int col = 0; col < 3; ++col){
		tmp_n = tmp_d = 0;
		for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof){
		    tmp_n += xpw[row](ind_dof) * Axpw[col](ind_dof);
		    tmp_d += xpw[row](ind_dof) * Bxpw[col](ind_dof);
		}
		a[row*3 + col] = tmp_n;
		b[row*3 + col] = tmp_d;
	    }


	// solve local eigenvalue problem
	info = LAPACKE_dsygvx(102, 1, 'V', 'I', 'U', 3, a, 3, b, 3, vl, vu, 1, 1, -1, &m, w, z, ldz, ifail);
	if (info > 0){
	    std::cout << "n_ite_lobpcg = " << n_ite << ", the algorithm failed to compute eigenvalues."
		      << ", info = " << info
		      << "\n";
	    break;
	}


	// update xpw, Axpw and Bxpw
	// x, Ax, Bx
	for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
	    xpw[0](ind_dof) = Axpw[0](ind_dof) = Bxpw[0](ind_dof) = 0;
	for (unsigned int ind_p = 0; ind_p < 3; ++ind_p){ // row index of z
	    xpw[ 0].add(z[ind_p], vec_last[ind_p]); // x, from xpw
	    Axpw[0].add(z[ind_p], vec_last[ind_p + 3]); // Ax, from Axpw
	    Bxpw[0].add(z[ind_p], vec_last[ind_p + 6]); // Bx, from Axpw
	}
	// p, Ap, Bp
	for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
	    xpw[1](ind_dof) = Axpw[1](ind_dof) = Bxpw[1](ind_dof) = 0;
	for (unsigned int ind_p = 1; ind_p < 3; ++ind_p){ // the update of p only relates to p and w
	    xpw[ 1].add(z[ind_p], vec_last[ind_p]); // p, from xpw
	    Axpw[1].add(z[ind_p], vec_last[ind_p + 3]); // Ap, from Axpw
	    Bxpw[1].add(z[ind_p], vec_last[ind_p + 6]); // Bp, from Bxpw
	}

	
	// calculate err, break if dif met criterion
	double err = fabs(w[0] - eig_last);
	if (Flag_Output_Intermediate)
	    std::cout << "\tn_ite_lobpcg = " << n_ite << ", err_ene = " << err << '\n';
	if (err < tol) break;
	
	// update vec_last, eig
	// vec_last
	vec_last[0] =  xpw[0]; vec_last[1] =  xpw[1]; vec_last[3] = Axpw[0];
	vec_last[4] = Axpw[1]; vec_last[6] = Bxpw[0]; vec_last[7] = Bxpw[1];
	// eig
	eig_last = eig = w[0];
    }

    x = xpw[0];
}

#endif

/*
 * Tetrahedral Spectral Element Method example for ground state calculation
 *     Flag_Case = 0: harmonic oscillator, with inverse power method
 *     Flag_Case = 1: hydrogen atom, with LOBPCG method (locally optimal problem is solved by LAPACK)
 */

#include <iostream>
#include <iomanip>

#include <AFEPack/Geometry.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>

#define PI (4.0*atan(1.0))
#define DIM 3

#include "Multiindex.h"
#include "Option.h"
#include "Correspondence.h"
Correspondence<DIM> correspondence;
#include "TetrahedralSEM.h"
#include "TSEMSolver.h"

const unsigned int Flag_Case = 1; // 0: harmonic oscillator; 1: hydrogen
const double Ene_Ref = (Flag_Case == 0) ? 1.5 : -0.5;
const double Tol_Solver = 1.0e-8;

double V(const double *);
double rho(const double *);
void setup_linearSystem(TSEM<double> &, std::vector<std::vector<double> > &,
			std::vector<unsigned int> &, SparsityPattern &, SparseMatrix<double> &,
			std::vector<unsigned int> &, SparsityPattern &, SparseMatrix<double> &,
			Vector<double> &);


int main(int argc, char * argv[])
{
    // setup
    std::cout << std::setprecision(15);
    init_unitary_multiindex();
    init_zero_multiindex();
    
    
    // read mesh from .mesh file
    HGeometryTree<DIM> h_tree;
    h_tree.readMesh(argv[1]);
    IrregularMesh<DIM> *irregular_mesh;
    irregular_mesh = new IrregularMesh<DIM>;
    irregular_mesh->reinit(h_tree);
    irregular_mesh->semiregularize();
    irregular_mesh->regularize(false);
    RegularMesh<DIM> &mesh = irregular_mesh->regularMesh();
    
    
    // prepare 3D template info for FEM space
    TemplateGeometry<DIM> template_geometry;
    CoordTransform<DIM, DIM> coord_transform;
    TemplateDOF<DIM> template_dof;
    BasisFunctionAdmin<double, DIM, DIM> basis_function;
    template_geometry.readData("tetrahedron.tmp_geo");
    coord_transform.readData("tetrahedron.crd_trs");
    template_dof.reinit(template_geometry);    template_dof.readData("tetrahedron.2.tmp_dof");
    basis_function.reinit(template_dof);       basis_function.readData("tetrahedron.2.bas_fun");
    std::vector<TemplateElement<double, DIM, DIM> > template_element(1);
    template_element[0].reinit(template_geometry, template_dof, coord_transform, basis_function);

    
    // build FEM space
    FEMSpace<double, DIM> fem_space(mesh, template_element);
    int n_element = mesh.n_geometry(DIM);
    fem_space.element().resize(n_element);
    for (int i = 0; i < n_element; ++i) fem_space.element(i).reinit(fem_space, i, 0);
    fem_space.buildElement();
    fem_space.buildDof();
    fem_space.buildDofBoundaryMark();

    
    // setup for multiindex
    int M = atoi(argv[2]); // polynomial order
    correspondence.init(M);
    

    // initialize TSEM space
    TSEM<double> tsem(M, mesh, fem_space);
    // build boundary flag
    tsem.build_flag_bm_mesh(mesh);

    
    // calculate discrete potential value
    std::vector<std::vector<double> > value_V(n_element, std::vector<double> (tsem.n_q_point[2]));
    for (unsigned int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	std::vector<AFEPack::Point<3> > q_point = fem_space.element(ind_ele).local_to_global(tsem.QPoint);
	for (int p = 0; p < tsem.n_q_point[2]; ++p)
	    value_V[ind_ele][p] = V(q_point[p]);
    }
    // assign matrix for eigenvalue problem
    std::vector<unsigned int> n_nonzero_per_row_A, n_nonzero_per_row_B;
    SparsityPattern sp_pattern_A, sp_pattern_B;
    SparseMatrix<double> sp_matrix_A, sp_matrix_B;
    Vector<double> psi;
    setup_linearSystem(tsem, value_V,
		       n_nonzero_per_row_A, sp_pattern_A, sp_matrix_A,
		       n_nonzero_per_row_B, sp_pattern_B, sp_matrix_B,
		       psi);

    // solve eigenvalue system
    TSEMSolver<double> solver;
    switch (Flag_Case){
    case 0:
	std::cout << "Calculate the ground state of harmonic oscillator by inverse power method.\n";
	solver.solve_InversePower(psi, sp_matrix_A, sp_matrix_B, true, Tol_Solver);
	break;
    case 1:
	std::cout << "Calculate the ground state of hydrogen atom by LOBPCG method.\n";
	solver.solve_LOBPCG(tsem, psi, sp_matrix_A, sp_matrix_B, true, Tol_Solver, 100);
	break;
    default:
	std::cout << "Undefined case!\n\n";
    }


    // output energy and errors
    // calculate discrete exact density
    std::vector<std::vector<double> > value_rho_exact(n_element, std::vector<double> (tsem.n_q_point[2]));
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	std::vector<AFEPack::Point<3> > q_point = fem_space.element(ind_ele).local_to_global(tsem.QPoint);
	for (int p = 0; p < tsem.n_q_point[2]; ++p)
	    value_rho_exact[ind_ele][p] = rho(q_point[p]);
    }
    // calculate energy, output info
    double energy = 0;
    Vector<double> tmp(psi.size());
    sp_matrix_A.vmult(tmp, psi);
    for (int i = 0; i < psi.size(); ++i)
	energy += tmp(i) * psi(i);
    // calculate l2 error;
    double error_l2 = tsem.calc_l2_density_difference(psi, value_rho_exact);
    std::cout << "energy = " << energy << ", err_ene = " << fabs(energy-Ene_Ref) << ", l2 error = " << error_l2 << '\n';

    
    return 0;
}

double V(const double* p)
{
    double x = p[0], y = p[1], z = p[2];
    switch (Flag_Case){
    case 0: return (x*x + y*y + z*z) * 0.5;
    case 1: return -1.0 / sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    }
}

double rho(const double* p)
{
    double x = p[0], y = p[1], z = p[2];
    switch (Flag_Case){
    case 0: return exp(-(x*x + y*y + z*z)) / pow(sqrt(PI), 3);
    case 1: return exp(-2*sqrt(x*x + y*y + z*z)) / PI;
    }
}

void setup_linearSystem(TSEM<double> &tsem, std::vector<std::vector<double> > &value_V,
			std::vector<unsigned int> &nnz_A, SparsityPattern &spp_A, SparseMatrix<double> &spm_A,
			std::vector<unsigned int> &nnz_B, SparsityPattern &spp_B, SparseMatrix<double> &spm_B,
			Vector<double> &x)
{
    // assign mass type matrix for potential, M_pot
    std::vector<unsigned int> n_nonzero_per_row_mass_V;
    SparsityPattern sp_mass_V;
    SparseMatrix<double> mass_matrix_V;
    tsem.build_mass_V_matrix(n_nonzero_per_row_mass_V, sp_mass_V, mass_matrix_V, value_V);

    
    // assign matrix A = 0.5S+M_pot
    int n_dof_total = tsem.n_dof_total;
    SparseMatrix<double>::iterator spm_ite = tsem.stiff_matrix.begin(0);
    SparseMatrix<double>::iterator spm_end = tsem.stiff_matrix.end(tsem.mass_matrix.m()-1);
    nnz_A.resize(n_dof_total, 0);
    // count nonzero entry
    for (; spm_ite != spm_end; ++spm_ite) nnz_A[spm_ite->row()]++;
    spm_ite = mass_matrix_V.begin(0);
    spm_end = mass_matrix_V.end(mass_matrix_V.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) nnz_A[spm_ite->row()]++;
    // assign nonzero entry location
    spp_A.reinit(n_dof_total, n_dof_total, nnz_A);
    spm_ite = tsem.stiff_matrix.begin(0);
    spm_end = tsem.stiff_matrix.end(tsem.stiff_matrix.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) spp_A.add(spm_ite->row(), spm_ite->column());
    spm_ite = mass_matrix_V.begin(0);
    spm_end = mass_matrix_V.end(mass_matrix_V.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) spp_A.add(spm_ite->row(), spm_ite->column());
    spp_A.compress();
    // assign nonzero entry
    spm_A.reinit(spp_A);
    spm_ite = tsem.stiff_matrix.begin(0);
    spm_end = tsem.stiff_matrix.end(tsem.stiff_matrix.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) spm_A.add(spm_ite->row(), spm_ite->column(), spm_ite->value() * 0.5);
    spm_ite = mass_matrix_V.begin(0);
    spm_end = mass_matrix_V.end(mass_matrix_V.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) spm_A.add(spm_ite->row(), spm_ite->column(), spm_ite->value());
    // impose zero boundary condition
    tsem.impose_zero_boundary_condition(spm_A);


    // assign matrix B = M
    nnz_B.resize(n_dof_total, 0);
    spm_ite = tsem.mass_matrix.begin(0);
    spm_end = tsem.mass_matrix.end(tsem.mass_matrix.m()-1);
    for (; spm_ite != spm_end; ++spm_ite) nnz_B[spm_ite->row()]++;
    spp_B.reinit(n_dof_total, n_dof_total, nnz_B);
    spm_ite = tsem.mass_matrix.begin(0);
    for (; spm_ite != spm_end; ++spm_ite) spp_B.add(spm_ite->row(), spm_ite->column());
    spp_B.compress();
    spm_B.reinit(spp_B);
    spm_ite = tsem.mass_matrix.begin(0);
    for (; spm_ite != spm_end; ++spm_ite) spm_B.add(spm_ite->row(), spm_ite->column(), spm_ite->value());
    tsem.impose_zero_boundary_condition(spm_B);


    // assign wave function psi, with random initial guess
    x.reinit(n_dof_total);
    for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
        x(ind_dof) = rand() * 1.0 / RAND_MAX;
    double normal_tmp = 0;
    Vector<double> vmult_tmp(n_dof_total);
    tsem.mass_matrix.vmult(vmult_tmp, x);
    for (unsigned int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
        normal_tmp += x(ind_dof) * vmult_tmp(ind_dof);
    x /= sqrt(normal_tmp);
    tsem.impose_zero_boundary_condition(x);
}

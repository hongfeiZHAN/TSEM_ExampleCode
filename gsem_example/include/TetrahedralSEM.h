/*
 * realization of tetrahedral spectral element method
 *
 * used correspondence, which is declared in Correspondence.h
 * require Unitary_Multiindex and Zero_Multiindex defined in Multiindex.h
 *     and been initialized before using TSEM
 */
#ifndef __TETRAHEDRALSEM_
#define __TETRAHEDRALSEM_

#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include "AFEPack/FEMSpace.h"
#include "AFEPack/Geometry.h"
#include "lac/sparsity_pattern.h"
#include "lac/sparse_matrix.h"
#include "lac/vector.h"
#include "Multiindex.h"
#include "Correspondence.h"
#include "Option.h"

// #define VALUETYPE_ITERATOR_TSEM double

template <typename valuetype> class TSEM
{
protected:
    const valuetype tol_zero = 1.0e-12;
public:
    int M, n_index; // polynomial order, number of polynomials

public:
    // int n_quad_accuracy[3];
    std::vector<int> n_q_point; // [i]: number of quadrature point in i dimensional rule
    std::vector<AFEPack::Point<3> > QPoint; // 3-d quadrature point
    std::vector<std::vector<std::vector<valuetype> > > QPoint_Barycentric; // 1 & 2-d quadrature point in barycenter coordinate
    std::vector<std::vector<valuetype> > Weight; // 1, 2 & 3-d quadrature weight

protected:
    int n_dof_edge, n_dof_face;
    // number of nonzero contribution on edges
    std::vector<std::vector<std::vector<int> > > n_expr_edge;
    // index of nonzero contribution on edges
    std::vector<std::vector<std::vector<std::vector<int> > > > ind_expr_edge;
    // value of nonzero contribution on edges
    std::vector<std::vector<std::vector<std::vector<valuetype> > > > val_expr_edge;
    // number of nonzero contribution on face
    std::vector<std::vector<int> > n_expr_face;
    // index of nonzero contribution on face
    std::vector<std::vector<std::vector<int> > > ind_expr_face;
    // value of nonzero contribution on face
    std::vector<std::vector<std::vector<valuetype> > > val_expr_face;

    std::vector<std::vector<AFEPack::Point<3> > > point_ref_mesh; // [i = 0:3], barycenter of i dimensional geometry
public:
    int n_dof_geometry[4];
    int n_geometry[4];
    int n_element, n_geometry_total, n_dof_total;
    // number_node[ind+1=1:2][order+1=1:ind+1][]: the node number of ind+1-dimensional geometry
    std::vector<std::vector<std::vector<int> > > number_node;
    std::vector<valuetype> val_volume;
protected:
    std::vector<std::vector<int> > number_edge;
    // flag that whether order of start/end point on global face is the same as that on global edge
    std::vector<std::vector<bool> > flag_sameorder_edgeonface;
    // type of projection for edge and face on each element
    std::vector<std::vector<std::vector<int> > > type_projection;
    std::vector<std::vector<int> > type_projection_inv; // inverse projection type, for face on each element
    std::vector<std::vector<std::vector<int> > > index_geometry_onelement;

    // int n_dof;
    /* {weight_local, geometry_dimension, geometry_order}:
     *    weight_local:       a number for sorting, read from fem_space, determine the order of all dimensional geomtry
     *    geometry_dimension: the dimension of geometry corresponds to this set
     *    geometry_order:     the order of geometry in the same dimensional ones corresponds to this set
     */
    std::vector<valuetype> weight_location; // the weight_location of 0: 3 dimensional geometry in turns
    std::vector<int> geometry_dimension;
    std::vector<int> geometry_order; // [i = 0:n_geometry_total-1] the order of i-th entry in weight_location, whose dimension is geometry_dimension[i]
    // correspondence between fem dof/element and geometry
    // transform_femdof2geometry: index from fem dof to 0 & 1 dimensional geometry, for 1 dimensional geometry, its index plus mesh.n_geometry(0)
    std::vector<int> transform_femdof2geometry;
    // location_geometry: location of all geometry (0:3 dimensional) according to increasing order of weight_location
    std::vector<int> location_geometry;
    // location_actualdof: start index of geometry in actual discretized matrix
    std::vector<int> location_actualdof;
public:
    // expression of local coefficient by linear summation of global ones
    std::vector<std::vector<int> >                     transform_n_global2local;
    std::vector<std::vector<std::vector<int> > >       transform_ind_global2local;
    std::vector<std::vector<std::vector<valuetype> > > transform_val_global2local;
protected:
    // expression of global coefficient on each-dimensional geometry by local ones, equivalent to column compress of transformation matrix
    std::vector<std::vector<int> >                     transform_n_local2global;
    std::vector<std::vector<std::vector<int> > >       transform_ind_local2global;
    std::vector<std::vector<std::vector<valuetype> > > transform_val_local2global;
    
    std::vector<std::vector<std::vector<valuetype> > > basis_value; // [ind = 0:2][p = 0:n_q_point[ind]-1][]: basis_value for ind+1 dimensional quadrature region
    std::vector<std::vector<std::vector<valuetype> > > basis_value_interp; // basis value for interpolation
    std::vector<std::vector<std::vector<valuetype> > > basis_value_addition; // additional basis value for interpolation on face
    std::vector<std::vector<std::vector<valuetype> > > basis_gradient; // [p = 0:n_q_point[2]-1][ind_index = 0:n_index-1][3]: gradient of basis function on quadrature points

    std::vector<int> n_transform_local;
    std::vector<std::vector<int> > transform_local;
    std::vector<std::vector<valuetype> > weight_transform_local;
public:
    std::vector<std::vector<valuetype> > basis_value_actual; // [p = 0:n_q_point[ind]-1][]: basis_value for 3 dimensional quadrature region
    std::vector<std::vector<std::vector<valuetype> > > basis_gradient_actual; // actual gradient of basis on quadrature points

protected:
    std::vector<int> n_index_variation; // number of the coefficient for the derivative of generalized jacobi polynomial
    std::vector<std::vector<Multiindex<3> > > index_variation; // variation of multiindex corresponding to the coefficients
    std::vector<std::vector<std::vector<valuetype> > > coefficient_derivative;
    std::vector<SparsityPattern> sp_stiff_matrix_element_actual;
    std::vector<SparseMatrix<valuetype> > stiff_matrix_element_actual;

    SparsityPattern sp_stiff_matrix;

    SparsityPattern sp_mass_matrix_element_actual;
    SparseMatrix<valuetype> mass_matrix_element_actual;

    SparsityPattern sp_mass_matrix;
    SparsityPattern sp_mass_matrix_essential;

public:
    std::vector<std::vector<bool> > flag_bm; // boundary flag for all dimensional geometry
    std::vector<bool> flag_bm_dof;

protected:
    std::vector<std::vector<unsigned int> > conversion;

    std::vector<int> n_quadPoint{20, 130, 923};
    std::vector<std::string> path_quadInfo{"./local/quad_info/quad_info_1d_p38.dat",
					   "./quad_info/quad_2d_deg26_n130.dat",
					   "./quad_info/quad_3d_deg26_n923.dat"};
    std::string path_interpInfo = "./quad_info/Info_Interpolate";

protected:
    valuetype calc_generalized_jacobi_polynomial(int alpha, int beta, int k, valuetype x);
    valuetype calc_coefficient_a(int alpha, int beta, int ind, int k);
    valuetype calc_coefficient_b(int alpha, int beta, int ind, int k);
    valuetype calc_coefficient_c(int alpha, int beta, int ind, int k);
    valuetype calc_coefficient_d(int alpha, int beta, int k);
    valuetype calc_coefficient_e(int alpha, int beta, int ind, int k);
    valuetype calc_coefficient_g(int alpha, int beta, int ind, int k);
    valuetype calc_coefficient_rho(Multiindex<3> index);
    valuetype calc_coefficient_kappa(Multiindex<3> index);
    valuetype calc_coefficient_theta(Multiindex<3> index);
    valuetype calc_coefficient_D(int ind_derivative, int ind_variation, Multiindex<3> index);
    bool is_on_boundary(AFEPack::Point<3> &p, valuetype bnd_left, valuetype bnd_right);
    bool is_on_boundary(AFEPack::Point<3> &p, valuetype radius);
    valuetype calc_length_line(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1);
    valuetype calc_area_triangle(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1, AFEPack::Point<3> &p2);
public:
    valuetype calc_volume_tetrahedron(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1, AFEPack::Point<3> &p2, AFEPack::Point<3> &p3);
public:
    TSEM(){};
    TSEM(int polynomial_order,
	 std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename,
	 const std::string &interp_filename,
	 RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space){
	init(polynomial_order,
	     n_quadrature_point, quad_filename,
	     interp_filename,
	     mesh, fem_space);
	       
    };
    TSEM(int polynomial_order, RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space){
	init(polynomial_order, mesh, fem_space);
    };

    void init(int polynomial_order,
	      std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename,
	      const std::string &interp_filename,
	      RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space);
    void init(int polynomial_order, RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space);
    void init_lazy(int polynomial_order,
	      std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename,
	      const std::string &interp_filename,
	      RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space);

    SparseMatrix<valuetype> stiff_matrix;
    SparseMatrix<valuetype> mass_matrix;
    SparseMatrix<valuetype> mass_matrix_essential;
    
    void read_parameter(int polynomial_order);
    void read_quad_info(std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename);
    void read_interp_info(const std::string &interp_filename);
    void build_geometry_info(RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space);
    void build_basis_value();
    void build_local_transform();
    void build_global_transform(RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space);
    void build_stiff_matrix_init();
    void build_stiff_matrix(FEMSpace<double, 3> &fem_space);
    void build_mass_matrix_init();
    void build_mass_matrix(FEMSpace<double, 3> &fem_space);
    void build_mass_V_matrix(std::vector<unsigned int> &n_nonzero_per_row_mass_V, SparsityPattern &sp_pattern, SparseMatrix<valuetype> &sp_matrix,
			     std::vector<std::vector<valuetype> > &value_V);
    
    void calc_rhs(Vector<valuetype> &rhs, std::vector<std::vector<valuetype> > &value_f);
    void build_flag_bm_cube(valuetype bnd_left, valuetype bnd_right);
    void build_flag_bm_ball(valuetype radius);
    void build_flag_bm_cube(valuetype bnd_left, valuetype bnd_right, valuetype axis[]);
    void build_flag_bm_mesh(RegularMesh<3> &mesh);
    void impose_zero_boundary_condition(SparseMatrix<valuetype> &sp_matrix);
    void impose_zero_boundary_condition(Vector<valuetype> &rhs);
    void impose_boundary_condition_rowOnly(SparseMatrix<valuetype> &sp_matrix);
    void impose_boundary_condition_rowOnly(SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &rhs, std::vector<std::vector<std::vector<valuetype> > > &value_bnd);
    void impose_boundary_condition(SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &rhs, std::vector<std::vector<std::vector<valuetype> > > &value_bnd,
				   bool flag_modify_matrix);

    void calc_coef_onElement(Vector<valuetype> &src, std::vector<valuetype> &dst, unsigned int ind_ele);
    void calc_val_qp_onElement(Vector<valuetype> &src, std::vector<valuetype> &dst, unsigned int ind_ele);

    valuetype calc_l2_error(Vector<valuetype> &sol, std::vector<std::vector<valuetype> > &val_exc);
    valuetype calc_l2_error_gradient(RegularMesh<3> &mesh,
				     Vector<valuetype> &sol, std::vector<std::vector<std::vector<valuetype> > > &val_g_exc);
    valuetype calc_l2_density_difference(Vector<valuetype> &u, std::vector<std::vector<valuetype> > &v);
    valuetype calc_l2_density_difference(Vector<valuetype> &u, Vector<valuetype> &v);
    valuetype calc_l2_difference(Vector<valuetype> &u, Vector<valuetype> &v);
    valuetype calc_l2_difference(Vector<valuetype> &u, Vector<valuetype> &v, int polynomial_order, int flag);
    valuetype calc_l2_error_component(Vector<valuetype> &u, int polynomial_order_less, int polynomial_order_more);
    valuetype calc_density_l2_difference(std::vector<Vector<valuetype> > &u, std::vector<Vector<valuetype> > &v, std::vector<valuetype> &n_occupation);

    void calc_interpolation(Vector<valuetype> &interp, std::vector<std::vector<std::vector<valuetype> > > &val_interp);

    void read_coef(Vector<valuetype> &dst, std::string filename);
    void read_coef(std::vector<Vector<valuetype> > &dst, std::string filename);
    void write_coef(Vector<valuetype> &src, std::string filename);
    void write_coef(std::vector<Vector<valuetype> > &src, std::string filename);

    void calc_sum_dof(std::vector<unsigned int> &sum_dof);

    void calc_val_qpoint(Vector<valuetype> &src, int ind_element, std::vector<valuetype> &dst);

    // bool is_on_element(RegularMesh<3> &mesh, int ind_dim, int ind_geo, AFEPack::Point<3> &pos);
    valuetype calc_val_inElement(Correspondence<3> &correspondence, RegularMesh<3> &mesh, int ind_ele, AFEPack::Point<3> &pos, std::vector<Vector<valuetype> > &psi);
    valuetype calc_val_point(Correspondence<3> &correspondence, RegularMesh<3> &mesh, AFEPack::Point<3> &pos, std::vector<Vector<valuetype> > &psi, std::vector<valuetype> &n_occupation);
};


#define TEMPLATE_TSEM template<typename valuetype>
#define THIS_TSEM TSEM<valuetype>


TEMPLATE_TSEM
void THIS_TSEM::init(int polynomial_order,
		     std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename,
		     const std::string &interp_filename,
		     RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space)
{
    read_parameter(polynomial_order);
    read_quad_info(n_quadrature_point, quad_filename);
    read_interp_info(interp_filename);
    build_geometry_info(mesh, fem_space);
    build_basis_value();
    build_local_transform();
    build_global_transform(mesh, fem_space);
    build_stiff_matrix_init();
    build_stiff_matrix(fem_space);
    build_mass_matrix_init();
    build_mass_matrix(fem_space);
};

TEMPLATE_TSEM
void THIS_TSEM::init(int polynomial_order, RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space)
{
    read_parameter(polynomial_order);
    read_quad_info(n_quadPoint, path_quadInfo);
    read_interp_info(path_interpInfo);
    build_geometry_info(mesh, fem_space);
    build_basis_value();
    build_local_transform();
    build_global_transform(mesh, fem_space);
    build_stiff_matrix_init();
    build_stiff_matrix(fem_space);
    build_mass_matrix_init();
    build_mass_matrix(fem_space);
};

TEMPLATE_TSEM
void THIS_TSEM::init_lazy(int polynomial_order,
			  std::vector<int> &n_quadrature_point, std::vector<std::string> &quad_filename,
			  const std::string &interp_filename,
			  RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space)
{
    read_parameter(polynomial_order);
    read_quad_info(n_quadrature_point, quad_filename);
    read_interp_info(interp_filename);
    build_geometry_info(mesh, fem_space);
    build_basis_value();
    build_local_transform();
    build_global_transform(mesh, fem_space);
};


TEMPLATE_TSEM
void THIS_TSEM::read_parameter(int polynomial_order)
{
    M = polynomial_order;
    n_index = correspondence.n_index();
    std::cerr << "read coefficient, n_index = " << n_index << '\n';
    std::cerr << "set polynomial order for TSEM, M = " << M << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::read_quad_info(std::vector<int> &n_qp, std::vector<std::string> &q_fn)
{ // read quadrature information, initialize corresponding variables

    n_q_point.resize(3);
    // std::cerr << "n_q_point.size() = " << n_q_point.size() << '\n';
    // record number of quadrature points
    for (int ind = 0; ind < 3; ++ind)
	n_q_point[ind] = n_qp[ind];
    // initialize QPoint, QPoint_Barycentric
    QPoint.resize(n_q_point[2]);
    QPoint_Barycentric.resize(2);
    for (int ind = 0; ind < 2; ++ind){
        QPoint_Barycentric[ind].resize(n_q_point[ind]);
        for (int i = 0; i < n_q_point[ind]; ++i)
            QPoint_Barycentric[ind][i].resize(ind+2);
    }
    Weight.resize(3);
    for (int ind = 0; ind < 3; ++ind)
	Weight[ind].resize(n_q_point[ind]);
    // read quadrature info from q_fn
    for (int ind = 0; ind < 3; ++ind){
	std::ifstream infile;
	infile.open(q_fn[ind]);
	valuetype sum_weight = 0;
        for (int p = 0; p < n_q_point[ind]; ++p){
            for (int indt = 0; indt <= ind; ++indt)
                if (ind == 2)
                    infile >> QPoint[p][indt];
                else
                    infile >> QPoint_Barycentric[ind][p][indt];
            if (ind == 0) // modify coordinate to be barycenter one
                QPoint_Barycentric[ind][p][0] = (QPoint_Barycentric[ind][p][0] + 1) / 2;
            if (ind < 2){ // calculate the barycenter coordinate of the last point
                QPoint_Barycentric[ind][p][ind+1] = 1;
                for (int indt = 0; indt <= ind; ++indt)
                    QPoint_Barycentric[ind][p][ind+1] -= QPoint_Barycentric[ind][p][indt];
            }
            infile >> Weight[ind][p];
	    sum_weight += Weight[ind][p];
        }
        std::cerr << "Read " << ind+1 << "d quadrature info, find " << n_q_point[ind] << " pairs of points and weights, with weight summation " << sum_weight << '\n';
        infile.close();
    }
    // std::cerr << "QPoint.size() = " << QPoint.size() << '\n';
    // for (int p = 0; p < QPoint.size(); ++p)
    // 	std::cerr << "([" << QPoint[p][0] << ',' << QPoint[p][1] << ',' << QPoint[p][2] << "], " << Weight[2][p] << ") ";
    // std::cerr << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::read_interp_info(const std::string &interp_filename)
{
    // number of nonzero contribution on edges
    n_expr_edge.resize(6);;
    // index of nonzero contribution on edges
    ind_expr_edge.resize(6);
    // value of nonzero contribution on edges
    val_expr_edge.resize(6);
    // number of nonzero contribution on face
    n_expr_face.resize(6);
    // index of nonzero contribution on face
    ind_expr_face.resize(6);
    // value of nonzero contribution on face
    val_expr_face.resize(6);
    
    // read projection info from file Info_Interpolate
    std::ifstream input_interp(interp_filename);
    int M_tmp;
    input_interp >> M_tmp;
    n_dof_edge = M - 1;
    n_dof_face = (M-2) * (M-3) / 2 + M-2;
    int n_dof_edge_read = M_tmp - 1;
    int n_dof_face_read = (M_tmp-2) * (M_tmp-3) / 2 + M_tmp-2;
    
    // read info
    for (int ind_var = 0; ind_var < 6; ++ind_var){
        // initialize
        n_expr_edge[ind_var].resize(n_dof_face_read);
        ind_expr_edge[ind_var].resize(n_dof_face_read);
        val_expr_edge[ind_var].resize(n_dof_face_read);
        n_expr_face[ind_var].resize(n_dof_face_read);
        ind_expr_face[ind_var].resize(n_dof_face_read);
        val_expr_face[ind_var].resize(n_dof_face_read);
        // traverse dof, read edge interpolation info
        for (int ind_dof = 0; ind_dof < n_dof_face_read; ++ind_dof){
            n_expr_edge[ind_var][ind_dof].resize(3);
            ind_expr_edge[ind_var][ind_dof].resize(3);
            val_expr_edge[ind_var][ind_dof].resize(3);
            for (int ind_e = 0; ind_e < 3; ++ind_e){
                int cnt;
                input_interp >> cnt;
                n_expr_edge[ind_var][ind_dof][ind_e] = cnt;
                ind_expr_edge[ind_var][ind_dof][ind_e].resize(cnt);
                val_expr_edge[ind_var][ind_dof][ind_e].resize(cnt);
                for (int ind_nnz = 0; ind_nnz < cnt; ++ind_nnz){
                    input_interp >> ind_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                    input_interp >> val_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                }
            }
        }
        // traverse dof, read face interpolation info
        for (int ind_dof = 0; ind_dof < n_dof_face_read; ++ind_dof){
            int count;
            input_interp >> count;
            n_expr_face[ind_var][ind_dof] = count;
            ind_expr_face[ind_var][ind_dof].resize(count);
            val_expr_face[ind_var][ind_dof].resize(count);
            for (int ind_nnz = 0; ind_nnz < count; ++ind_nnz){
                input_interp >> ind_expr_face[ind_var][ind_dof][ind_nnz];
                input_interp >> val_expr_face[ind_var][ind_dof][ind_nnz];
            }
        }
    }
    input_interp.close();
    std::cerr << "read projection info, M_tmp = " << M_tmp
              << ", n_dof_edge_read = " << n_dof_edge_read << ", n_dof_face_read = " << n_dof_face_read
              << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::build_basis_value()
{ // calculate basis value on quadrature points
    
    // calculate the value of basis function at each quadrature point
    basis_value.resize(3);
    for (int ind = 0; ind < 3; ++ind)
        basis_value[ind].resize(n_q_point[ind]);
    // basis function value at 3-d quadrature points
    for (int p = 0; p < n_q_point[2]; ++p){
        valuetype x = QPoint[p][0], y = QPoint[p][1], z = QPoint[p][2];
        valuetype xi = 2*x/(1-y-z)-1, eta = 2*y/(1-z)-1, zeta = 2*z-1;
        basis_value[2][p].resize(n_index);
        // calculate $J_k^{-1,-1}(xi)$ for k = 0: M
        valuetype Jxi[M+1], Jeta[M+1], Jzeta[M+1];
        Jxi[0] = Jeta[0] = Jzeta[0] = 1;
        Jxi[1] = xi; // as J_1^{-1,-1}(xi) = xi
        for (int k = 1; k < M; ++k)
            Jxi[k+1] = ((xi - calc_coefficient_a(-1,-1,2,k)) * Jxi[k] - calc_coefficient_a(-1,-1,3,k) * Jxi[k-1]) / calc_coefficient_a(-1,-1,1,k);
        // traverse first component of multiindex
        for (int l1 = 0; l1 <= M; ++l1){
            int aph2 = 2 * l1 - 1; // alpha for the generalized jacobi polynomial of eta
            // calculate value of generalized jacobi polynomial Jeta
            Jeta[1] = calc_generalized_jacobi_polynomial(aph2, -1, 1, eta);
            for (int k = 1; k < M-l1; ++k)
                Jeta[k+1] = ((eta - calc_coefficient_a(aph2,-1,2,k)) * Jeta[k] - calc_coefficient_a(aph2,-1,3,k) * Jeta[k-1]) / calc_coefficient_a(aph2,-1,1,k);
            // traverse second component
            for (int l2 = 0; l2 <= M-l1; ++l2){
                int aph3 = 2 * l1 + 2 * l2 - 1;
                Jzeta[1] = calc_generalized_jacobi_polynomial(aph3, -1, 1, zeta);
                for (int k = 1; k < M-l1-l2; ++k)
                    Jzeta[k+1] = ((zeta - calc_coefficient_a(aph3,-1,2,k)) * Jzeta[k] - calc_coefficient_a(aph3,-1,3,k) * Jzeta[k-1]) / calc_coefficient_a(aph3,-1,1,k);
                // traverse third component
                for (int l3 = 0; l3 <= M-l1-l2; ++l3){
                    Multiindex<3> index_now = Unitary_Multiindex[0] * l1 + Unitary_Multiindex[1] * l2 + Unitary_Multiindex[2] * l3;
                    basis_value[2][p][correspondence.index2number(index_now)] = pow(1-y-z,l1)*Jxi[l1] * pow(1-z,l2)*Jeta[l2] * Jzeta[l3];
		    // if (fabs(basis_value[2][p][correspondence.index2number(index_now)]) > 1.0e8)
		    // 	std::cerr << "find large basis_value, Jxi[" << l1 << "] = " << Jxi[l1]
		    // 		  << ", Jeta[" << l2 << "] = " << Jeta[l2]
		    // 		  << ", Jzeta[" << l3 << "] = " << Jzeta[l3]
		    // 		  << ", a(" << aph2 << ", -1, 1, " << l2-1 << ") = " << calc_coefficient_a(aph2, -1, 1, l2-1)
		    // 		  << ", a(" << aph3 << ", -1, 1, " << l3-1 << ") = " << calc_coefficient_a(aph3, -1, 1, l3-1)
		    // 		  << '\n';
                }
            }
        }
    }
    // 2-d quadrature point
    for (int p = 0; p < n_q_point[1]; ++p){
        basis_value[1][p].resize(n_dof_geometry[2]);
        valuetype x = QPoint_Barycentric[1][p][0], y = QPoint_Barycentric[1][p][1];
        valuetype xi = 2*x/(1-y)-1, eta = 2*y-1;
        valuetype Jxi[M+1], Jeta[M+1];
        Jxi[0] = Jeta[0] = 1;
        Jxi[1] = xi;
        for (int l = 1; l < M; ++l)
            Jxi[l+1] = ((xi - calc_coefficient_a(-1,-1,2,l))*Jxi[l] - calc_coefficient_a(-1,-1,3,l)*Jxi[l-1]) / calc_coefficient_a(-1,-1,1,l);
        for (int l1 = 2; l1 <= M; ++l1){
            int aph = 2 * l1 - 1;
            Jeta[1] = calc_generalized_jacobi_polynomial(aph, -1, 1, eta);
            for (int l = 1; l < M; ++l)
                Jeta[l+1] = ((eta - calc_coefficient_a(aph,-1,2,l))*Jeta[l] - calc_coefficient_a(aph,-1,3,l)*Jeta[l-1]) / calc_coefficient_a(aph,-1,1,l);
            for (int l2 = 1; l2 <= M-l1; ++l2){
                int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2 - 1;
                basis_value[1][p][ind_index] = 2 * pow(1-y, l1) * Jxi[l1] * Jeta[l2];
            }
        }
    }
    // 1-d quadrature point
    for (int p = 0; p < n_q_point[0]; ++p){
        basis_value[0][p].resize(n_dof_geometry[1]);
        valuetype xi = QPoint_Barycentric[0][p][0] * 2 - 1;
        valuetype Jxi[M+1];
        Jxi[0] = 1;
        Jxi[1] = xi;
        for (int l = 1; l < M; ++l)
            Jxi[l+1] = ((xi - calc_coefficient_a(-1,-1,2,l))*Jxi[l] - calc_coefficient_a(-1,-1,3,l)*Jxi[l-1]) / calc_coefficient_a(-1,-1,1,l);
        for (int l = 2; l <= M; ++l)
            basis_value[0][p][l-2] = 2 * Jxi[l];
    }
    std::cerr << "calculate function value of generalized jacobi polynomial at quadrature points\n";
    // for (int ind_index = 0; ind_index < n_index; ++ind_index){
    // 	std::cerr << "ind_index = " << ind_index << ", basis_value:";
    // 	for (int p = 0; p < n_q_point[2]; ++p)
    // 	    std::cerr << ' ' << basis_value[2][p][ind_index];
    // 	std::cerr << '\n';
    // }

    // calculate the value of generalized jacobi polynomial for the interpolation of function
    basis_value_interp.resize(3); // basis value for interpolation
    for (int ind = 0; ind < 3; ++ind){
        basis_value_interp[ind].resize(n_q_point[ind]);
        for (int p = 0; p < n_q_point[ind]; ++p)
            basis_value_interp[ind][p].resize(n_dof_geometry[ind+1]);
    }
    // for the interpolation of the dof on edge
    for (int p = 0; p < n_q_point[0]; ++p){
        valuetype xp = 2 * QPoint_Barycentric[0][p][0] - 1;
        basis_value_interp[0][p][0] = 1;
        basis_value_interp[0][p][1] = calc_generalized_jacobi_polynomial(1, 1, 1, xp);
        for (int l = 1; l < M-2; ++l)
            basis_value_interp[0][p][l+1] = ((xp - calc_coefficient_a(1,1,2,l)) * basis_value_interp[0][p][l]
                                             - calc_coefficient_a(1,1,3,l) * basis_value_interp[0][p][l-1]) / calc_coefficient_a(1,1,1,l);
    }
    // for the interpolation of the dof on face
    // additional basis value, [p][l][0]: $(1-y)^l*J_{l-2}^{1,1}(\xi_p)$, [p][l][1]: $J_{l-2}^{1,1}(\eta_p)$
    basis_value_addition.resize(n_q_point[1]);
    for (int p = 0; p < n_q_point[1]; ++p){
        basis_value_addition[p].resize(n_dof_geometry[1]);
        for (int l = 0; l < n_dof_geometry[1]; ++l)
            basis_value_addition[p][l].resize(2);
    }
    for (int p = 0; p < n_q_point[1]; ++p){
        valuetype xp = QPoint_Barycentric[1][p][0], yp = QPoint_Barycentric[1][p][1];
        valuetype xi = 2*xp/(1-yp) - 1, eta = 2*yp - 1;
        valuetype Jxi[M-1], Jeta[M];
        // evaluate basis_value_interp
        Jxi[0] = Jeta[0] = 1;
        Jxi[1] = calc_generalized_jacobi_polynomial(1, 1, 1, xi);
        for (int l = 1; l < M-2; ++l) // evaluate Jxi by recursion relation
            Jxi[l+1] = ((xi - calc_coefficient_a(1,1,2,l)) * Jxi[l] - calc_coefficient_a(1,1,3,l) * Jxi[l-1]) / calc_coefficient_a(1,1,1,l);
        for (int l1 = 2; l1 <= M; ++l1){
            Jeta[1] = calc_generalized_jacobi_polynomial(2*l1-1, 1, 1, eta);
            for (int l2 = 1; l2 < M-l1; ++l2) // evaluate Jeta by recursion relation, Jeta[M-l1] is un-used
                Jeta[l2+1] = ((eta - calc_coefficient_a(2*l1-1,1,2,l2)) * Jeta[l2] - calc_coefficient_a(2*l1-1,1,3,l2) * Jeta[l2-1]) / calc_coefficient_a(2*l1-1,1,1,l2);
            for (int l2 = 1; l2 <= M-l1; ++l2){ // assign basis_value_interp, corresponds to (l1-2, l2-1)
                int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
                basis_value_interp[1][p][ind_index] = pow(1-eta, l1-2) * Jxi[l1-2] * Jeta[l2-1];
            }
        }
        // evaluate basis_value_addition, which correspond to $(1-yp)^{l-2}J_{l-2}^{1,1}(xi)$ and $J_{l-2}^{1,1}(eta)$ for l = 2: M
        for (int l = 2; l <= M; ++l)
            basis_value_addition[p][l-2][0] = pow(1-yp, l-2) * Jxi[l-2]; // as Jxi is the same
        Jeta[1] = calc_generalized_jacobi_polynomial(1, 1, 1, eta);
        for (int l = 1; l < M-2; ++l)
            Jeta[l+1] = ((eta - calc_coefficient_a(1,1,2,l)) * Jeta[l] - calc_coefficient_a(1,1,3,l) * Jeta[l-1]) / calc_coefficient_a(1,1,1,l);
        for (int l = 2; l <= M; ++l)
            basis_value_addition[p][l-2][1] = Jeta[l-2];
    }
    // for the interpolation of the interior dof
    for (int p = 0; p < n_q_point[2]; ++p){
        valuetype xp = QPoint[p][0], yp = QPoint[p][1], zp = QPoint[p][2];
        valuetype xi = 2*xp/(1-yp-zp)-1, eta = 2*yp/(1-zp)-1, zeta = 2*zp-1;
        valuetype Jxi[M-1], Jeta[M], Jzeta[M];
        Jxi[0] = Jeta[0] = Jzeta[0] = 1;
        Jxi[1] = calc_generalized_jacobi_polynomial(1, 1, 1, xi);
        for (int l = 1; l < M-2; ++l) // l1 = 2: M -> l1-2 = 0: M-2
            Jxi[l+1] = ((xi - calc_coefficient_a(1,1,2,l)) * Jxi[l] - calc_coefficient_a(1,1,3,l) * Jxi[l-1]) / calc_coefficient_a(1,1,1,l);
        for (int l1 = 2; l1 <= M; ++l1){
            Jeta[1] = calc_generalized_jacobi_polynomial(2*l1-1, 1, 1, eta);
            for (int l = 1; l < M - l1; ++l) // in fact, we consider l2 = 1: M-l1 -> l2-1 = 0: M-l1-1, so Jeta[M-l1] is un-used
                Jeta[l+1] = ((eta - calc_coefficient_a(2*l1-1,1,2,l)) * Jeta[l] - calc_coefficient_a(2*l1-1,1,3,l) * Jeta[l-1]) / calc_coefficient_a(2*l1-1,1,1,l);
            for (int l2 = 1; l2 <= M - l1; ++l2){
                int aph = 2 * l1 + 2 * l2 - 1;
                Jzeta[1] = calc_generalized_jacobi_polynomial(aph, 1, 1, zeta);
                for (int l = 1; l < M - l1 - l2; ++l) // similarly, Jzeta[M-l1-l2] is un-used
                    Jzeta[l+1] = ((zeta - calc_coefficient_a(aph,1,2,l)) * Jzeta[l] - calc_coefficient_a(aph,1,3,l) * Jzeta[l-1]) / calc_coefficient_a(aph,1,1,l);
                for (int l3 = 1; l3 <= M - l1 - l2; ++l3){
                    int ind_index = correspondence.index2number(Unitary_Multiindex[0] * (l1-2) + Unitary_Multiindex[1] * (l2-1) + Unitary_Multiindex[2] * (l3-1));
                    basis_value_interp[2][p][ind_index] = Jxi[l1-2] * Jeta[l2-1]*pow(1-eta,l1-2) * Jzeta[l3-1]*pow(1-zeta,l1+l2-3);
                }
            }
        }
    }
    std::cerr << "calculate the basis value for interpolation\n";

    // calculate gradient of basis function on quadrature points
    basis_gradient.resize(n_q_point[2], std::vector<std::vector<valuetype> > (n_index, std::vector<valuetype> (3, 0)));
    // prepare auxiliary varaible: coefficient and point value of generalize Jacobi polynomial
    valuetype coeff_D1[n_index];
    valuetype coeff_D2[n_index][2]; // p = 0, 1
    valuetype coeff_D3[n_index][4]; // (p, q) = (0, 0) (1, 0) (0, 1) (1, 1)
    for (int ind_index = 0; ind_index < n_index; ++ind_index){
	Multiindex<3> index_now = correspondence.number2index(ind_index);
	coeff_D1[ind_index] = calc_coefficient_D(0, 0, index_now);
	coeff_D2[ind_index][0] = calc_coefficient_D(3, 0, index_now);
	coeff_D2[ind_index][1] = calc_coefficient_D(3, 1, index_now);
	coeff_D3[ind_index][0] = calc_coefficient_D(5, 0, index_now);
	coeff_D3[ind_index][1] = calc_coefficient_D(5, 1, index_now);
	coeff_D3[ind_index][2] = calc_coefficient_D(5, 2, index_now);
	coeff_D3[ind_index][3] = calc_coefficient_D(5, 3, index_now);
    }
    // traverse quadrature points, calculate val_Jacobi, then assign basis_gradient[][p][]
    for (int p = 0; p < n_q_point[2]; ++p){
	// setup
	std::vector<std::vector<valuetype> > curlicue_J(3, std::vector<valuetype> (n_index));
	// assign curlicue J by calculating generalize Jacobi polynomials
	valuetype x1 = QPoint[p][0], x2 = QPoint[p][1], x3 = QPoint[p][2];
	valuetype t1 = 1-x2-x3, t2 = 1-x3;
	valuetype xi = 2*x1/t1-1, eta = 2*x2/t2-1, zeta = 2*x3-1;
	valuetype pow_t1[M+1], pow_t2[M+1];
	pow_t1[0] = (valuetype) 1; pow_t2[0] = (valuetype) 1;
	for (int l = 1; l <= M; ++l){
	    pow_t1[l] = pow_t1[l-1] * t1;
	    pow_t2[l] = pow_t2[l-1] * t2;
	}
	valuetype Jxi  [2][M+1]; // [0][]: ^{0,0};          [1][]: ^{0,-1}
	valuetype Jeta [3][M+1]; // [0][]: ^{2l1+1,-1};     [1][]: ^{2l1,0};     [2][]: ^{2l1,-1}
	valuetype Jzeta[2][M+1]; // [0][]: ^{2l1+2l2+1,-1}; [1][]: ^{2l1+2l2,0}
	Jxi[0][0] = Jxi[1][0] = (valuetype) 1;
	Jeta[0][0] = Jeta[1][0] = Jeta[2][0] = (valuetype) 1;
	Jzeta[0][0] = Jzeta[1][0] = (valuetype) 1;
	Jxi[0][1] = calc_generalized_jacobi_polynomial(0,  0, 1, xi);
	Jxi[1][1] = calc_generalized_jacobi_polynomial(0, -1, 1, xi);
	for (int l = 1; l < M; ++l){
	    Jxi[0][l+1] = ((xi - calc_coefficient_a(0, 0,2,l)) * Jxi[0][l] - calc_coefficient_a(0, 0,3,l) * Jxi[0][l-1]) / calc_coefficient_a(0, 0,1,l);
	    Jxi[1][l+1] = ((xi - calc_coefficient_a(0,-1,2,l)) * Jxi[1][l] - calc_coefficient_a(0,-1,3,l) * Jxi[1][l-1]) / calc_coefficient_a(0,-1,1,l);
	}
	for (int l1 = 0; l1 <= M; ++l1){
	    int aph2_0 = 2*l1+1, aph2_1 = 2*l1;
	    Jeta[0][1] = calc_generalized_jacobi_polynomial(aph2_0, -1, 1, eta);
	    Jeta[1][1] = calc_generalized_jacobi_polynomial(aph2_1,  0, 1, eta);
	    Jeta[2][1] = calc_generalized_jacobi_polynomial(aph2_1, -1, 1, eta);
	    for (int l = 1; l < M-l1; ++l){
		Jeta[0][l+1] = ((eta - calc_coefficient_a(aph2_0,-1,2,l)) * Jeta[0][l] - calc_coefficient_a(aph2_0,-1,3,l) * Jeta[0][l-1]) / calc_coefficient_a(aph2_0,-1,1,l);
		Jeta[1][l+1] = ((eta - calc_coefficient_a(aph2_1, 0,2,l)) * Jeta[1][l] - calc_coefficient_a(aph2_1, 0,3,l) * Jeta[1][l-1]) / calc_coefficient_a(aph2_1, 0,1,l);
		Jeta[2][l+1] = ((eta - calc_coefficient_a(aph2_1,-1,2,l)) * Jeta[2][l] - calc_coefficient_a(aph2_1,-1,3,l) * Jeta[2][l-1]) / calc_coefficient_a(aph2_1,-1,1,l);
	    }
	    for (int l2 = 0; l2 <= M-l1; ++l2){
		int aph3_0 = 2*l1+2*l2+1, aph3_1 = 2*l1+2*l2;
		Jzeta[0][1] = calc_generalized_jacobi_polynomial(aph3_0, -1, 1, zeta);
		Jzeta[1][1] = calc_generalized_jacobi_polynomial(aph3_1,  0, 1, zeta);
		for (int l = 1; l < M-l1-l2; ++l){
                    Jzeta[0][l+1] = ((zeta-calc_coefficient_a(aph3_0,-1,2,l)) * Jzeta[0][l] - calc_coefficient_a(aph3_0,-1,3,l) * Jzeta[0][l-1]) / calc_coefficient_a(aph3_0,-1,1,l);
                    Jzeta[1][l+1] = ((zeta-calc_coefficient_a(aph3_1, 0,2,l)) * Jzeta[1][l] - calc_coefficient_a(aph3_1, 0,3,l) * Jzeta[1][l-1]) / calc_coefficient_a(aph3_1, 0,1,l);
		}
		for (int l3 = 0; l3 <= M-l1-l2; ++l3){
		    Multiindex<3> index_now = Unitary_Multiindex[0]*l1 + Unitary_Multiindex[1]*l2 + Unitary_Multiindex[2]*l3;
		    int ind_index = correspondence.index2number(index_now);
		    curlicue_J[0][ind_index] = pow_t1[l1]*Jxi[0][l1] * pow_t2[l2]*Jeta[0][l2] * Jzeta[0][l3];
		    curlicue_J[1][ind_index] = pow_t1[l1]*Jxi[1][l1] * pow_t2[l2]*Jeta[1][l2] * Jzeta[0][l3];
		    curlicue_J[2][ind_index] = pow_t1[l1]*Jxi[1][l1] * pow_t2[l2]*Jeta[2][l2] * Jzeta[1][l3];
		}
	    }
	}
	// assign basis_gradient[p][][]
	for (int ind_index = 0; ind_index < n_index; ++ind_index){
	    Multiindex<3> index_tmp = correspondence.number2index(ind_index);
	    // 1st component
	    index_tmp.index[0]--;
	    if (0 <= index_tmp.index[0])
		basis_gradient[p][ind_index][0] += coeff_D1[ind_index]    * curlicue_J[0][correspondence.index2number(index_tmp)];
	    index_tmp.index[0]++;
	    // 2nd component
	    // p = 0
	    index_tmp.index[1]--;
	    if (0 <= index_tmp.index[1])
		basis_gradient[p][ind_index][1] += coeff_D2[ind_index][0] * curlicue_J[1][correspondence.index2number(index_tmp)];
	    index_tmp.index[1]++;
	    // p = 1;
	    index_tmp.index[0]--;
	    if (0 <= index_tmp.index[0])
		basis_gradient[p][ind_index][1] += coeff_D2[ind_index][1] * curlicue_J[1][correspondence.index2number(index_tmp)];
	    index_tmp.index[0]++;
	    // 3rd component
            // p = 0, q = 0
	    index_tmp.index[2]--;
	    if (0 <= index_tmp.index[2])
		basis_gradient[p][ind_index][2] += coeff_D3[ind_index][0] * curlicue_J[2][correspondence.index2number(index_tmp)];
	    index_tmp.index[2]++;
	    // p = 1, q = 0
	    index_tmp.index[0]--; index_tmp.index[1]++; index_tmp.index[2]--;
	    if (0 <= index_tmp.index[0] && 0 <= index_tmp.index[2])
		basis_gradient[p][ind_index][2] += coeff_D3[ind_index][1] * curlicue_J[2][correspondence.index2number(index_tmp)];
	    index_tmp.index[0]++; index_tmp.index[1]--; index_tmp.index[2]++;
	    // p = 0, q = 1
	    index_tmp.index[1]--;
	    if (0 <= index_tmp.index[1])
		basis_gradient[p][ind_index][2] += coeff_D3[ind_index][2] * curlicue_J[2][correspondence.index2number(index_tmp)];
	    index_tmp.index[1]++;
	    // p = 1, q = 1
	    index_tmp.index[0]--;
	    if (0 <= index_tmp.index[0])
		basis_gradient[p][ind_index][2] += coeff_D3[ind_index][3] * curlicue_J[2][correspondence.index2number(index_tmp)];
	}
    }
    std::cerr << "calculate the basis gradient\n";
}

TEMPLATE_TSEM
void THIS_TSEM::build_local_transform()
{
    // construct tranform_local, weight_transform_local: generalize jacobi polynomial -> basis function
    n_transform_local.resize(n_index);
    transform_local.resize(n_index);
    weight_transform_local.resize(n_index);
    // vertex modes
    n_transform_local[0] = 4; n_transform_local[1] = 2; n_transform_local[2] = 3; n_transform_local[3] = 4;
    for (int ind = 0; ind <= 3; ++ind){
        transform_local[ind].resize(n_transform_local[ind]);
        weight_transform_local[ind].resize(n_transform_local[ind]);
    }
    transform_local[0][0] = 0; weight_transform_local[0][0] = (valuetype) 0.125; // $J_{0,0,0} -> \varphi_{0,0,0}$
    transform_local[0][1] = 1; weight_transform_local[0][1] = (valuetype) 0.125; // $J_{0,0,0} -> \varphi_{1,0,0}$
    transform_local[0][2] = 2; weight_transform_local[0][2] = (valuetype) 0.25;  // $J_{0,0,0} -> \varphi_{0,1,0}$
    transform_local[0][3] = 3; weight_transform_local[0][3] = (valuetype) 0.5;   // $J_{0,0,0} -> \varphi_{0,0,1}$
    transform_local[1][0] = 0; weight_transform_local[1][0] = (valuetype) -0.5;  // $J_{1,0,0} -> \varphi_{0,0,0}$
    transform_local[1][1] = 1; weight_transform_local[1][1] = (valuetype) 0.5;   // $J_{1,0,0} -> \varphi_{1,0,0}$
    transform_local[2][0] = 0; weight_transform_local[2][0] = (valuetype) -0.25; // $J_{0,1,0} -> \varphi_{0,0,0}$
    transform_local[2][1] = 1; weight_transform_local[2][1] = (valuetype) -0.25; // $J_{0,1,0} -> \varphi_{1,0,0}$
    transform_local[2][2] = 2; weight_transform_local[2][2] = (valuetype) 0.5;   // $J_{1,1,0} -> \varphi_{0,1,0}$
    transform_local[3][0] = 0; weight_transform_local[3][0] = (valuetype) -0.125;// $J_{0,0,1} -> \varphi_{0,0,0}$
    transform_local[3][1] = 1; weight_transform_local[3][1] = (valuetype) -0.125;// $J_{0,0,1} -> \varphi_{1,0,0}$
    transform_local[3][2] = 2; weight_transform_local[3][2] = (valuetype) -0.25; // $J_{0,0,1} -> \varphi_{0,1,0}$
    transform_local[3][3] = 3; weight_transform_local[3][3] = (valuetype) 0.5;   // $J_{0,0,1} -> \varphi_{0,0,1}$
    // edge modes
    for (int l = 2; l <= M; ++l){
        // $J_{l,0,0}$ -> $\varphi_{l,0,0}$ 
        Multiindex<3> index_tmp3 = Unitary_Multiindex[0] * l;
        int ind_tmp3 = correspondence.index2number(index_tmp3);
        n_transform_local[ind_tmp3] = 1;
        transform_local[ind_tmp3].resize(n_transform_local[ind_tmp3]);
        weight_transform_local[ind_tmp3].resize(n_transform_local[ind_tmp3]);
        transform_local[ind_tmp3][0] = ind_tmp3; weight_transform_local[ind_tmp3][0] = (valuetype) 2;
        // $J_{0,l,0}$, $J_{1,l-1,0}$ -> $\varphi_{0,l,0}$, $\varphi_{1,l-1,0}$
        Multiindex<3> index_tmp1 = Unitary_Multiindex[1] * l;
        Multiindex<3> index_tmp2 = Unitary_Multiindex[0] + Unitary_Multiindex[1] * (l-1);
        int ind_tmp1 = correspondence.index2number(index_tmp1);
        int ind_tmp2 = correspondence.index2number(index_tmp2);
        n_transform_local[ind_tmp1] = 2;
        n_transform_local[ind_tmp2] = 2;
        transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
        transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
        weight_transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
        weight_transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
        transform_local[ind_tmp1][0] = ind_tmp1; weight_transform_local[ind_tmp1][0] = (valuetype) 1;         // $J_{0,l,0}$   -> $\varphi_{0,l,0}$
        transform_local[ind_tmp1][1] = ind_tmp2; weight_transform_local[ind_tmp1][1] = (valuetype) 1;         // $J_{0,l,0}$   -> $\varphi_{1,l-1,0}$
        transform_local[ind_tmp2][0] = ind_tmp1; weight_transform_local[ind_tmp2][0] = ((valuetype) (l-1))/l; // $J_{1,l-1,0}$ -> $\varphi_{0,l,0}$
        transform_local[ind_tmp2][1] = ind_tmp2; weight_transform_local[ind_tmp2][1] =((valuetype) -(l-1))/l; // $J_{1,l-1,0}$ -> $\varphi_{1,l-1,0}$
        // $J_{0,0,l}$, $J_{1,0,l-1}$, $J_{0,1,l-1}$ -> $\varphi_{0,0,l}$, $varphi_{1,0,l-1}$, $varphi_{0,1,l-1}$
        index_tmp1 = Unitary_Multiindex[2] * l;
        index_tmp2 = Unitary_Multiindex[0] + Unitary_Multiindex[2] * (l-1);
        index_tmp3 = Unitary_Multiindex[1] + Unitary_Multiindex[2] * (l-1);
        ind_tmp1 = correspondence.index2number(index_tmp1);
        ind_tmp2 = correspondence.index2number(index_tmp2);
        ind_tmp3 = correspondence.index2number(index_tmp3);
        n_transform_local[ind_tmp1] = 3;
        n_transform_local[ind_tmp2] = 2;
        n_transform_local[ind_tmp3] = 3;
        transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
        transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
        transform_local[ind_tmp3].resize(n_transform_local[ind_tmp3]);
        weight_transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
        weight_transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
        weight_transform_local[ind_tmp3].resize(n_transform_local[ind_tmp3]);
        transform_local[ind_tmp1][0] = ind_tmp1; weight_transform_local[ind_tmp1][0] = (valuetype) 0.5;           // $J_{0,0,l}$   -> $\varphi_{0,0,l}$
        transform_local[ind_tmp1][1] = ind_tmp2; weight_transform_local[ind_tmp1][1] = (valuetype) 0.5;           // $J_{0,0,l}$   -> $\varphi_{1,0,l-1}$
        transform_local[ind_tmp1][2] = ind_tmp3; weight_transform_local[ind_tmp1][2] = (valuetype) 1;             // $J_{0,0,l}$   -> $\varphi_{0,1,l-1}$
        transform_local[ind_tmp2][0] = ind_tmp1; weight_transform_local[ind_tmp2][0] = ((valuetype) (l-1))/l;     // $J_{1,0,l-1}$ -> $\varphi_{0,0,l}$
        transform_local[ind_tmp2][1] = ind_tmp2; weight_transform_local[ind_tmp2][1] =((valuetype) -(l-1))/l;     // $J_{1,0,l-1}$ -> $\varphi_{1,0,l-1}$
        transform_local[ind_tmp3][0] = ind_tmp1; weight_transform_local[ind_tmp3][0] = ((valuetype) (l-1))/(2*l); // $J_{0,1,l-1}$ -> $\varphi_{0,0,l}$
        transform_local[ind_tmp3][1] = ind_tmp2; weight_transform_local[ind_tmp3][1] = ((valuetype) (l-1))/(2*l); // $J_{0,1,l-1}$ -> $\varphi_{1,0,l-1}$
        transform_local[ind_tmp3][2] = ind_tmp3; weight_transform_local[ind_tmp3][2] =((valuetype) -(l-1))/l;     // $J_{0,1,l-1}$ -> $\varphi_{0,1,l-1}$
    }
    // face modes
    for (int l1 = 2; l1 <= M; ++l1)
        for (int l2 = 1; l2 <= M-l1; ++l2){
            // $J_{0,l1,l2}$, $J_{1,l1-1,l2}$ -> $\varphi_{0,l1,l2}$, $varphi_{1,l1-1,l2}$
            Multiindex<3> index_tmp1 = Unitary_Multiindex[1] * l1 + Unitary_Multiindex[2] * l2;
            Multiindex<3> index_tmp2 = Unitary_Multiindex[0] + Unitary_Multiindex[1] * (l1-1) + Unitary_Multiindex[2] * l2;
            int ind_tmp1 = correspondence.index2number(index_tmp1);
            int ind_tmp2 = correspondence.index2number(index_tmp2);
            n_transform_local[ind_tmp1] = (valuetype) 2;
            n_transform_local[ind_tmp2] = (valuetype) 2;
            transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
            transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
            weight_transform_local[ind_tmp1].resize(n_transform_local[ind_tmp1]);
            weight_transform_local[ind_tmp2].resize(n_transform_local[ind_tmp2]);
            transform_local[ind_tmp1][0] = ind_tmp1; weight_transform_local[ind_tmp1][0] = (valuetype) 1;           // $J_{0,l1,l2}$   -> $\varphi_{0,l1,l2}$
            transform_local[ind_tmp1][1] = ind_tmp2; weight_transform_local[ind_tmp1][1] = (valuetype) 1;           // $J_{0,l1,l2}$   -> $\varphi_{1,l1-1,l2}$
            transform_local[ind_tmp2][0] = ind_tmp1; weight_transform_local[ind_tmp2][0] = ((valuetype) (l1-1))/l1; // $J_{1,l1-1,l2}$ -> $\varphi_{0,l1,l2}$
            transform_local[ind_tmp2][1] = ind_tmp2; weight_transform_local[ind_tmp2][1] =((valuetype) -(l1-1))/l1; // $J_{1,l1-1,l2}$ -> $\varphi_{1,l1-1,l2}$
            // $J_{l1,0,l2}$ -> $\varphi_{l1,0,l2}$ and $J_{l1,l2,0}$ -> $\varphi_{l1,l2,0}$
            for (int ind = 1; ind <= 2; ++ind){
                Multiindex<3> index_tmp = Unitary_Multiindex[0] * l1 + Unitary_Multiindex[ind] * l2;
                int ind_tmp = correspondence.index2number(index_tmp);
                n_transform_local[ind_tmp] = (valuetype) 1;
                transform_local[ind_tmp].resize(n_transform_local[ind_tmp]);
                weight_transform_local[ind_tmp].resize(n_transform_local[ind_tmp]);
                transform_local[ind_tmp][0] = ind_tmp; weight_transform_local[ind_tmp][0] = (valuetype) 2;
            }
        }
    // interior modes
    for (int l1 = 2; l1 <= M; ++l1)
        for (int l2 = 1; l2 <= M-l1; ++l2)
            for (int l3 = 1; l3 <= M-l1-l2; ++l3){
                Multiindex<3> index_tmp = Unitary_Multiindex[0] * l1 + Unitary_Multiindex[1] * l2 + Unitary_Multiindex[2] * l3;
                int ind_tmp = correspondence.index2number(index_tmp);
                n_transform_local[ind_tmp] = 1;
                transform_local[ind_tmp].resize(n_transform_local[ind_tmp]);
                weight_transform_local[ind_tmp].resize(n_transform_local[ind_tmp]);
                transform_local[ind_tmp][0] = ind_tmp; weight_transform_local[ind_tmp][0] = (valuetype) 1;
            }

    // for (int ind_index = 0; ind_index < n_index; ++ind_index){
    // 	std::cerr << "ind_index = " << ind_index << ", n_transform_local = " << n_transform_local[ind_index]
    // 		  << ", weight_transform_local:";
    // 	for (int i = 0; i < n_transform_local[ind_index]; ++i)
    // 	    std::cerr << ' ' << weight_transform_local[ind_index][i];
    // 	std::cerr << '\n';
    // }
    // generate actual basis function value at local fem element, by transform_local and weigth_transform_local
    basis_value_actual.resize(n_q_point[2], std::vector<valuetype> (n_index, 0));
    // traverse 3-d quadrature point, use transform_local and weight_transform_local, calculate basis_value_actual
    for (int p = 0; p < n_q_point[2]; ++p)
        for (int i = 0; i < n_index; ++i)
            for (int j = 0; j < n_transform_local[i]; ++j)
                basis_value_actual[p][transform_local[i][j]] += weight_transform_local[i][j] * basis_value[2][p][i];
    std::cerr << "assigned basis_value_actual\n";

    basis_gradient_actual.resize(n_q_point[2], std::vector<std::vector<valuetype> > (n_index, std::vector<valuetype> (3, 0)));
    for (int p = 0; p < n_q_point[2]; ++p)
	for (int ind_index = 0; ind_index < n_index; ++ind_index)
	    for (int ind_tl = 0; ind_tl < n_transform_local[ind_index]; ++ind_tl)
		for (int ind = 0; ind < 3; ++ind)
		    basis_gradient_actual[p][transform_local[ind_index][ind_tl]][ind] += weight_transform_local[ind_index][ind_tl] * basis_gradient[p][ind_index][ind];
    std::cerr << "assigned basis_gradient_actual\n";
}

TEMPLATE_TSEM
void THIS_TSEM::build_geometry_info(RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space)
{
    // setup, basic variables
    point_ref_mesh.resize(4); // [i = 0:3], barycenter of i dimensional geometry
    for (int ind = 0; ind <= 3; ++ind)
        n_geometry[ind] = mesh.n_geometry(ind);
    n_element = n_geometry[3];
    std::cerr << "read number of geometries, n_geometry[0:3] = [" << n_geometry[0] << ", " << n_geometry[1] << ", " << n_geometry[2] << ", " << n_geometry[3] << "]\n";
    std::cerr << "n_element = " << n_element << '\n';
    n_geometry_total = 0;
    for (int ind = 0; ind <= 3; ++ind)
        n_geometry_total += n_geometry[ind];
    for (int ind = 0; ind <= 3; ++ind) // number of dof on ind dimensional geometry
        n_dof_geometry[ind] = calc_binomial_coefficient(M-1, ind);
    for (int ind = 0; ind <= 3; ++ind)
        std::cerr << "n_dof_geometry[" << ind << "] = " << n_dof_geometry[ind] << ",\t";
    std::cerr << '\n';
    n_dof_total = 0; // number of total degree of freedom
    for (int ind = 0; ind <= 3; ++ind)
        n_dof_total += n_geometry[ind] * n_dof_geometry[ind]; // sum up all dof on each dimensional geometry
    std::cerr << "n_dof_total = " << n_dof_total << '\n';
    // assign point_ref
    for (int ind = 0; ind <= 3; ++ind){
        point_ref_mesh[ind].resize(mesh.n_geometry(ind));
        for (int i = 0; i < mesh.n_geometry(ind); ++i){
            AFEPack::Point<3> point_tmp = mesh.point(mesh.geometry(ind, i).vertex(0));
            for (int indt = 1; indt <= ind; ++indt)
                point_tmp += mesh.point(mesh.geometry(ind, i).vertex(indt));
            point_tmp /= (ind + 1);
            point_ref_mesh[ind][i] = point_tmp;
        }
    }


    // record order of nodes on each dimensional geometry
    // number_node[ind+1=1:2][order+1=1:ind+1][]: the node number of ind+1-dimensional geometry
    number_node.resize(2);
    number_edge.resize(n_geometry[2]);
    for (int ind = 1; ind <= 2; ++ind){
        number_node[ind-1].resize(n_geometry[ind]);
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo)
            number_node[ind-1][ind_geo].resize(ind+1, -1);
    }
    for (int ind_geo = 0; ind_geo < n_geometry[2]; ++ind_geo)
        number_edge[ind_geo].resize(3, -1);
    // flag that whether order of start/end point on global face is the same as that on global edge
    flag_sameorder_edgeonface.resize(n_geometry[2]);
    for (int ind_face = 0; ind_face < n_geometry[2]; ++ind_face)
	flag_sameorder_edgeonface[ind_face].resize(3, true);
    // type of projection for edge and face on each element
    type_projection.resize(n_element);
    type_projection_inv.resize(n_element); // inverse projection type, for face on each element
    // traverse finite element space, record number_node
    // the order of node number determine others, i.e., correspond to nodes on reference as P0, P1, P2, P3
    std::vector<bool> flag_assign_edge(n_geometry[1], false);
    std::vector<bool> flag_assign_face(n_geometry[2], false);
    
    index_geometry_onelement.resize(n_element);
    int index_start_point[] = {0, 0, 0, 1, 1, 2};
    int index_end_point[]   = {1, 2, 3, 2, 3, 3};
    int index_face_point[] = {1, 2, 3,
                              0, 2, 3,
                              0, 1, 3,
                              0, 1, 2};
    int index_face_edge[] = {5, 4, 3,
                             5, 2, 1,
                             4, 2, 0,
                             3, 1, 0};
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // initialize
        index_geometry_onelement[ind_ele].resize(3);
        index_geometry_onelement[ind_ele][0].resize(4); // index for nodes
        index_geometry_onelement[ind_ele][1].resize(6); // index for edges
        index_geometry_onelement[ind_ele][2].resize(4); // index for faces
        type_projection[ind_ele].resize(2);
        type_projection[ind_ele][0].resize(6, -1); // 6 edge, 0-the same order; 1-inverse order
        type_projection[ind_ele][1].resize(4, -1); // 4 face, 0-the same order; 1-5-corresponding variation
        type_projection_inv[ind_ele].resize(4, -1); // 4 face, 0-the same order; 1-5-corresponding variation
        // read node number of this element, correspond to P0, P1, P2, P3
        // the node number determine the order of edge and face on this element
        for (int ind_node = 0; ind_node < 4; ++ind_node)
            index_geometry_onelement[ind_ele][0][ind_node] = mesh.geometry(3, ind_ele).vertex(ind_node);

        // read edge number of this element, correspond to E01, E02, E03, E12, E13, E23
        int ind_face_this;
        // consider face in front of P3, which is P0P1P2
        ind_face_this = mesh.geometry(3, ind_ele).boundary(3);
        for (int ind_e = 0; ind_e < 3; ++ind_e){
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][0]) // in front of P0
                index_geometry_onelement[ind_ele][1][3] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E12
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][1]) // in front of P1
                index_geometry_onelement[ind_ele][1][1] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E02
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][2]) // in front of P2
                index_geometry_onelement[ind_ele][1][0] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E01
        }
        // consider face in front of P2, which is P0P1P3
        ind_face_this = mesh.geometry(3, ind_ele).boundary(2);
        for (int ind_e = 0; ind_e < 3; ++ind_e){
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][0]) // in front of P0
                index_geometry_onelement[ind_ele][1][4] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E13
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][1]) // in front of P1
                index_geometry_onelement[ind_ele][1][2] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E03
        }
        // consider face in front of P1, which is P0P2P3
        ind_face_this = mesh.geometry(3, ind_ele).boundary(1);
        for (int ind_e = 0; ind_e < 3; ++ind_e)
            if (mesh.geometry(2, ind_face_this).vertex(ind_e) == index_geometry_onelement[ind_ele][0][0]) // in front of P0
                index_geometry_onelement[ind_ele][1][5] = mesh.geometry(2, ind_face_this).boundary(ind_e); // correspond to E23

        // read face number of this element, correspond to F0=P1P2P3, F1=P0P2P3, F2=P0P1P3, F3=P0P1P2
        for (int ind_face = 0; ind_face < 4; ++ind_face)
            index_geometry_onelement[ind_ele][2][ind_face] = mesh.geometry(3, ind_ele).boundary(ind_face);
       
        // traverse 6 edges of this element
        for (int ind_edge = 0; ind_edge < 6; ++ind_edge){
            int &index_e = index_geometry_onelement[ind_ele][1][ind_edge];
            if (!flag_assign_edge[index_e]){ // if this edge hasn't been traversed
                // record start point and end point
                number_node[0][index_e][0] = index_geometry_onelement[ind_ele][0][index_start_point[ind_edge]];
                number_node[0][index_e][1] = index_geometry_onelement[ind_ele][0][index_end_point[ind_edge]];
                // record projection type
                type_projection[ind_ele][0][ind_edge] = 0;
                // update flag
                flag_assign_edge[index_e] = true;
            }
            else{
                // record projection type
                if (number_node[0][index_e][0] == index_geometry_onelement[ind_ele][0][index_start_point[ind_edge]]) // the same start point
                    type_projection[ind_ele][0][ind_edge] = 0;
                else
                    type_projection[ind_ele][0][ind_edge] = 1;
            }
        }
        
        // traveres 4 faces of this element
        for (int ind_face = 0; ind_face < 4; ++ind_face){
            int &index_f = index_geometry_onelement[ind_ele][2][ind_face];
            if (!flag_assign_face[index_f]){ // if this face hasn't been traversed
                // record order of nodes
                number_node[1][index_f][0] = index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 0]];
                number_node[1][index_f][1] = index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 1]];
                number_node[1][index_f][2] = index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 2]];
                number_edge[index_f][0] = index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 0]];
                number_edge[index_f][1] = index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 1]];
                number_edge[index_f][2] = index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 2]];
                // record whether order of endpoints of edge the same as those on global edge, denote face as 012
                if (number_node[0][index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 0]]][0]
                    != index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 1]]) // startpoint of 12 != 1
                    flag_sameorder_edgeonface[index_f][0] = false;
                if (number_node[0][index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 1]]][0]
                    != index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 0]]) // startpoint of 02 != 0
                    flag_sameorder_edgeonface[index_f][1] = false;
                if (number_node[0][index_geometry_onelement[ind_ele][1][index_face_edge[3*ind_face + 2]]][0]
                    != index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + 0]]) // startpoint of 01 != 0
                    flag_sameorder_edgeonface[index_f][2] = false;
                // record projection type
                type_projection[ind_ele][1][ind_face] = 0;
                type_projection_inv[ind_ele][ind_face] = 0;
                // update flag
                flag_assign_face[index_f] = true;
            }
            else{
                // record projection type: 012, 021, 102, 120, 201, 210
                std::vector<int> order_node_on_ref(3, -1); // order of nodes on reference element
                for (int ind_n = 0; ind_n < 3; ++ind_n)
                    for (int ind_n_r = 0; ind_n_r < 3; ++ind_n_r)
                        if (index_geometry_onelement[ind_ele][0][index_face_point[3*ind_face + ind_n]] == number_node[1][index_f][ind_n_r])
                            order_node_on_ref[ind_n] = ind_n_r;
                int flag_number = order_node_on_ref[0] * 100 + order_node_on_ref[1] * 10 + order_node_on_ref[2];
                switch (flag_number){
                case 12: // 0 1 2, 0 1 2
                    type_projection[ind_ele][1][ind_face] = 0;
                    type_projection_inv[ind_ele][ind_face] = 0;
                    break;
                case 21: // 0 2 1, 0 2 1
                    type_projection[ind_ele][1][ind_face] = 1;
                    type_projection_inv[ind_ele][ind_face] = 1;
                    break;
                case 102: // 1 0 2, 1 0 2
                    type_projection[ind_ele][1][ind_face] = 2;
                    type_projection_inv[ind_ele][ind_face] = 2;
                    break;
                case 120: // 1 2 0, 2 0 1
                    type_projection[ind_ele][1][ind_face] = 3;
                    type_projection_inv[ind_ele][ind_face] = 4;
                    break;
                case 201: // 2 0 1, 1 2 0
                    type_projection[ind_ele][1][ind_face] = 4;
                    type_projection_inv[ind_ele][ind_face] = 3;
                    break;
                case 210: // 2 1 0, 2 1 0
                    type_projection[ind_ele][1][ind_face] = 5;
                    type_projection_inv[ind_ele][ind_face] = 5;
                    break;
                default:
                    std::cerr << "error: assign type of projection!\n";
                }
            }
        }
    }
    std::cerr << "assign type_projection\n";

    // calculate volume for each element
    val_volume.resize(fem_space.n_element());
    AFEPack::Point<3> point_tmp;
    point_tmp[0] = point_tmp[1] = point_tmp[2] = 1.0/3.0;
    for (unsigned int ind_ele = 0; ind_ele < fem_space.n_element(); ++ind_ele){
    	valuetype volume = fem_space.element(ind_ele).templateElement().volume();
    	valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed
    	val_volume[ind_ele] = fabs(volume * jacobian);
    }
    // val_volume.resize(mesh.n_geometry(3));
    // for (unsigned int ind_ele = 0; ind_ele < mesh.n_geometry(3); ++ind_ele){
    // 	AFEPack::Point<3> &p0 = mesh.point(mesh.geometry(3, ind_ele).vertex(0));
    // 	AFEPack::Point<3> &p1 = mesh.point(mesh.geometry(3, ind_ele).vertex(1));
    // 	AFEPack::Point<3> &p2 = mesh.point(mesh.geometry(3, ind_ele).vertex(2));
    // 	AFEPack::Point<3> &p3 = mesh.point(mesh.geometry(3, ind_ele).vertex(3));
    // 	val_volume[ind_ele] = calc_volume_tetrahedron(p0, p1, p2, p3);
    // }
}

TEMPLATE_TSEM
void THIS_TSEM::build_global_transform(RegularMesh<3> &mesh, FEMSpace<double, 3> &fem_space)
{
    // assign the weight_location, geometry_dimension and geometry_order of dof, which determine the order of these geometry in discretized matrix
    /* {weight_local, geometry_dimension, geometry_order}:
     *    weight_local:       a number for sorting, read from fem_space, determine the order of all dimensional geomtry
     *    geometry_dimension: the dimension of geometry corresponds to this set
     *    geometry_order:     the order of geometry in the same dimensional ones corresponds to this set
     */
    weight_location.resize(n_geometry_total); // the weight_location of 0: 3 dimensional geometry in turns
    geometry_dimension.resize(n_geometry_total);
    geometry_order.resize(n_geometry_total); // [i = 0:n_geometry_total-1] the order of i-th entry in weight_location, whose dimension is geometry_dimension[i]
    // correspondence between fem dof/element and geometry
    // transform_femdof2geometry: index from fem dof to 0 & 1 dimensional geometry, for 1 dimensional geometry, its index plus mesh.n_geometry(0)
    int n_dof = fem_space.n_dof();
    transform_femdof2geometry.resize(n_dof, -1);
    // location_geometry: location of all geometry (0:3 dimensional) according to increasing order of weight_location
    location_geometry.resize(n_geometry_total);
    // location_actualdof: start index of geometry in actual discretized matrix
    location_actualdof.resize(n_geometry_total);
    /* geometry in finite element space                       -> total index of all geometries     -> position in actual discretized system
     * transform_femdof2geometry (0 & 1 dimensional geometry) -> total index = 0: n_geometry_total -> location_actualdof[location_geometry[total index]]
     * transform_femele2geometry (2 & 3 dimensional geometry)
     */
    // assign geometry_dimension and geometry_order
    for (int ind = 0; ind <= 3; ++ind){
        int index_start = 0;
        for (int indt = 0; indt < ind; ++indt)
            index_start += n_geometry[indt];
        for (int i = 0; i < n_geometry[ind]; ++i){
            geometry_dimension[index_start + i] = ind;
            geometry_order[index_start + i] = i;
        }
    }
    // construct correspondence between fem dof and geometry (I): find weight_location for 0 & 1 dimensional geometry according to the order in fem_space
    for (int i = 0; i < n_dof; ++i)
        for (int ind = 0; ind <= 1 && transform_femdof2geometry[i] < 0; ++ind) // as all order in fem_space is natural number, so >= 0
            for (int j = 0; j < n_geometry[ind]; ++j)
                if (distance(point_ref_mesh[ind][j], fem_space.dofInfo(i).interp_point) < tol_zero){
                    transform_femdof2geometry[i] = ind * n_geometry[0] + j; // if ind == 1, add n_geometry[0], total index of point or edge
                    weight_location[transform_femdof2geometry[i]] = i;
                    break;
                }
    // calculate weight_location for 2 & 3 dimensional geometry
    for (int ind = 2; ind <= 3; ++ind){
        int index_start = n_geometry[0] + n_geometry[1] + (ind-2) * n_geometry[2]; // if ind == 3, add n_geometry[2]
        for (int i = 0; i < n_geometry[ind]; ++i){
            valuetype weight_tmp = 0.0;
            for (int indt = 0; indt <= ind; ++indt)
                weight_tmp += weight_location[mesh.geometry(ind, i).vertex(indt)];
            weight_tmp /= ind + 1;
            weight_location[index_start + i] = weight_tmp;
        }
    }
    // sort according to weight_location
    for (int i = 1; i < n_geometry_total; ++i)
        for (int j = i; j > 0; --j)
            if (weight_location[j] < weight_location[j-1]){
                valuetype tmp = weight_location[j];
                weight_location[j] = weight_location[j-1];
                weight_location[j-1] = tmp;
                int tmpi = geometry_dimension[j];
                geometry_dimension[j] = geometry_dimension[j-1];
                geometry_dimension[j-1] = tmpi;
                tmpi = geometry_order[j];
                geometry_order[j] = geometry_order[j-1];
                geometry_order[j-1] = tmpi;
            }
    for (int i = 0; i < n_geometry_total; ++i){ // assign location for geometry_order[i]-th geometry_dimension[i] dimensional geometry
        int index = geometry_order[i]; // recover the index of geometry in whole order
        for (int ind = 0; ind < geometry_dimension[i]; ++ind)
            index += n_geometry[ind];
        location_geometry[index] = i;
    }
    // insert dof into each location_geometry
    location_actualdof[0] = 0;
    for (int i = 1; i < n_geometry_total; ++i)
        location_actualdof[i] = location_actualdof[i-1] + n_dof_geometry[geometry_dimension[i-1]];
    std::cerr << "assign location_actualdof, location_geometry\n";
    
    
    // traverse element, construct transformation between local and global
    // initialize
    // assign conversion between lexicography order and the order in jia2022
    conversion.resize(2);
    conversion[0].resize(n_dof_face);
    conversion[1].resize(n_dof_geometry[3]);
    for (unsigned int l1 = 2; l1 <= M; ++l1)
	for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
	    unsigned int ind_lex = (l1+l2+1-3) * (l1+l2-3) / 2 + l2-1; // location of (l1-2, l2-1) in lexicography order
	    unsigned int ind_jia = (M+1)*(l1-2) - (l1-1-2)*(l1-2)/2 + l2-1;
	    conversion[0][ind_lex] = ind_jia;
	    conversion[0][ind_lex] = ind_lex;
	}
    for (unsigned int ind_index = 0; ind_index < n_dof_geometry[3]; ++ind_index){
	Multiindex<3> index_now = correspondence.number2index(ind_index);
	unsigned int l1 = index_now.index[0], l2 = index_now.index[1], l3 = index_now.index[2];
	unsigned int ind_jia = ((l1-1)*l1*(2*l1-1) + l1*(M+1)*(M+2)*6 - (2*M+3)*(l1-1)*l1*3) / 12
	    + (M-l1+1)*l2 - (l2-1)*l2/2
	    + l3;
	conversion[1][ind_index] = ind_jia;
	conversion[1][ind_index] = ind_index;
    }
    // expression of local coefficient by linear summation of global ones
    transform_n_global2local.resize(n_element);
    transform_ind_global2local.resize(n_element);
    transform_val_global2local.resize(n_element);
    // expression of global coefficient on each-dimensional geometry by local ones, equivalent to column compress of transformation matrix
    transform_n_local2global.resize(n_element);
    transform_ind_local2global.resize(n_element);
    transform_val_local2global.resize(n_element);
    // assign transform_local2global and transform_global2local
    std::vector<int> index_edgedof_local(n_dof_edge * 6); // index of multiindex corresponds to edge dof on local element
    for (int l = 2; l <= M; ++l){
        index_edgedof_local[n_dof_edge*0 + l-2] = correspondence.index2number(Unitary_Multiindex[0]*l); // (l, 0, 0)
        index_edgedof_local[n_dof_edge*1 + l-2] = correspondence.index2number(Unitary_Multiindex[1]*l); // (0, l, 0)
        index_edgedof_local[n_dof_edge*2 + l-2] = correspondence.index2number(Unitary_Multiindex[2]*l); // (0, 0, l)
        index_edgedof_local[n_dof_edge*4 + l-2] = correspondence.index2number(Unitary_Multiindex[0] + Unitary_Multiindex[2]*(l-1)); // (1,   0, l-1)
        index_edgedof_local[n_dof_edge*3 + l-2] = correspondence.index2number(Unitary_Multiindex[0] + Unitary_Multiindex[1]*(l-1)); // (1, l-1,   0)
        index_edgedof_local[n_dof_edge*5 + l-2] = correspondence.index2number(Unitary_Multiindex[1] + Unitary_Multiindex[2]*(l-1)); // (0,   1, l-1)
    }
    std::vector<int> index_facedof_local(n_dof_face * 4); // index of multiindex corresponds to face dof on local element
    for (int l1 = 2; l1 <= M; ++l1)
        for (int l2 = 1; l2 <= M-l1; ++l2){
            int ind_index = (l1+l2-3)*(l1+l2-2)/2 + l2-1;
            index_facedof_local[n_dof_face*0 + ind_index] =
                correspondence.index2number(Unitary_Multiindex[0] + Unitary_Multiindex[1]*(l1-1) + Unitary_Multiindex[2]*l2); // (1, l1-1, l2)
            index_facedof_local[n_dof_face*1 + ind_index] =
                correspondence.index2number(Unitary_Multiindex[1]*l1 + Unitary_Multiindex[2]*l2); // ( 0, l1, l2)
            index_facedof_local[n_dof_face*2 + ind_index] =
                correspondence.index2number(Unitary_Multiindex[0]*l1 + Unitary_Multiindex[2]*l2); // (l1,  0, l2)
            index_facedof_local[n_dof_face*3 + ind_index] =
                correspondence.index2number(Unitary_Multiindex[0]*l1 + Unitary_Multiindex[1]*l2); // (l1, l2,  0)
        }
    int index_edge_local[] = {5, 4, 3, // edge number for each face on local element
			      5, 2, 1,
			      4, 2, 0,
			      3, 1, 0};
    int order_edge_global[] = {0, 1, 2, // order of edge on global face in each variation
                               0, 2, 1,
                               1, 0, 2,
                               1, 2, 0,
                               2, 0, 1,
                               2, 1, 0};
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // initialize
        transform_n_global2local[ind_ele].resize(n_index);
        transform_ind_global2local[ind_ele].resize(n_index);
        transform_val_global2local[ind_ele].resize(n_index);
        transform_n_local2global[ind_ele].resize(n_index, 0);
        transform_ind_local2global[ind_ele].resize(n_index);
        transform_val_local2global[ind_ele].resize(n_index);
        
        // construct transformation node dof
        // P0, P1, P2, P3 excatly correspond first 4 multiindex (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        for (int ind_node = 0; ind_node < 4; ++ind_node){ // traverse node: P0, P1, P2, P3
            int ind_index = ind_node;
            int ind_geo = index_geometry_onelement[ind_ele][0][ind_node];
            int ind_dof_global = location_actualdof[location_geometry[ind_geo]];
            int n_dep = 1; // number of dependency
            transform_n_global2local[ind_ele][ind_index] = n_dep;
            transform_ind_global2local[ind_ele][ind_index].resize(n_dep);
            transform_val_global2local[ind_ele][ind_index].resize(n_dep);
            transform_ind_global2local[ind_ele][ind_index][0] = ind_dof_global;
            transform_val_global2local[ind_ele][ind_index][0] = 1;
        }
        
        // construct transformation for edge dof
        for (int ind_edge = 0; ind_edge < 6; ++ind_edge){ // traverse edge: E01, E02, E03, E12, E13, E23
            int ind_geo = index_geometry_onelement[ind_ele][1][ind_edge];
            int ind_dof_start = location_actualdof[location_geometry[ind_geo + n_geometry[0]]];
            for (int l = 2; l <= M; ++l){
                int ind_index = index_edgedof_local[n_dof_edge*ind_edge + l-2];
                int ind_dof_global = ind_dof_start + l-2;
                int n_dep = 1;
                transform_n_global2local[ind_ele][ind_index] = n_dep;
                transform_ind_global2local[ind_ele][ind_index].resize(n_dep);
                transform_val_global2local[ind_ele][ind_index].resize(n_dep);
                transform_ind_global2local[ind_ele][ind_index][0] = ind_dof_global;
                transform_val_global2local[ind_ele][ind_index][0] = 1;
                if (type_projection[ind_ele][0][ind_edge] == 1 && l%2 == 1)
                    transform_val_global2local[ind_ele][ind_index][0] = -1;
            }
        }
        
        // construct transformation for face dof
        for (int ind_face = 0; ind_face < 4; ++ind_face){
            int ind_geo = index_geometry_onelement[ind_ele][2][ind_face];
            int ind_facedof_start = location_actualdof[location_geometry[ind_geo + n_geometry[0] + n_geometry[1]]];
            for (int ind_dof = 0; ind_dof < n_dof_geometry[2]; ++ind_dof){
                int ind_index = index_facedof_local[n_dof_face*ind_face + ind_dof];
                int ind_var = type_projection_inv[ind_ele][ind_face];
                int n_dep = 0;
                for (int ind_e = 0; ind_e < 3; ++ind_e)
                    n_dep += n_expr_edge[ind_var][ind_dof][ind_e];
                n_dep += n_expr_face[ind_var][ind_dof];
                transform_n_global2local[ind_ele][ind_index] = n_dep;
                transform_ind_global2local[ind_ele][ind_index].resize(n_dep);
                transform_val_global2local[ind_ele][ind_index].resize(n_dep);
                // assign edge contribution
                int pos_start = 0;
                for (int ind_e = 0; ind_e < 3; ++ind_e){
                    int ind_edge = number_edge[index_geometry_onelement[ind_ele][2][ind_face]][ind_e];
                    int ind_edgedof_start = location_actualdof[location_geometry[ind_edge + n_geometry[0]]];
                    for (int ind_nnz = 0; ind_nnz < n_expr_edge[ind_var][ind_dof][ind_e]; ++ind_nnz){
                        transform_ind_global2local[ind_ele][ind_index][pos_start + ind_nnz]
                            = ind_edgedof_start + ind_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                        transform_val_global2local[ind_ele][ind_index][pos_start + ind_nnz]
                            = val_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                        if (!flag_sameorder_edgeonface[index_geometry_onelement[ind_ele][2][ind_face]][ind_e]
                            && ind_expr_edge[ind_var][ind_dof][ind_e][ind_nnz] % 2 == 1)
                            transform_val_global2local[ind_ele][ind_index][pos_start + ind_nnz] *= -1;
                    }
                    pos_start += n_expr_edge[ind_var][ind_dof][ind_e];
                }
                // assign face contribution
                for (int ind_nnz = 0; ind_nnz < n_expr_face[ind_var][ind_dof]; ++ind_nnz){
                    transform_ind_global2local[ind_ele][ind_index][pos_start + ind_nnz]
                        = ind_facedof_start + conversion[0][ind_expr_face[ind_var][ind_dof][ind_nnz]];
                    transform_val_global2local[ind_ele][ind_index][pos_start + ind_nnz]
                        = val_expr_face[ind_var][ind_dof][ind_nnz];
                }
                if (pos_start + n_expr_face[ind_var][ind_dof] != n_dep)
                    std::cerr << "error: assign transform_global2local for face dof\n";
            }
        }
        
        // construct transformation for interior dof
        int ind_interiordof_start = location_actualdof[location_geometry[ind_ele + n_geometry[0] + n_geometry[1] + n_geometry[2]]];
        for (int ind_dof = 0; ind_dof < n_dof_geometry[3]; ++ind_dof){
            Multiindex<3> index_now = correspondence.number2index(ind_dof);
            index_now = index_now + Unitary_Multiindex[0]*2 + Unitary_Multiindex[1] + Unitary_Multiindex[2];
            int ind_index = correspondence.index2number(index_now);
            int n_dep = 1;
            transform_n_global2local[ind_ele][ind_index] = n_dep;
            transform_ind_global2local[ind_ele][ind_index].resize(n_dep);
            transform_val_global2local[ind_ele][ind_index].resize(n_dep);
            transform_ind_global2local[ind_ele][ind_index][0] = ind_interiordof_start + conversion[1][ind_dof];
            transform_val_global2local[ind_ele][ind_index][0] = 1;
        }

        // traverse all dimensional geometries, count contribution for transform_local2global
        for (int ind_node = 0; ind_node < 4; ++ind_node) // ind_node -> ind_index -> 1-1 global vertex coefficient
            transform_n_local2global[ind_ele][ind_node]++;
        for (int ind_edge = 0; ind_edge < 6; ++ind_edge) // ind_edge, ind_dof -> ind_index -> 1-1 global edge coefficient
            for (int ind_dof = 0; ind_dof < n_dof_edge; ++ind_dof)
                transform_n_local2global[ind_ele][index_edgedof_local[n_dof_edge*ind_edge + ind_dof]]++;
        for (int ind_face = 0; ind_face < 4; ++ind_face){
            int& ind_var = type_projection[ind_ele][1][ind_face];
            for (int ind_dof = 0; ind_dof < n_dof_face; ++ind_dof){
                // count edge contribution
                for (int ind_e = 0; ind_e < 3; ++ind_e){ // ind_dof, ind_e -> edge index -> corresponding multiindex
                    int& edge_index_local = index_edge_local[3*ind_face + ind_e]; // local index of this edge, between 0 and 5
                    for (int ind_nnz = 0; ind_nnz < n_expr_edge[ind_var][ind_dof][ind_e]; ++ind_nnz){
                        int& edgedof_index = ind_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                        int& multiindex_index = index_edgedof_local[n_dof_edge*edge_index_local + edgedof_index];
                        transform_n_local2global[ind_ele][multiindex_index]++;
                    }
                }
                // count face contribution
                for (int ind_nnz = 0; ind_nnz < n_expr_face[ind_var][ind_dof]; ++ind_nnz){
                    int& facedof_index = ind_expr_face[ind_var][ind_dof][ind_nnz];
                    int& multiindex_index = index_facedof_local[n_dof_face*ind_face + facedof_index];
                    transform_n_local2global[ind_ele][multiindex_index]++;
                }
            }
        }
        for (int ind_interiordof = 0; ind_interiordof < n_dof_geometry[3]; ++ind_interiordof){ // index of interior dof -> corresponding multiindex 
            Multiindex<3> index_tmp = correspondence.number2index(ind_interiordof); //                                     -> 1-1 global interior coefficient
            index_tmp = index_tmp + Unitary_Multiindex[0]*2 + Unitary_Multiindex[1] + Unitary_Multiindex[2];
            int multiindex_index = correspondence.index2number(index_tmp);
            transform_n_local2global[ind_ele][multiindex_index]++;
        }

        for (int ind_index = 0; ind_index < n_index; ++ind_index){
            int& n_dep = transform_n_local2global[ind_ele][ind_index];
            transform_ind_local2global[ind_ele][ind_index].resize(n_dep);
            transform_val_local2global[ind_ele][ind_index].resize(n_dep);
        }
        
        // traverse all dimensional geometries again, assign transform_local2global
        std::vector<int> pointer(n_index, 0);
        for (int ind_node = 0; ind_node < 4; ++ind_node){
            int& multiindex_index = ind_node;
            int& ind_geo = index_geometry_onelement[ind_ele][0][ind_node];
            int& ind_dof_global = location_actualdof[location_geometry[ind_geo]];
            transform_ind_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = ind_dof_global;
            transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = 1;
            pointer[multiindex_index]++;
        }
        for (int ind_edge = 0; ind_edge < 6; ++ind_edge){
            int& ind_geo = index_geometry_onelement[ind_ele][1][ind_edge];
            int& ind_startdof_global = location_actualdof[location_geometry[ind_geo + n_geometry[0]]];
            for (int ind_dof = 0; ind_dof < n_dof_edge; ++ind_dof){
                int& multiindex_index = index_edgedof_local[n_dof_edge*ind_edge + ind_dof];
                transform_ind_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = ind_startdof_global + ind_dof;
                transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = 1;
                if (type_projection[ind_ele][0][ind_edge] == 1 && ind_dof % 2 == 1)
                    transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = -1;
                pointer[multiindex_index]++;
            }
        }
        for (int ind_face = 0; ind_face < 4; ++ind_face){
            int& ind_var = type_projection[ind_ele][1][ind_face];
            int& ind_geo = index_geometry_onelement[ind_ele][2][ind_face];
            int& ind_startdof_global = location_actualdof[location_geometry[ind_geo + n_geometry[0] + n_geometry[1]]];
            for (int ind_dof = 0; ind_dof < n_dof_face; ++ind_dof){
                // int& multiindex_index = index_facedof_local[n_dof_face*ind_face + ind_dof];
                // add edge contribution
                for (int ind_e = 0; ind_e < 3; ++ind_e){
                    int& edge_index_local = index_edge_local[3*ind_face + ind_e]; // local index of this edge, between 0 and 5
                    for (int ind_nnz = 0; ind_nnz < n_expr_edge[ind_var][ind_dof][ind_e]; ++ind_nnz){
                        int& edgedof_index = ind_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                        int& multiindex_index = index_edgedof_local[n_dof_edge*edge_index_local + edgedof_index];
                        transform_ind_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = ind_startdof_global + conversion[0][ind_dof];
                        transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = val_expr_edge[ind_var][ind_dof][ind_e][ind_nnz];
                        pointer[multiindex_index]++;
                    }
                }
                // add face contribution
                for (int ind_nnz = 0; ind_nnz < n_expr_face[ind_var][ind_dof]; ++ind_nnz){
                    int& facedof_index = ind_expr_face[ind_var][ind_dof][ind_nnz];
                    int& multiindex_index = index_facedof_local[n_dof_face*ind_face + facedof_index];
                    transform_ind_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = ind_startdof_global + conversion[0][ind_dof];
                    transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = val_expr_face[ind_var][ind_dof][ind_nnz];
                    pointer[multiindex_index]++;
                }
            }
        }
        int& ind_startdof_interior_global = location_actualdof[location_geometry[ind_ele + n_geometry[0] + n_geometry[1] + n_geometry[2]]];
        for (int ind_interiordof = 0; ind_interiordof < n_dof_geometry[3]; ++ind_interiordof){
            Multiindex<3> index_tmp = correspondence.number2index(ind_interiordof);
            index_tmp = index_tmp + Unitary_Multiindex[0]*2 + Unitary_Multiindex[1] + Unitary_Multiindex[2];
            int multiindex_index = correspondence.index2number(index_tmp);
            transform_ind_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = ind_startdof_interior_global + conversion[1][ind_interiordof];
            transform_val_local2global[ind_ele][multiindex_index][pointer[multiindex_index]] = 1;
            pointer[multiindex_index]++;
        }

        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            if (pointer[ind_index] != transform_n_local2global[ind_ele][ind_index])
                std::cerr << "error: assign transform_n_local2global\n";
    }
    std::cerr << "assign transform_global2local and transform_local2global\n";
    // for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
    // 	std::cerr << "ind_ele = " << ind_ele << '\n';
    // 	for (int ind_index = 0; ind_index < n_index; ++ind_index){
    // 	    std::cerr << "\tind_index = " << ind_index << ", transform_n_global2local = " << transform_n_global2local[ind_ele][ind_index]
    // 		      << ", (index, weight) =";
    // 	    for (int i = 0; i < transform_n_global2local[ind_ele][ind_index]; ++i)
    // 		std::cerr << " (" << transform_ind_global2local[ind_ele][ind_index][i] << ',' << transform_val_global2local[ind_ele][ind_index][i] << ')';
    // 	    std::cerr << '\n';
    // 	}
    // }

    // given correct transform_global2local, test transform_local2global
    Vector<valuetype> u_g(n_dof_total); // g for global
    for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
        u_g(ind_dof) = rand() * 1.0 / RAND_MAX;
    Vector<valuetype> u_l(n_index); // l for local
    for (int ind_index = 0; ind_index < n_index; ++ind_index)
        u_l(ind_index) = rand() * 1.0 / RAND_MAX;
    valuetype dif_max_global_local = -1;
    valuetype dif_max_local_global = -1;
    int n_geometry_local[4] = {4, 6, 4, 1};
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // initialize global u
        Vector<valuetype> u_g_t(n_dof_total);
        for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
            u_g_t(ind_dof) = 0;
        // assign global u by transform_local2global
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_local2global[ind_ele][ind_index]; ++ind_nnz)
                u_g_t(transform_ind_local2global[ind_ele][ind_index][ind_nnz]) += u_l(ind_index) * transform_val_local2global[ind_ele][ind_index][ind_nnz];
        // initialize local u
        Vector<valuetype> u_l_t(n_index);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            u_l_t(ind_index) = 0;
        // assign tmp local u by transform_global2local
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                u_l_t(ind_index) += u_g_t(transform_ind_global2local[ind_ele][ind_index][ind_nnz]) * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        u_l_t -= u_l;
        if (dif_max_global_local < u_l_t.linfty_norm()) dif_max_global_local = u_l_t.linfty_norm();

        // test global -> local -> global
        std::vector<bool> flag_involved(n_dof_total, false);
        for (int ind = 0; ind < 3; ++ind)
            for (int ind_geo_local = 0; ind_geo_local < n_geometry_local[ind]; ++ind_geo_local){
                int ind_dof_start = index_geometry_onelement[ind_ele][ind][ind_geo_local];
                for (int indt = 0; indt < ind; ++indt)
                    ind_dof_start += n_geometry[indt];
                int& ind_dof_start_global = location_actualdof[location_geometry[ind_dof_start]];
                for (int ind_dof = 0; ind_dof < n_dof_geometry[ind]; ++ind_dof)
                    flag_involved[ind_dof_start_global + ind_dof] = true;
        }
        for (int ind_dof = 0; ind_dof < n_dof_geometry[3]; ++ind_dof)
            flag_involved[location_actualdof[location_geometry[ind_ele + n_geometry[0] + n_geometry[1] + n_geometry[2]]] + ind_dof] = true;
        Vector<valuetype> v_g(n_dof_total);
        for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
            if (flag_involved[ind_dof]) v_g(ind_dof) = u_g(ind_dof);
            else v_g(ind_dof) = 0;
        Vector<valuetype> v_l(n_index);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            v_l(ind_index) = 0;
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                v_l(ind_index) += v_g(transform_ind_global2local[ind_ele][ind_index][ind_nnz]) * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        Vector<valuetype> v_g_t(n_dof_total);
        for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
            v_g_t(ind_dof) = 0;
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_local2global[ind_ele][ind_index]; ++ind_nnz)
                v_g_t(transform_ind_local2global[ind_ele][ind_index][ind_nnz]) += v_l(ind_index) * transform_val_local2global[ind_ele][ind_index][ind_nnz];
        v_g_t -= v_g;
        if (dif_max_local_global < v_g_t.linfty_norm()) dif_max_local_global = v_g_t.linfty_norm();
    }
    std::cerr << "test transform_global2local and transform_local2global with rand vector,\n\tdif_max_global_local = " << dif_max_global_local << ", dif_max_local_global = " << dif_max_local_global << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::build_stiff_matrix_init()
{ // prepare coefficients and sparse matrix for assignment of stiff matrix

    // construct sparsity pattern for stiff matrix element
    int n_index_stiff = correspondence.n_index_end(M-1); // as the order of polynomial for stiff matrix is M - 1
    // std::cerr << "stiff matrix element has size " << n_index_stiff << '\n';
    std::vector<unsigned int> n_nnz_per_row(n_index_stiff, 5 * 5 * 5); // the largest one, h^{(3,2)}
    std::vector<SparsityPattern> sp_stiff_matrix_element(6);
    for (int i = 0; i < 6; ++i)
        sp_stiff_matrix_element[i].reinit(n_index_stiff, n_index_stiff, n_nnz_per_row);
    for (int row = 0; row < n_index_stiff; ++row){
        Multiindex<3> index_now = correspondence.number2index(row);
        for (int d1 = -2; d1 <= 2; ++d1)
            for (int d2 = -2; d2 <= 2; ++d2)
                for (int d3 = -2; d3 <= 2; ++d3){
                    Multiindex<3> index_tmp = index_now + Unitary_Multiindex[0] * d1 + Unitary_Multiindex[1] * (d2-d1) + Unitary_Multiindex[2] * (d3-d2);
		    // std::cerr << "index_tmp = [" << index_tmp.index[0] << ", " << index_tmp.index[1] << ", " << index_tmp.index[2] << "]\n";
                    if (!(Zero_Multiindex <= index_tmp) || index_tmp.sum() >= M) continue;
                    int col = correspondence.index2number(index_tmp);
		    // std::cerr << "index_tmp = [" << index_tmp.index[0] << ", " << index_tmp.index[1] << ", " << index_tmp.index[2] <<  "]\n";
		    // if (col >= 20) std::cerr << "find col = " << col << " exceeds size, index_tmp = [" << index_tmp.index[0] << ", " << index_tmp.index[1] << ", " << index_tmp.index[2] << "], sum = " << index_tmp.sum() << "\n";
                    // the sparsity pattern of stiff matrix element (1, 1), which corresponds to multiindex (2, 0, 0)
                    if (d1 == 0 && abs(d2) <= 1)
                        sp_stiff_matrix_element[0].add(row, col);
                    // the sparsity pattern of stiff matrix element (2, 1), which corresponds to multiindex (1, 1, 0)
                    if (abs(d1) <= 1 && abs(d2) <= 1)
                        sp_stiff_matrix_element[1].add(row, col);
                    // the sparsity pattern of stiff matrix element (3, 1), which corresponds to multiindex (1, 0, 1)
                    if (abs(d1) <= 1)
                        sp_stiff_matrix_element[2].add(row, col);
                    // the sparsity pattern of stiff matrix element (2, 2), which corresponds to multiindex (0, 2, 0)
                    if (abs(d1) <= 1 && abs(d2) <= 1)
                        sp_stiff_matrix_element[3].add(row, col);
                    // the sparsity pattern of stiff matrix element (3, 2), which corresponds to multiindex (0, 1, 1)
                    sp_stiff_matrix_element[4].add(row, col);
                    // the sparsity pattern of stiff matrix element (3, 3), which corresponds to multiindex (0, 0, 2)
                    if (abs(d1) <= 1)
                        sp_stiff_matrix_element[5].add(row, col);
                }
    }
    for (int i = 0; i < 6; ++i)
        sp_stiff_matrix_element[i].compress();
    
    // construct sparse matrix of stiff matrix element
    std::vector<SparseMatrix<valuetype> > stiff_matrix_element(6);
    for (int i = 0; i < 6; ++i)
        stiff_matrix_element[i].reinit(sp_stiff_matrix_element[i]);
    valuetype C1[6]; // coefficient C1
    valuetype C2[6]; // coefficient C2 for each choice of d1
    valuetype C3[6]; // coefficient C3 for each choice of d1+d2a = d2
    for (int row = 0; row < n_index_stiff; ++row){
        Multiindex<3> index_row = correspondence.number2index(row);
        int l1 = index_row.index[0], l2 = index_row.index[1], l3 = index_row.index[2];
        for (int d1 = -2; d1 <= 2; ++d1){
            // initialize
            for (int i = 0; i < 6; ++i)
                C1[i] = (valuetype) 0;
            // continue if multiindex has negative component
            int k1 = l1 + d1;
            if (k1 < 0) continue;
            // corresponds to (1, 1), or multiindex (2, 0, 0)
            if (d1 == 0) C1[0] = ((valuetype) 2) / (2*l1 + 1); // d1 = 0
            if (l1 == 0){
                if (0 <= d1 && d1 <= 1){
                    // corresponds to (2, 1), or multiindex (1, 1, 0)
                    C1[1] = 2 - 3 * d1; // d1 = 0: 1
                    // corresponds to (3, 1), or multiindex (1, 0, 1)
                    C1[2] = 2 - 3 * d1; // d1 = 0: 1
                    // corresponds to (2, 2), or multiindex (0, 2, 0)
                    C1[3] = 2 - d1; // d1 = 0: 1
                    // corresponds to (3, 3), or multiindex (0, 0, 2)
                    C1[5] = 2 - d1; // d1 = 0: 1
                }
                // corresponds to (3, 2), or mulltindex (0, 1, 1)
                switch (d1){ // d1 = 0: 2
                case 0: C1[4] = (valuetype) 2; break;
                case 2: C1[4] = ((valuetype) -1)/3; break;
                default: C1[4] = (valuetype) 0;
                }
            }
            else{
                if (abs(d1) <= 1){
                    // corresponds to (2, 1), or multiindex (1, 1, 0)
                    C1[1] = ((valuetype) 1) / (2*l1) * (calc_delta(d1, 0) - calc_coefficient_a(-1, 0, d1+2, l1+d1)); // d1 = -1: 1
                    // corresponds to (3, 1), or multiindex (1, 0, 1)
                    C1[2] = ((valuetype) 1) / (2*l1) * (calc_delta(d1, 0) - calc_coefficient_a(-1, 0, d1+2, l1+d1)); // d1 = -1: 1
                    // corresponds to (2, 2), or multiindex (0, 2, 0)
                    C1[3] = ((valuetype) 1) / (2*l1) * (calc_delta(d1, 0) + calc_coefficient_a(0, -1, d1+2, l1+d1)); // d1 = -1: 1
                    // correspondes to (3, 3), or multiindex (0, 0, 2)
                    C1[5] = ((valuetype) 1) / (2*l1) * (calc_delta(d1, 0) + calc_coefficient_a(0, -1, d1+2, l1+d1)); // d1 = -1: 1
                }
                // corresponds to (3, 2), or multiindex (0, 1, 1)
                if (l1 == 1)
                    switch (k1){
                    case 1: C1[4] = ((valuetype) 2)/3; break;
                    case 3: C1[4] = ((valuetype) -2)/15; break;
                    default: C1[4] = (valuetype) 0;
                    }
                else{
                    valuetype sum = 0;
                    for (int i1 = ((d1-1 > -1) ? d1-1 : -1);
                         i1 <= ((d1+1 < 1) ? d1+1 : 1); ++i1)
                        sum += (calc_delta(i1, 0) + calc_coefficient_a(-1, -1, i1+2, l1+i1)) * (calc_delta(d1, i1) - calc_coefficient_a(-1, -1, d1-i1+2, l1+d1));
                    C1[4] = ((valuetype) (l1-1)) / ((2*l1-1)*2*l1) * sum; // d1 = -2: 2
                }
            }
            for (int d2 = -2; d2 <= 2; ++d2){
                int d2a = d2 - d1; // actual d2
                // continue if multiindex index has negative component
                int k2 = l2 + d2a;
                if (k2 < 0) continue;
                int aph2 = 2 * l1;
                for (int i = 0; i < 6; ++i) // initialize
                    C2[i] = (valuetype) 0;
                if (abs(d1) <= 1 && abs(d2) <= 1){
                    // corresponds to (2, 1), or multiindex (1, 1, 0)
                    C2[1] = ((valuetype) pow(2, aph2+1)) / (aph2+2*l2+1);
                    switch (d1){ // d2 = -1: 1
                    case -1: C2[1] *= calc_coefficient_c(aph2-2, 0, d2a+1, l2+d2a);
                        break;
                    case  0: C2[1] *= ((valuetype) calc_delta(d2a, 0)) - calc_coefficient_a(aph2, 0, d2a+2, l2+d2a);
                        break;
                    case  1: C2[1] *= 4 * calc_coefficient_g(aph2, 0, d2a+3, l2+d2a);
		    }
                    // corresponds to (2, 2), or multiindex (0, 2, 0)
                    C2[3] = C2[1];
		}
                if (l2 == 0){
                    if (d1 == 0 && 0 <= d2a && d2a <= 1)
                        // corresponds to (1, 1), or multiindex (2, 0, 0)
                        C2[0] = ((valuetype) pow(2, aph2+2)) / (aph2+2+d2a); // d1 = 0, d2a = 0: 1
                    if (abs(d1) <= 1 && 0 <= d2a && d2a <= 3){
                        // corresponds to (3, 1), or multiindex (1, 0, 1)
                        C2[2] = (valuetype) pow(2, aph2+d1+2);
                        valuetype numerator = 1;
                        if (d2a != 0){
                            for (int i = 0; i < d2a-1; ++i)
                                numerator *= (valuetype) d1-1 + i;
                            numerator *= (valuetype) aph2 + 2*d1 + d2a;
                        }
                        valuetype denominator = 1;
                        for (int i = 0; i < d2a+1; ++i)
                            denominator *= (valuetype) aph2+d1+2 + i;
                        C2[2] *= numerator / denominator; // d2a = 0: 3
                        // corresponds to (3, 3), or multiindex (0, 0, 2)
                        C2[5] = C2[2];// d2a = 0: 3
                    }
                }
                else{
                    if (d1 == 0 && abs(d2) <= 1)
                        // corresponds to (1, 1), or multiindex (2, 0, 0)
                        C2[0] = ((valuetype) pow(2, aph2+1) * (aph2+l2+1)) / (l2 * (aph2+2*l2+1))
                            * (((valuetype) calc_delta(d2a, 0)) + calc_coefficient_a(aph2+1, -1, d2a+2, l2+d2a)); // d1 = 0, d2 = -1: 1
                    if (abs(d1) <= 1){
                        // corresponds to (3, 1), or multiindex (1, 0, 1)
                        C2[2] = ((valuetype) pow(2, aph2) * (aph2+l2)) / (l2 * (aph2+2*l2));
                        valuetype sum = 0;
                        for (int i2 = ((d2a-1 + d1 > -1) ? d2a-1 + d1 : -1);
                             i2 <= ((d2a+1 + d1 < 1) ? d2a+1 + d1 : 1); ++i2){
                            // if (l2 + i2 < 0) continue;
                            valuetype tmp = ((valuetype) calc_delta(i2, 0)) + calc_coefficient_a(aph2, -1, i2+2, l2+i2);
                            switch (d1){
                            case -1: sum += tmp * calc_coefficient_c(aph2-2, -1, d2a-i2+1, l2+d2a);
                                break;
                            case  0: sum += tmp * (((valuetype) calc_delta(d2a, i2)) - calc_coefficient_a(aph2, -1, d2a-i2+2, l2+d2a));
                                break;
                            case  1: sum += tmp * 4 * calc_coefficient_g(aph2, -1, d2a-i2+3, l2+d2a);
                            }
                        }
                        C2[2] *= sum; // d1 = -1: 1, d2 = -2: 2
                        // corresponds to (3, 3), or multiindex (0, 0, 2)
                        C2[5] = C2[2]; // d1 = -1: 1, d2 = -2: 2
                    }
                }
                // corresponds to (3, 2), or multiindex (0, 1, 1)
                if (l1 == 0 && l2 == 0)
                    switch (d1){
                    case 0:
                        switch (d2a){
                        case 0: C2[4] = (valuetype) 2; break;
                        case 1: C2[4] = ((valuetype) -4)/3; break;
                        case 2: C2[4] = ((valuetype) 1)/3; break;
                        default: C2[4] = (valuetype) 0;
                        }
                        break;
                    case 1:
                        switch (d2a){
                        case 0: C2[4] = ((valuetype) 8)/3; break;
                        case 1: C2[4] = ((valuetype) -2)/3; break;
                        default: C2[4] = (valuetype) 0;
                        }
                        break;
                    case 2:
                        if (d2a == 0) C2[4] = (valuetype) 4;
                        break;
                    default: C2[4] = (valuetype) 0;
                    }
                else{
                    C2[4] = ((valuetype) pow(2, aph2)) / (aph2+2*l2);
                    valuetype sum = (valuetype) 0;
                    if (d1 == -2)
                        for (int i2 = ((d2a-2 > 0) ? d2a-2 : 0);
                             i2 <= ((d2a < 2) ? d2a : 2); ++i2){
                            sum += calc_coefficient_c(aph2-3, 0, i2+1, l2+i2) * calc_coefficient_c(aph2-5, 0, d2a-i2+1, l2+d2a);
                        }
                    else if (d1 == 2)
                        for (int i2 = ((d2a > -2) ? d2a : -2);
                             i2 <= ((d2a+2 < 0) ? d2a+2 : 0); ++i2){
                            sum += 16 * calc_coefficient_g(aph2-1, 0, i2+3, l2+i2) * calc_coefficient_g(aph2+1, 0, d2a-i2+3, l2+d2a);
                        }
                    else
                        for (int i2 = ((d2a-1 + d1 > -1) ? d2a-1 + d1 : -1);
                             i2 <= ((d2a+1 + d1 < 1) ? d2a+1 + d1 : 1); ++i2){
                            valuetype tmp = 1-abs(i2) - calc_coefficient_a(aph2-1, 0, i2+2, l2+i2);
                            switch (d1){
                            case -1: sum += tmp * calc_coefficient_c(aph2-3, 0, d2a-i2+1, l2+d2a);
                                break;
                            case  0: sum += tmp * (((valuetype) calc_delta(d2a, i2)) - calc_coefficient_a(aph2-1, 0, d2a-i2+2, l2+d2a));
                                break;
                            case  1: sum += tmp * 4 * calc_coefficient_g(aph2-1, 0, d2a-i2+3, l2+d2a);
                            }
                        }
                    C2[4] *= sum; // d1 = -2: 2, d2 = -2: 2
                }
                // special case for (3, 2), or multiindex (0, 1, 1)
                if (l1 == 0 && k1 == 2 && l2 >= 1 && k2 == 0)
                    switch (l2){
                    case 1: C2[4] = ((valuetype) -16)/5;  break;
                    case 2: C2[4] = ((valuetype) 8)/5;    break;
                    case 3: C2[4] = ((valuetype) -16)/35; break;
                    case 4: C2[4] = ((valuetype) 2)/35;   break;
                    }
                for (int d3 = -2; d3 <= 2; ++d3){
                    int d = d1 + d2a; // d1 is exactly actual d1
                    int d3a = d3 - d; // actual d3
                    int k3 = l3 + d3a;
                    // continue if multiindex has negative component or has sum larger than M
                    Multiindex<3> index_col = Unitary_Multiindex[0] * k1 + Unitary_Multiindex[1] * k2 + Unitary_Multiindex[2] * k3;
                    if (!(Zero_Multiindex <= index_col) || index_col.sum() >= M) continue;
                    int aph3 = 2 * l1 + 2 * l2;
                    // corresponds to (3, 2), or multiindex (0, 1, 1)
                    for (int i = 0; i < 6; ++i) // initialize
                        C3[i] = (valuetype) 0;
                    C3[4] = pow(2, aph3+1) / (aph3+2*l3+1.0);
                    valuetype sum = 0;
                    if (d == -2)
                        for (int i3 = ((d3a-2 > 0) ? d3a-2 : 0);
                             i3 <= ((d3a < 2) ? d3a : 2); ++i3){
                                sum += calc_coefficient_c(aph3-2, 0, i3+1, l3+i3) * calc_coefficient_c(aph3-4, 0, d3a-i3+1, l3+d3a);
                        }
                    if (d == 2)
                        for (int i3 = ((d3a > -2) ? d3a : -2);
                             i3 <= ((d3a+2 < 0) ? d3a+2 : 0); ++i3){
                                sum += 16 * calc_coefficient_g(aph3, 0, i3+3, l3+i3) * calc_coefficient_g(aph3+2, 0, d3a-i3+3, l3+d3a);
                        }
                    if (-1 <= d && d <= 1)
                        for (int i3 = ((d3a-1 + d > -1) ? d3a-1 + d : -1);
                             i3 <= ((d3a+1 + d < 1) ? d3a+1 + d : 1); ++i3){
                            valuetype tmp = calc_delta(i3, 0) - calc_coefficient_a(aph3, 0, i3+2, l3+i3);
                            switch (d){
                            case -1: sum += tmp * calc_coefficient_c(aph3-2, 0, d3a-i3+1, l3+d3a);
                                break;
                            case  0: sum += tmp * (((valuetype) calc_delta(d3a, i3)) - calc_coefficient_a(aph3, 0, d3a-i3+2, l3+d3a));
                                break;
                            case  1: sum += tmp * 4 * calc_coefficient_g(aph3, 0, d3a-i3+3, l3+d3a);
                            }
                        }
                    C3[4] *= sum; // d1 = -2: 2, d2 = -2: 2, d3 = -2: 2
                    if (abs(d1) <= 1){
                        // corresponds to (3, 1), or multiindex (1, 0, 1)
                        C3[2] = C3[4]; // d1 = -1: 1, d2 = -2: 2, d3 = -2: 2
                        // corresponds to (3, 3), or multiindex (0, 0, 2)
                        C3[5] = C3[4]; // d1 = -1: 1, d2 = -2: 2, d3 = -2: 2
                    }
                    if (l3 == 0){
                        if (abs(d1) <= 1 && abs(d2) <= 1 && 0 <= d3a && d3a <= 3){
                            // corresponds to (2, 1), or multiindex (1, 1, 0)
                            C3[1] = (valuetype) pow(2, aph3+d+3);
                            valuetype numerator = 1;
                            if (d3a != 0){
                                for (int i = 0; i < d3a-1; ++i)
                                    numerator *= (valuetype) d-1 + i;
                                numerator *= (valuetype) aph3 + 2*d + 1 + d3a;
                            }
                            valuetype denominator = (valuetype) 1;
                            for (int i = 0; i < d3a+1; ++i)
                                denominator *= (valuetype) aph3+d+3 + i;
                            C3[1] *= numerator / denominator; // d1 = -1: 1, d2 = -1: 1, d3a = 0: 3
                            if (d1 == 0)
                                // corresponds to (1, 1), or multiindex (2, 0, 0)
                                C3[0] = C3[1]; // d1 = 0, d2 = -1: 1, d3a = 0: 3
                            // corresponds to (2, 2), or multiindex (0, 2, 0)
                            C3[3] = C3[1]; // d1 = -1: 1, d2 = -1: 1, d3a = 0: 3
                        }
                    }
                    else{
                        if (abs(d1) <= 1 && abs(d2) <= 1){
                            // corresponds to (2, 1), or multiindex (1, 1, 0)
                            C3[1] = ((valuetype) pow(2, aph3+1) * (aph3+l3+1)) / (l3 * (aph3+2*l3+1));
                            sum = (valuetype) 0;
                            for (int i3 = ((d3a-1 + d > -1) ? d3a-1 + d : -1);
                                 i3 <= ((d3a+1 + d < 1) ? d3a+1 + d : 1); ++i3){
                                valuetype tmp = ((valuetype) calc_delta(i3, 0)) + calc_coefficient_a(aph3+1, -1, i3+2, l3+i3);
                                switch (d){
                                case -1: sum += tmp * calc_coefficient_c(aph3-1, -1, d3a-i3+1, l3+d3a);
                                    break;
                                case  0: sum += tmp * (((valuetype) calc_delta(d3a, i3)) - calc_coefficient_a(aph3+1, -1, d3a-i3+2, l3+d3a));
                                    break;
                                case  1: sum += tmp * 4 * calc_coefficient_g(aph3+1, -1, d3a-i3+3, l3+d3a);
                                }
                            }
                            C3[1] *= sum; // d1 = -1: 1, d2 = -1: 1, d3 = -2: 2
                            if (d1 == 0)
                                // corresponds to (1, 1), or multiindex (2, 0, 0)
                                C3[0] = C3[1]; // d1 = 0, d2 = -1: 1, d3 = -2: 2
                            // corresponds to (2, 2), or multiindex (0, 2, 0)
                            C3[3] = C3[1]; // d1 = -1: 1, d2 = -1: 1, d3 = -2: 2
                        }
                    }
                    int col = correspondence.index2number(index_col);
                    for (int i = 0; i < 6; ++i){
                        valuetype ttmp = C1[i] * C2[i] * C3[i];
                        if (fabs(ttmp) < tol_zero) continue;
                        ttmp *= pow((valuetype) 0.5, 2*(l1+k1)+l2+k2+6);
                        stiff_matrix_element[i].add(row, col, ttmp);
                    }
                }
	    }
	}
    }
    std::cerr << "generate 6 stiff matrix element\n";
    
    // calculate coefficient and multiindex for the assignment of discretized matrix
    n_index_variation.resize(6); // number of the coefficient for the derivative of generalized jacobi polynomial
    index_variation.resize(6); // variation of multiindex corresponding to the coefficients
    // assign n_index_variation, intialize index_variation
    n_index_variation[0] = 1; // $\partial x1$,               or multiindex (2, 0, 0)
    n_index_variation[1] = 2; // $\partial x2 - \partial x1$, or multiindex (1, 1, 0)
    n_index_variation[2] = 4; // $\partial x1 - \partial x3$, or multiindex (1, 0, 1)
    n_index_variation[3] = 2; // $\partial x2$,               or multiindex (0, 2, 0)
    n_index_variation[4] = 2; // $\partial x3 - \partial x2$, or multiindex (0, 1, 1)
    n_index_variation[5] = 4; // $\partial x3$,               or multiindex (0, 0, 2)
    for (int ind = 0; ind < 6; ++ind)
        index_variation[ind].resize(n_index_variation[ind]);
    // assign index_variation respectively
    // $\partial x1$,               or multiindex (2, 0, 0)
    index_variation[0][0] = Unitary_Multiindex[0];
    // $\partial x2 - \partial x1$, or multiindex (1, 1, 0), p = 0, 1 in turns
    index_variation[1][0] = Unitary_Multiindex[1];
    index_variation[1][1] = Unitary_Multiindex[0];
    // $\partial x1 - \partial x3$, or multiindex (1, 0, 1), (p, q) = (0, 0), (1, 0), (0, 1), (1, 1) in turns
    index_variation[2][0] = Unitary_Multiindex[2];
    index_variation[2][1] = Unitary_Multiindex[0] - Unitary_Multiindex[1] + Unitary_Multiindex[2];
    index_variation[2][2] = Unitary_Multiindex[1];
    index_variation[2][3] = Unitary_Multiindex[0];
    // $\partial x2$,               or multiindex (0, 2, 0), p = 0, 1 in turns
    for (int i = 0; i < n_index_variation[3]; ++i)
        index_variation[3][i] = index_variation[1][i];
    // $\partial x3 - \partial x2$, or multiindex (0, 1, 1), q = 0, 1 in turns, correspond to (0, q) for (p, q)
    index_variation[4][0] = Unitary_Multiindex[2];
    index_variation[4][1] = Unitary_Multiindex[1];
    // $\partial x3$,               or multiindex (0, 0, 2), (p, q) = (0, 0), (1, 0), (0, 1), (1, 1) in turns
    for (int i = 0; i < n_index_variation[5]; ++i)
        index_variation[5][i] = index_variation[2][i];
    // calculate the corresponding coefficient
    coefficient_derivative.resize(6);
    for (int ind = 0; ind < 6; ++ind){
        coefficient_derivative[ind].resize(n_index_variation[ind]);
        for (int indt = 0; indt < n_index_variation[ind]; ++indt){
            coefficient_derivative[ind][indt].resize(n_index);
            for (int i = 0; i < n_index; ++i)
                if (index_variation[ind][indt] <= correspondence.number2index(i))
                    coefficient_derivative[ind][indt][i] = calc_coefficient_D(ind, indt, correspondence.number2index(i));
                else coefficient_derivative[ind][indt][i] = 0;
        }
    }
    std::cerr << "calculate coefficients for stiff matrix\n";
    
    // assign stiff matrix actual with n_transform_local, transform_local, weight_transform_local
    std::vector<unsigned int> n_nnz_per_row_actual(n_index, 5 * 5 * 5 * 16 * 16);
    //     the largest one, h^{(3,2)} * max(n_transform_local)^2 * max(n_index_variation)^2
    sp_stiff_matrix_element_actual.resize(6);
    stiff_matrix_element_actual.resize(6);
    for (int ind_var = 0; ind_var < 6; ++ind_var){
	// std::cerr << "ind_var = " << ind_var << '\n';
        sp_stiff_matrix_element_actual[ind_var].reinit(n_index, n_index, n_nnz_per_row_actual);
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = stiff_matrix_element[ind_var].begin(0);
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = stiff_matrix_element[ind_var].end(stiff_matrix_element[ind_var].m()-1);
	const SparsityPattern &sp_pattern = stiff_matrix_element[ind_var].get_sparsity_pattern();
	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	const unsigned int *col_nums = sp_pattern.get_column_numbers();
	for (unsigned int row = 0; row < stiff_matrix_element[ind_var].m(); ++row)
        // for (; spm_ite != spm_end; ++spm_ite){
        //     int row = spm_ite->row();
	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
            // int col = spm_ite->column();
		const unsigned int &col = col_nums[pos_whole];
	    // std::cerr << "row = " << row << ", col = " << col << '\n';
		for (int ind_row_v = 0; ind_row_v < n_index_variation[ind_var]; ++ind_row_v) // v for variation
		    for (int ind_col_v = 0; ind_col_v < n_index_variation[ind_var]; ++ind_col_v){
			Multiindex<3> index_row = correspondence.number2index(row) + index_variation[ind_var][ind_row_v];
			Multiindex<3> index_col = correspondence.number2index(col) + index_variation[ind_var][ind_col_v];
			if (!(Zero_Multiindex <= index_row) || !(Zero_Multiindex <= index_col)) continue;
			int row_actual = correspondence.index2number(index_row);
			int col_actual = correspondence.index2number(index_col);
			for (int ind_row_tl = 0; ind_row_tl < n_transform_local[row_actual]; ++ind_row_tl) // tr for transform local
			    for (int ind_col_tl = 0; ind_col_tl < n_transform_local[col_actual]; ++ind_col_tl){
				sp_stiff_matrix_element_actual[ind_var].add(transform_local[row_actual][ind_row_tl],
									    transform_local[col_actual][ind_col_tl]);
			    }
		    }
	    }
        sp_stiff_matrix_element_actual[ind_var].compress();
	// std::cerr << "build sparsity pattern for " << ind_var << "-th actual stiff matrix element\n";

	stiff_matrix_element_actual[ind_var].reinit(sp_stiff_matrix_element_actual[ind_var]);
        // spm_ite = stiff_matrix_element[ind_var].begin(0);
        // for (; spm_ite != spm_end; ++spm_ite){
        //     int row = spm_ite->row();
        //     int col = spm_ite->column();
        //     valuetype val = spm_ite->value();
	for (unsigned int row = 0; row < stiff_matrix_element[ind_var].m(); ++row)
        // for (; spm_ite != spm_end; ++spm_ite){
        //     int row = spm_ite->row();
	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
            // int col = spm_ite->column();
		const unsigned int &col = col_nums[pos_whole];
		valuetype val = stiff_matrix_element[ind_var].global_entry(pos_whole);
		for (int ind_row_v = 0; ind_row_v < n_index_variation[ind_var]; ++ind_row_v) // v for variation
		    for (int ind_col_v = 0; ind_col_v < n_index_variation[ind_var]; ++ind_col_v){
			Multiindex<3> index_row = correspondence.number2index(row) + index_variation[ind_var][ind_row_v];
			Multiindex<3> index_col = correspondence.number2index(col) + index_variation[ind_var][ind_col_v];
			if (!(Zero_Multiindex <= index_row) || !(Zero_Multiindex <= index_col)) continue;
			int row_actual = correspondence.index2number(index_row);
			int col_actual = correspondence.index2number(index_col);
			for (int ind_row_tl = 0; ind_row_tl < n_transform_local[row_actual]; ++ind_row_tl) // tr for transform local
			    for (int ind_col_tl = 0; ind_col_tl < n_transform_local[col_actual]; ++ind_col_tl)
				stiff_matrix_element_actual[ind_var].add(transform_local[row_actual][ind_row_tl],
									 transform_local[col_actual][ind_col_tl],
									 val
									 * coefficient_derivative[ind_var][ind_row_v][row_actual]
									 * coefficient_derivative[ind_var][ind_col_v][col_actual]
									 * weight_transform_local[row_actual][ind_row_tl]
									 * weight_transform_local[col_actual][ind_col_tl]);
		    
		    }
	    }
    }
    std::cerr << "given transform_local, generate 6 actual stiff matrix element\n";
}

TEMPLATE_TSEM
void THIS_TSEM::build_stiff_matrix(FEMSpace<double, 3> &fem_space)
{ // given fem_space, construct stiff matrix

    // construct discretized matrix and right-hand-side
    std::vector<unsigned int> n_nonzero_per_row(n_dof_total, 0);
    // n_nonzero_per_row.resize(n_dof_total, 0);
    // std::cerr << n_nonzero_per_row.size() << '\n';
    // std::cerr << "assign n_nonzero_per_row\n";
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele)
        for (int ind_var = 0; ind_var < 6; ++ind_var){
	    // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = stiff_matrix_element_actual[ind_var].begin(0);
	    // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = stiff_matrix_element_actual[ind_var].end(stiff_matrix_element_actual[ind_var].m()-1);
	    const SparsityPattern &sp_pattern = stiff_matrix_element_actual[ind_var].get_sparsity_pattern();
	    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	    const unsigned int *col_nums = sp_pattern.get_column_numbers();
	    for (unsigned int row = 0; row < stiff_matrix_element_actual[ind_var].m(); ++row)
		for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		    const unsigned int &col = col_nums[pos_whole];
	    // for (; spm_ite != spm_end; ++spm_ite){
	    // 	int row = spm_ite->row();
	    // 	int col = spm_ite->column();
		    for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][row]; ++ind_nnz)
			n_nonzero_per_row[transform_ind_global2local[ind_ele][row][ind_nnz]] += transform_n_global2local[ind_ele][col];
		}   
        }
    // std::cerr << "count nonzero elements for stiff matrix\n";
    // std::cerr << n_nonzero_per_row.size() << '\n';

    sp_stiff_matrix.reinit(n_dof_total, n_dof_total, n_nonzero_per_row);
    // std::cerr << "reinit sparsity pattern for stiff matrix\n";
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele)
        for (int ind_var = 0; ind_var < 6; ++ind_var){
            // int m_matrix = stiff_matrix_element_actual[ind_var].m();
            // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = stiff_matrix_element_actual[ind_var].begin(0);
            // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = stiff_matrix_element_actual[ind_var].end(stiff_matrix_element_actual[ind_var].m()-1);
	    const SparsityPattern &sp_pattern = stiff_matrix_element_actual[ind_var].get_sparsity_pattern();
	    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	    const unsigned int *col_nums = sp_pattern.get_column_numbers();
	    for (unsigned int row = 0; row < stiff_matrix_element_actual[ind_var].m(); ++row)
		for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		    const unsigned int &col = col_nums[pos_whole];
            // for (; spm_ite != spm_end; ++spm_ite){
            //     int row = spm_ite->row();
            //     int col = spm_ite->column();
		    if (col >= n_index) std::cerr << "find illegal col = " << col << '\n';
		    // std::cerr << "row = " << row << ", col = " << col << '\n';
		    for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][row]; ++ind_nnz_row)
			for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
			    sp_stiff_matrix.add(transform_ind_global2local[ind_ele][row][ind_nnz_row],
						transform_ind_global2local[ind_ele][col][ind_nnz_col]);
		}
        }
    sp_stiff_matrix.compress();
    // std::cerr << "build sparsity pattern for stiff matrix\n";
    
    stiff_matrix.reinit(sp_stiff_matrix);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	// std::cerr << "ind_ele = " << ind_ele << '\n';
        const std::vector<int>& element_dof = fem_space.element(ind_ele).dof();
        valuetype coef = ((valuetype) 2) / (3 * val_volume[ind_ele]);
        // traverse each stiff_matrix_element, assign their contribution to stiff matrix
        for (int ind1 = 0; ind1 <= 2; ++ind1) // traverse index of face
            for (int ind2 = ind1+1; ind2 <= 3; ++ind2){
                int ind_p1, ind_p2; // endpoints of the common line between face $F_{ind1}$ and $F_{ind2}$
                if (ind1 == 0)
                    ind_p1 = ind2 % 2 + 1; // ind2 = 1: 2, 3; 2: 1, 3; 3: 1, 2
                else
                    ind_p1 = 0;
                ind_p2 = 6 - ind1 - ind2 - ind_p1;
                AFEPack::Point<3> p1 = fem_space.dofInfo(element_dof[ind1]).interp_point,   p2 = fem_space.dofInfo(element_dof[ind2]).interp_point;
                AFEPack::Point<3> ps = fem_space.dofInfo(element_dof[ind_p1]).interp_point, pe = fem_space.dofInfo(element_dof[ind_p2]).interp_point;
                valuetype coef_dihedral_angle = 0.25 * calc_inner_product(calc_cross_product(pe - ps, p1 - ps), calc_cross_product(pe - ps, p2 - ps));
                if (fabs(coef_dihedral_angle) < tol_zero) continue;
                int ind = ((ind1 == 0) ? correspondence.index2number(Unitary_Multiindex[ind2-1] * 2) - 4 // corresponds to $x_{ind2}^2$, 4=correspondence.n_begin(2)
                           : correspondence.index2number(Unitary_Multiindex[ind1-1] + Unitary_Multiindex[ind2-1]) - 4); // corresponds to $x_{ind2} - x_{ind1}$
                // add contribution of stiff_matrix_element
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = stiff_matrix_element_actual[ind].begin(0);
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = stiff_matrix_element_actual[ind].end(stiff_matrix_element_actual[ind].m()-1);
                // for (; spm_ite != spm_end; ++spm_ite){
                //     int row = spm_ite->row();
                //     int col = spm_ite->column();
                //     valuetype val = spm_ite->value();
		const SparsityPattern &sp_pattern = stiff_matrix_element_actual[ind].get_sparsity_pattern();
		const std::size_t *row_start = sp_pattern.get_rowstart_indices();
		const unsigned int *col_nums = sp_pattern.get_column_numbers();
		for (unsigned int row = 0; row < stiff_matrix_element_actual[ind].m(); ++row)
		    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
			const unsigned int &col = col_nums[pos_whole];
			valuetype val = stiff_matrix_element_actual[ind].global_entry(pos_whole);
			for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][row]; ++ind_nnz_row)
			    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
				stiff_matrix.add(transform_ind_global2local[ind_ele][row][ind_nnz_row],
						 transform_ind_global2local[ind_ele][col][ind_nnz_col],
						 val * coef * coef_dihedral_angle
						 * transform_val_global2local[ind_ele][row][ind_nnz_row] * transform_val_global2local[ind_ele][col][ind_nnz_col]);
		    }
            }
    }
    std::cerr << "construct stiff matrix\n";
}

TEMPLATE_TSEM
void THIS_TSEM::build_mass_matrix_init()
{
    // construct sparsity pattern for mass matrix element
    std::vector<unsigned int> n_nnz_per_row_mass(n_index, 5 * 7 * 9); // the largest one
					   SparsityPattern sp_mass_matrix_element(n_index, n_index, n_nnz_per_row_mass);
    for (int row = 0; row < n_index; ++row){
        Multiindex<3> index_row = correspondence.number2index(row);
        for (int d1 = -2; d1 <= 2; ++d1)
            for (int d2 = -3; d2 <= 3; ++d2)
                for (int d3 = -4; d3 <= 4; ++d3){
                    Multiindex<3> index_col = index_row + Unitary_Multiindex[0] * d1 + Unitary_Multiindex[1] * (d2-d1) + Unitary_Multiindex[2] * (d3-d2);
                    if (!(Zero_Multiindex <= index_col) || index_col.sum() > M) continue;
		    // std::cerr << "index_col = [" << index_col.index[0] << ", " << index_col.index[1] << ", " << index_col.index[2] <<  "]\n";
                    int col = correspondence.index2number(index_col);
                    sp_mass_matrix_element.add(row, col);
                }
    }
    sp_mass_matrix_element.compress();

    // construct sparse matrix of stiff matrix element
    SparseMatrix<valuetype> mass_matrix_element(sp_mass_matrix_element);
    valuetype C[3]; // coefficient [C1, C2, C3]
    for (int row = 0; row < n_index; ++row){
        Multiindex<3> index_row = correspondence.number2index(row);
        int l1 = index_row.index[0], l2 = index_row.index[1], l3 = index_row.index[2];
        for (int d1 = -2; d1 <= 2; ++d1){
            int k1 = l1 + d1;
            if (k1 < 0) continue; // continue if multiindex has negative component
            C[0] = (valuetype) 0;
            if (l1 == 0)
                switch (k1){
                case 0:  C[0] = (valuetype) 2;       break;
                case 2:  C[0] = ((valuetype) -1)/3;  break;
                default: C[0] = (valuetype) 0;
                }
            else if (l1 == 1)
                switch (k1){
                case 1:  C[0] = ((valuetype) 2)/3;   break;
                case 3:  C[0] = ((valuetype) -2)/15; break;
                default: C[0] = (valuetype) 0;
                }
            else{
                valuetype sum = (valuetype) 0;
                for (int i1 = ((d1-1 > -1) ? d1-1 : -1); i1 <= ((d1+1 < 1) ? d1+1 : 1); ++i1)
                    sum += (((valuetype) calc_delta(i1, 0)) + calc_coefficient_a(-1, -1, i1+2, l1+i1)) * (((valuetype) calc_delta(d1, i1)) - calc_coefficient_a(-1, -1, d1-i1+2, l1+d1));
                C[0] = ((valuetype) (l1-1)) / ((2*l1-1)*2*l1) * sum;
            }
            for (int d2 = -3; d2 <= 3; ++d2){
                int d2a = d2 - d1; // actual d2
                int k2 = l2 + d2a;
                if (k2 < 0) continue; // continue if multiindex index has negative component
                int aph2 = 2 * l1 - 1;
                C[1] = (valuetype) 0;
                if (l1 == 0 && l2 == 0)
                    switch (d1){
                    case 0: switch (k2){
                        case 0: C[1] = (valuetype) 2; break;
                        case 1: C[1] = ((valuetype) -2)/3; break;
                        case 2: C[1] = ((valuetype) -1)/3; break;
                        case 3: C[1] = ((valuetype) 2)/15; break;
                        }
                        break;
                    case 1: switch (k2){
                        case 0: C[1] = ((valuetype) 8)/3; break;
                        case 1: C[1] = ((valuetype) 4)/3; break;
                        case 2: C[1] = ((valuetype) -2)/5; break;
                        }
                        break;
                    case 2: switch (k2){
                        case 0: C[1] = (valuetype) 4; break;
                        case 1: C[1] = ((valuetype) 16)/5; break;
                        }
                    }
                else if (l1 > 0 && l2 == 0){
                    valuetype numerator = (valuetype) pow(2, 2*l1+d1+2);
                    if (d2a != 0){
                        numerator *= (valuetype) (2*l1 + 2*d1 - 1 + d2a);
                        for (int i = 0; i < d2a-1; ++i)
                            numerator *= (valuetype) d1-2 + i;
                    }
                    valuetype denominator = (valuetype) 1;
                    for (int i = 0; i < d2a+1; ++i)
                        denominator *= (valuetype) 2*l1+d1+2 + i;
                    C[1] = numerator / denominator;
                }
                else if (l1 == 0 && k1 == 0 && l2 == 1)
                    switch (k2){
                    case 0: C[1] = ((valuetype) -2)/3; break;
                    case 1: C[1] = ((valuetype) 2)/3; break;
                    case 2: C[1] = ((valuetype) 1)/15; break;
                    case 3: C[1] = ((valuetype) -2)/15; break;
                    case 4: C[1] = ((valuetype) 2)/35;
                    }
                else if (l1 == 0 && k1 > 0 && l2 == 1)
                    for (int j = 0; j <= 1; ++j){
                        valuetype numerator = (valuetype) pow(2, d1+2+j);
                        if (j == 1) numerator *= (valuetype) -1;
                        if (k2 != 0){
                            numerator *= (valuetype) 2*d1 - 1 + k2;
                            for (int i = 0; i < k2-1; ++i)
                                numerator *= (valuetype) d1-2-j + i;
                        }
                        valuetype denominator = (valuetype) 1;
                        for (int i = 0; i < k2+1; ++i)
                            denominator *= (valuetype) d1+2+j + i;
                        C[1] += numerator / denominator;
                    }
                else{
                    int ds = d2a + d1;
                    for (int i21 = ((ds-2 > -1) ? ds-2 : -1); i21 <= ((ds+2 < 1) ? ds+2 : 1); ++i21){
                        valuetype tmp = ((valuetype) calc_delta(i21, 0)) + calc_coefficient_a(aph2, -1, i21+2, l2+i21);
                        for (int i22 = ((ds-i21-1 > -1) ? ds-i21-1 : -1); i22 <= ((ds-i21+1 < 1) ? ds-i21+1 : 1); ++i22){
                            int i2 = i21 + i22;
                            valuetype tmp2;
                            switch (d1){
                            case -2: tmp2 =                       calc_coefficient_c(aph2-2, -1, i22+2, l2+i2+1); break;
                            case 2:  tmp2 =                     4*calc_coefficient_g(aph2,   -1, i22+2, l2+i2-1); break;
                            default: tmp2 = (((valuetype) calc_delta(i22, 0)) - calc_coefficient_a(aph2,   -1, i22+2, l2+i2));
                            }
                            switch (d1){
                            case -2: tmp2 *=                       calc_coefficient_c(aph2-4, -1, d2a-i2,   k2); break;
                            case -1: tmp2 *=                       calc_coefficient_c(aph2-2, -1, d2a-i2+1, k2); break;
                            case 0:  tmp2 *= ((valuetype) calc_delta(d2a, i2)) - calc_coefficient_a(aph2,   -1, d2a-i2+2, k2); break;
                            case 1:  tmp2 *=                     4*calc_coefficient_g(aph2,   -1, d2a-i2+3, k2); break;
                            case 2:  tmp2 *=                     4*calc_coefficient_g(aph2+2, -1, d2a-i2+4, k2);
                            }
                            C[1] += tmp * tmp2;
                        }
                    }
                    C[1] *= ((valuetype) pow(2, 2*l1-1) * (2*l1 + l2 - 1)) / (l2 * (2*l1 + 2*l2 - 1)); // the gamma
                }
                if (l1 >= 1 && k2 == 0){
                    valuetype numerator = (valuetype) pow(2, l1+k1+2);
                    if (l2 != 0){
                        numerator *= (valuetype) 2*l1 + l2 - 1;
                        for (int i = 0; i < l2-1; ++i)
                            numerator *= (valuetype) l1-k1-2 + i;
                    }
                    valuetype denominator = (valuetype) 1;
                    for (int i = 0; i < l2+1; ++i)
                        denominator *= (valuetype) l1+k1+2 + i;
                    C[1] = numerator / denominator;
                }
                if (l1 >= 1 && k1 == 0 && k2 == 1){
                    C[1] = (valuetype) 0;
                    for (int j = 0; j <= 1; ++j){
                        valuetype numerator = (valuetype) pow(2, l1+k1+j+2);
                        if (j == 1) numerator *= (valuetype) -1;
                        if (l2 != 0){
                            numerator *= (valuetype) 2*l1 + l2 - 1;
                            for (int i = 0; i < l2-1; ++i)
                                numerator *= (valuetype) l1-k1-j-2 + i;
                        }
                        valuetype denominator = (valuetype) 1;
                        for (int i = 0; i < l2+1; ++i)
                            denominator *= (valuetype) l1+k1+j+2 + i;
                        C[1] += numerator / denominator;
                    }
                }
                for (int d3 = -4; d3 <= 4; ++d3){
                    int d = d1 + d2a, l = l1 + l2, k = k1 + k2; // d1 is exactly actual d1
                    int d3a = d3 - d; // actual d3
                    int k3 = l3 + d3a;
                    if (k3 < 0 || k1+k2+k3 > M) continue; // continue if exist negative component or the summation exceed M
                    int aph3 = 2 * l1 + 2 * l2 - 1;
                    C[2] = (valuetype) 0;
                    if (l1 == 0 && l2 == 0 && l3 == 0)
                        switch (d){
                        case 0: switch (k3){
                            case 0: C[2] = ((valuetype) 8)/3; break;
                            case 1: C[2] = ((valuetype) -4)/3; break;
                            case 2: C[2] = ((valuetype) -2)/5; break;
                            case 3: C[2] = ((valuetype) 4)/15; break;
                            case 4: C[2] = ((valuetype) -2)/35; break;
                            }
                            break;
                        case 1: switch (k3){
                            case 0: C[2] = (valuetype) 4; break;
                            case 1: C[2] = ((valuetype) 8)/5; break;
                            case 2: C[2] = ((valuetype) -4)/5; break;
                            case 3: C[2] = ((valuetype) 16)/105;
                            }
                            break;
                        case 2: switch (k3){
                            case 0: C[2] = ((valuetype) 32)/5; break;
                            case 1: C[2] = ((valuetype) 64)/15; break;
                            case 2: C[2] = ((valuetype) -16)/21;
                            }
                            break;
                        case 3: switch (k3){
                            case 0: C[2] = ((valuetype) 32)/3; break;
                            case 1: C[2] = ((valuetype) 64)/7;
                            }
                        }
                    else if (l > 0 && l3 == 0){
                        valuetype numerator = (valuetype) pow(2, 2*l+d+3);
                        if (d3a != 0){
                            numerator *= (valuetype) (2*l + 2*d - 1 + d3a);
                            for (int i = 0; i < d3a-1; ++i)
                                numerator *= (valuetype) d-3 + i;
                        }
                        valuetype denominator = (valuetype) 1;
                        for (int i = 0; i < d3a+1; ++i)
                            denominator *= (valuetype) 2*l+d+3 + i;
                        C[2] = numerator / denominator;
                    }
                    else if (l1 == 0 && l2 == 0 && k1 == 0 && k2 == 0 && l3 == 1)
                        switch (k3){
                        case 0: C[2] = ((valuetype) -4)/3; break;
                        case 1: C[2] = ((valuetype) 16)/15; break;
                        case 2: C[2] = ((valuetype) 2)/15; break;
                        case 3: C[2] = ((valuetype) -4)/21; break;
                        case 4: C[2] = ((valuetype) 4)/35; break;
                        case 5: C[2] = ((valuetype) -8)/315;
                        }
                    else if (l1 == 0 && l2 == 0 && d > 0 && l3 == 1)
                        for (int j = 0; j <= 1; ++j){
                            valuetype numerator = (valuetype) pow(2, d+3+j);
                            if (j == 1) numerator *= (valuetype) -1;
                            if (k3 != 0){
                                numerator *= (valuetype) 2*d - 1 + k3;
                                for (int i = 0; i < k3-1; ++i)
                                    numerator *= (valuetype) d-3-j + i;
                            }
                            valuetype denominator = (valuetype) 1;
                            for (int i = 0; i < k3+1; ++i)
                                denominator *= (valuetype) d+3+j + i;
                            C[2] += numerator / denominator;
                        }
                    else{
                        int ds = d3a + d;
                        for (int i31 = ((ds-3 > -1) ? ds-3 : -1); i31 <= ((ds+3 < 1) ? ds+3 : 1); ++i31){
                            valuetype tmp = ((valuetype) calc_delta(i31, 0)) + calc_coefficient_a(aph3, -1, i31+2, l3+i31);
                            for (int i32 = ((ds-i31-2 > -1) ? ds-i31-2 : -1); i32 <= ((ds-i31+2 < 1) ? ds-i31+2 : 1); ++i32){
                                valuetype tmp2;
                                switch (d){
                                case -3: tmp2 =                      calc_coefficient_c(aph3-2, -1, i32+2, l3+i31+i32+1); break;
                                case 3:  tmp2 =                    4*calc_coefficient_g(aph3,   -1, i32+2, l3+i31+i32-1); break;
                                default: tmp2 = ((valuetype) calc_delta(i32, 0)) - calc_coefficient_a(aph3,   -1, i32+2, l3+i31+i32);
                                }
                                for (int i33 = ((ds-i31-i32-1 > -1) ? ds-i31-i32-1 : -1); i33 <= ((ds-i31-i32+1 < 1) ? ds-i31-i32+1 : 1); ++i33){
                                    int i3 = i31 + i32 + i33;
                                    valuetype tmp3;
                                    switch (d){
                                    case -3: tmp3 =                      calc_coefficient_c(aph3-4, -1, i33+2, l3+i3+2); break;
                                    case -2: tmp3 =                      calc_coefficient_c(aph3-2, -1, i33+2, l3+i3+1); break;
                                    case 2:  tmp3 =                    4*calc_coefficient_g(aph3,   -1, i33+2, l3+i3-1); break;
                                    case 3:  tmp3 =                    4*calc_coefficient_g(aph3+2, -1, i33+2, l3+i3-2); break;
                                    default: tmp3 = ((valuetype) calc_delta(i33, 0)) - calc_coefficient_a(aph3,   -1, i33+2, l3+i3);
                                    }
                                    switch (d){
                                    case -3: tmp3 *=                       calc_coefficient_c(aph3-6, -1, d3a-i3-1, k3); break;
                                    case -2: tmp3 *=                       calc_coefficient_c(aph3-4, -1, d3a-i3,   k3); break;
                                    case -1: tmp3 *=                       calc_coefficient_c(aph3-2, -1, d3a-i3+1, k3); break;
                                    case 0:  tmp3 *= ((valuetype) calc_delta(d3a, i3)) - calc_coefficient_a(aph3,   -1, d3a-i3+2, k3); break;
                                    case 1:  tmp3 *=                     4*calc_coefficient_g(aph3,   -1, d3a-i3+3, k3); break;
                                    case 2:  tmp3 *=                     4*calc_coefficient_g(aph3+2, -1, d3a-i3+4, k3); break;
                                    case 3:  tmp3 *=                     4*calc_coefficient_g(aph3+4, -1, d3a-i3+5, k3);
                                    }
                                    C[2] += tmp * tmp2 * tmp3;
                                }
                            }
                        }
                        C[2] *= ((valuetype) pow(2, 2*l-1) * (2*l + l3 - 1)) / (l3 * (2*l + 2*l3 - 1)); // the gamma
                    }
                    if (l >= 1 && k3 == 0){
                        valuetype numerator = (valuetype) pow(2, l+k+3);
                        if (l3 != 0){
                            numerator *= (valuetype) 2*l + l3 - 1;
                            for (int i = 0; i < l3-1; ++i)
                                numerator *= (valuetype) l-k-3 + i;
                        }
                        valuetype denominator = (valuetype) 1;
                        for (int i = 0; i < l3+1; ++i)
                            denominator *= (valuetype) l+k+3 + i;
                        C[2] = numerator / denominator;
                    }
                    if (l >= 1 && k == 0 && k3 == 1){
                        C[2] = (valuetype) 0;
                        for (int j = 0; j <= 1; ++j){
                            valuetype numerator = (valuetype) pow(2, l+k+j+3);
                            if (j == 1) numerator *= (valuetype) -1;
                            if (l3 != 0){
                                numerator *= (valuetype) 2*l + l3 - 1;
                                for (int i = 0; i < l3-1; ++i)
                                    numerator *= (valuetype) l-k-j-3 + i;
                            }
                            valuetype denominator = (valuetype) 1;
                            for (int i = 0; i < l3+1; ++i)
                                denominator *= (valuetype) l+k+j+3 + i;
                            C[2] += numerator / denominator;
                        }
                    }
                    valuetype val = C[0] * C[1] * C[2];
                    if (fabs(val) < tol_zero) continue; // continue if one of coefficient is 0
                    val *= pow((valuetype) 0.5, 2*(l1+k1)+l2+k2+6);
                    Multiindex<3> index_col = Unitary_Multiindex[0] * k1 + Unitary_Multiindex[1] * k2 + Unitary_Multiindex[2] * k3;
                    int col = correspondence.index2number(index_col);
                    mass_matrix_element.add(row, col, val);
                }
            }
        }
    }
    std::cerr << "generate mass matrix element\n";

    // std::ofstream sparsity("./info_mass_element_sparsity.m");
    // std::vector<unsigned int> ind_actual(n_index, 0);
    // unsigned int n_inner_index = 1;
    // Multiindex<3> index_cmp;
    // index_cmp.index[0] = 2; index_cmp.index[1] = 1; index_cmp.index[2] = 1;
    // for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index){
    // 	Multiindex<3> index_now = correspondence.number2index(ind_index);
    // 	if (index_cmp <= index_now){
    // 	    unsigned int l1 = index_now.index[0]-2, l2 = index_now.index[1]-1, l3 = index_now.index[2]-1;
    // 	    unsigned int tmp;
    // 	    tmp = ((l1-1)*l1*(2*l1-1) + l1*(M+1)*(M+2)*6 - (2*M+3)*(l1-1)*l1*3) / 12
    // 		+ (M-l1+1)*l2 - (l2-1)*l2/2
    // 		+ l3;
    // 	    ind_actual[ind_index] = tmp + 1;
    // 	}
    // }
    // SparseMatrix<double>::iterator spm_ite = mass_matrix_element.begin(0);
    // SparseMatrix<double>::iterator spm_end = mass_matrix_element.end(mass_matrix_element.m()-1);
    // sparsity << "row_massEle = [";
    // for (; spm_ite != spm_end; ++spm_ite)
    // 	if (ind_actual[spm_ite->row()] > 0 && ind_actual[spm_ite->column()] > 0)
    // 	    sparsity << ind_actual[spm_ite->row()] << ' ';
    // sparsity << "];\n";
    // spm_ite = mass_matrix_element.begin(0);
    // sparsity << "col_massEle = [";
    // for (; spm_ite != spm_end; ++spm_ite)
    // 	if (ind_actual[spm_ite->row()] > 0 && ind_actual[spm_ite->column()] > 0)
    // 	    sparsity << ind_actual[spm_ite->column()] << ' ';
    // sparsity << "];\n";
    // spm_ite = mass_matrix_element.begin(0);
    // sparsity << "val_massEle = [";
    // for (; spm_ite != spm_end; ++spm_ite)
    // 	if (ind_actual[spm_ite->row()] > 0 && ind_actual[spm_ite->column()] > 0)
    // 	    sparsity << spm_ite->value() << ' ';
    // sparsity << "];\n";
    // sparsity << "spm_massEle = sparse(row_massEle, col_massEle, val_massEle);";
    // sparsity.close();

    // assign stiff matrix actual with n_transform_local, transform_local, weight_transform_local
    std::vector<unsigned int> n_nnz_per_row_actual_mass(n_index, 5 * 7 * 9 * 16 * 16);
    //     the largest one, h^{(3,2)} * max(n_transform_local)^2 * max(n_index_variation)^2
    sp_mass_matrix_element_actual.reinit(n_index, n_index, n_nnz_per_row_actual_mass);
    // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = mass_matrix_element.begin(0);
    // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = mass_matrix_element.end(mass_matrix_element.m()-1);
    // for (; spm_ite != spm_end; ++spm_ite){
    // 	int row = spm_ite->row();
    // 	int col = spm_ite->column();
    const SparsityPattern &sp_pattern = mass_matrix_element.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    for (unsigned int row = 0; row < mass_matrix_element.m(); ++row)
	for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
	    const unsigned int &col = col_nums[pos_whole];
	    for (int ind_row_tl = 0; ind_row_tl < n_transform_local[row]; ++ind_row_tl) // tr for transform local
		for (int ind_col_tl = 0; ind_col_tl < n_transform_local[col]; ++ind_col_tl)
		    sp_mass_matrix_element_actual.add(transform_local[row][ind_row_tl],
						      transform_local[col][ind_col_tl]);
	}
    sp_mass_matrix_element_actual.compress();

    mass_matrix_element_actual.reinit(sp_mass_matrix_element_actual);
    // spm_ite = mass_matrix_element.begin(0);
    // for (; spm_ite != spm_end; ++spm_ite){
    // 	int row = spm_ite->row();
    // 	int col = spm_ite->column();
    // 	valuetype val = spm_ite->value();
    for (unsigned int row = 0; row < mass_matrix_element.m(); ++row)
	for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
	    const unsigned int &col = col_nums[pos_whole];
	    valuetype val = mass_matrix_element.global_entry(pos_whole);
	    for (int ind_row_tl = 0; ind_row_tl < n_transform_local[row]; ++ind_row_tl) // tr for transform local
		for (int ind_col_tl = 0; ind_col_tl < n_transform_local[col]; ++ind_col_tl)
		    mass_matrix_element_actual.add(transform_local[row][ind_row_tl],
						   transform_local[col][ind_col_tl],
						   val
						   * weight_transform_local[row][ind_row_tl] * weight_transform_local[col][ind_col_tl]);
	}
    std::cerr << "given transform_local, assign actual mass matrix element\n";

    
    // const SparsityPattern &sp_pattern_t = mass_matrix_element_actual.get_sparsity_pattern();
    // const std::size_t *row_start_t = sp_pattern_t.get_rowstart_indices();
    // const unsigned int *col_nums_t = sp_pattern_t.get_column_numbers();
    // std::ofstream output("./mass_element.m");
    // unsigned int index = 1;
    // for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row)
    // 	for (unsigned int pos_whole = row_start_t[row]; pos_whole < row_start_t[row+1]; ++pos_whole){
    // 	    const unsigned int &col = col_nums_t[pos_whole];
    // 	    output << "row(" << index << ") = " << row << "; "
    // 		   << "col(" << index << ") = " << col << "; "
    // 		   << "val(" << index << ") = " << mass_matrix_element_actual.global_entry(pos_whole) << ";\n";
    // 	    index++;
    // 	}
}

TEMPLATE_TSEM
void THIS_TSEM::build_mass_matrix(FEMSpace<double, 3> &fem_space)
{
    // assemble mass matrix
    std::vector<unsigned int> n_nonzero_per_row_mass;
    n_nonzero_per_row_mass.resize(n_dof_total, 0);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	const unsigned int *col_nums = sp_pattern.get_column_numbers();
	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row){
	    int cnt = 0;
	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		const unsigned int &col = col_nums[pos_whole];
        // int m_matrix = mass_matrix_element_actual.m();
        // for (int row = 0; row < m_matrix; ++row){
        //     SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = mass_matrix_element_actual.begin(row);
        //     SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = mass_matrix_element_actual.end(row);
		// for (; spm_ite != spm_end; ++spm_ite)
		    // cnt += transform_n_global2local[ind_ele][spm_ite->column()];
		cnt += transform_n_global2local[ind_ele][col];
	    }
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][row]; ++ind_nnz)
                n_nonzero_per_row_mass[transform_ind_global2local[ind_ele][row][ind_nnz]] += cnt;
        }
    }

    sp_mass_matrix.reinit(n_dof_total, n_dof_total, n_nonzero_per_row_mass);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // int m_matrix = mass_matrix_element_actual.m();
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = mass_matrix_element_actual.begin(0);
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = mass_matrix_element_actual.end(m_matrix-1);
        // for (; spm_ite != spm_end; ++spm_ite){
        //     int row = spm_ite->row();
        //     int col = spm_ite->column();
	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	const unsigned int *col_nums = sp_pattern.get_column_numbers();
	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row)
	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		const unsigned int &col = col_nums[pos_whole];
		for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][row]; ++ind_nnz_row)
		    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
			sp_mass_matrix.add(transform_ind_global2local[ind_ele][row][ind_nnz_row],
					   transform_ind_global2local[ind_ele][col][ind_nnz_col]);
	    }
    }
    sp_mass_matrix.compress();
    
    mass_matrix.reinit(sp_mass_matrix);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        const std::vector<int>& element_dof = fem_space.element(ind_ele).dof();
        valuetype coef = 6 * val_volume[ind_ele]; // the actual volume is fabs(volume * jacobian)
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = mass_matrix_element_actual.begin(0);
        // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = mass_matrix_element_actual.end(mass_matrix_element_actual.m()-1);
        // for (; spm_ite != spm_end; ++spm_ite){
        //     int row = spm_ite->row();
        //     int col = spm_ite->column();
        //     valuetype val = spm_ite->value();
	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
	const unsigned int *col_nums = sp_pattern.get_column_numbers();
	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row)
	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		const unsigned int &col = col_nums[pos_whole];
		valuetype val = mass_matrix_element_actual.global_entry(pos_whole);
		for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][row]; ++ind_nnz_row)
		    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
			mass_matrix.add(transform_ind_global2local[ind_ele][row][ind_nnz_row],
					transform_ind_global2local[ind_ele][col][ind_nnz_col],
					val * coef
					* transform_val_global2local[ind_ele][row][ind_nnz_row] * transform_val_global2local[ind_ele][col][ind_nnz_col]);
	    }
    }
    std::cerr << "construct mass matrix\n";

    
    // // assemble essential mass matrix
    // std::vector<unsigned int> n_nonzero_per_row_mass_essential;
    // n_nonzero_per_row_mass_essential.resize(n_dof_total, 0);
    // for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
    // 	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
    // 	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    // 	const unsigned int *col_nums = sp_pattern.get_column_numbers();
    // 	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row){
    // 	    int cnt = 0;
    // 	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
    // 		const unsigned int &col = col_nums[pos_whole];
    // 		// cnt += transform_n_global2local[ind_ele][col];
    // 		cnt += transform_n_local2global[ind_ele][col];
    // 	    }
    //         for (int ind_nnz = 0; ind_nnz < transform_n_local2global[ind_ele][row]; ++ind_nnz)
    //             n_nonzero_per_row_mass_essential[transform_ind_local2global[ind_ele][row][ind_nnz]] += cnt;
    //     }
    // }

    // sp_mass_matrix_essential.reinit(n_dof_total, n_dof_total, n_nonzero_per_row_mass_essential);
    // for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
    // 	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
    // 	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    // 	const unsigned int *col_nums = sp_pattern.get_column_numbers();
    // 	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row)
    // 	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
    // 		const unsigned int &col = col_nums[pos_whole];
    // 		// for (int ind_nnz_row = 0; ind_nnz_row < transform_n_local2global[ind_ele][row]; ++ind_nnz_row)
    // 		//     for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
    // 		// 	sp_mass_matrix_essential.add(transform_ind_local2global[ind_ele][row][ind_nnz_row],
    // 		// 				     transform_ind_global2local[ind_ele][col][ind_nnz_col]);
    // 		for (int ind_nnz_row = 0; ind_nnz_row < transform_n_local2global[ind_ele][row]; ++ind_nnz_row)
    // 		    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_local2global[ind_ele][col]; ++ind_nnz_col)
    // 			sp_mass_matrix_essential.add(transform_ind_local2global[ind_ele][row][ind_nnz_row],
    // 						     transform_ind_local2global[ind_ele][col][ind_nnz_col]);
    // 	    }
    // }
    // sp_mass_matrix_essential.compress();
    
    // mass_matrix_essential.reinit(sp_mass_matrix_essential);
    // for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
    //     valuetype volume = fem_space.element(ind_ele).templateElement().volume();
    //     AFEPack::Point<3> point_tmp;
    //     for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
    //     valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed
    //     const std::vector<int>& element_dof = fem_space.element(ind_ele).dof();
    //     valuetype coef = 6 * fabs(volume * jacobian); // the actual volume is fabs(volume * jacobian)
    // 	const SparsityPattern &sp_pattern = mass_matrix_element_actual.get_sparsity_pattern();
    // 	const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    // 	const unsigned int *col_nums = sp_pattern.get_column_numbers();
    // 	for (unsigned int row = 0; row < mass_matrix_element_actual.m(); ++row)
    // 	    for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
    // 		const unsigned int &col = col_nums[pos_whole];
    // 		valuetype val = mass_matrix_element_actual.global_entry(pos_whole);
    // 		// for (int ind_nnz_row = 0; ind_nnz_row < transform_n_local2global[ind_ele][row]; ++ind_nnz_row)
    // 		//     for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][col]; ++ind_nnz_col)
    // 		// 	mass_matrix_essential.add(transform_ind_local2global[ind_ele][row][ind_nnz_row],
    // 		// 				  transform_ind_global2local[ind_ele][col][ind_nnz_col],
    // 		// 				  val * coef
    // 		// 				  * transform_val_local2global[ind_ele][row][ind_nnz_row] * transform_val_global2local[ind_ele][col][ind_nnz_col]);
    // 		for (int ind_nnz_row = 0; ind_nnz_row < transform_n_local2global[ind_ele][row]; ++ind_nnz_row)
    // 		    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_local2global[ind_ele][col]; ++ind_nnz_col)
    // 			mass_matrix_essential.add(transform_ind_local2global[ind_ele][row][ind_nnz_row],
    // 						  transform_ind_local2global[ind_ele][col][ind_nnz_col],
    // 						  val * coef
    // 						  * transform_val_local2global[ind_ele][row][ind_nnz_row] * transform_val_local2global[ind_ele][col][ind_nnz_col]);
    // 	    }
    // }
}

TEMPLATE_TSEM
void THIS_TSEM::build_mass_V_matrix(std::vector<unsigned int> &n_nonzero_per_row_mass_V, SparsityPattern &sp_pattern, SparseMatrix<valuetype> &sp_matrix,
				    std::vector<std::vector<valuetype> > &value_V)
{ // function V is define in main cpp file
    
    std::vector<std::vector<std::vector<int> > > col_local_contribution;
    std::vector<std::vector<std::vector<valuetype> > > val_local_contribution;
    std::vector<std::vector<int> > n_local_contribution;
    col_local_contribution.resize(n_element);
    val_local_contribution.resize(n_element);
    n_local_contribution.resize(n_element);
    for (int i = 0; i < n_element; ++i){
        col_local_contribution[i].resize(n_index);
        val_local_contribution[i].resize(n_index);
        n_local_contribution[i].resize(n_index);
    }
    // std::cerr << "setup for calculation of mass_V\n";

    std::vector<int>       col_nonzero_entry(n_index);
    std::vector<valuetype> val_nonzero_entry(n_index);
    int n_nonzero_entry;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	// valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
	// valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed
        for (int ind_index_row = 0; ind_index_row < n_index; ++ind_index_row){
            n_nonzero_entry = 0;
            for (int ind_index_col = 0; ind_index_col < n_index; ++ind_index_col){
                valuetype count = 0; // the integral between row index and col index
                for (int p = 0; p < n_q_point[2]; ++p)
                    count += Weight[2][p] * basis_value_actual[p][ind_index_row] * basis_value_actual[p][ind_index_col] * value_V[ind_ele][p];
                if (fabs(count) > tol_zero){
                    col_nonzero_entry[n_nonzero_entry] = ind_index_col;
                    val_nonzero_entry[n_nonzero_entry] = count * val_volume[ind_ele];
                    n_nonzero_entry++;
                }
            }
            col_local_contribution[ind_ele][ind_index_row].resize(n_nonzero_entry);
            val_local_contribution[ind_ele][ind_index_row].resize(n_nonzero_entry);
            for (int j = 0; j < n_nonzero_entry; ++j){
                col_local_contribution[ind_ele][ind_index_row][j] = col_nonzero_entry[j];
                val_local_contribution[ind_ele][ind_index_row][j] = val_nonzero_entry[j];
            }
            n_local_contribution[ind_ele][ind_index_row] = n_nonzero_entry;
        }
    }
    // std::cerr << "calculate nonzero contribution on each elements for mass_V\n";
    
    n_nonzero_per_row_mass_V.resize(n_dof_total, 0);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele)
        for (int ind_index = 0; ind_index < n_index; ++ind_index){
            // count nonzero contribution for each col
            int cnt = 0;
            for (int ind_nnz = 0; ind_nnz < n_local_contribution[ind_ele][ind_index]; ++ind_nnz)
                cnt += transform_n_global2local[ind_ele][col_local_contribution[ind_ele][ind_index][ind_nnz]];
            // add number to nonzero array
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                n_nonzero_per_row_mass_V[transform_ind_global2local[ind_ele][ind_index][ind_nnz]] += cnt;
        }
    
    sp_pattern.reinit(n_dof_total, n_dof_total, n_nonzero_per_row_mass_V);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele)
        for (int ind_index_row = 0; ind_index_row < n_index; ++ind_index_row)
            for (int ind_nnz_contribution = 0; ind_nnz_contribution < n_local_contribution[ind_ele][ind_index_row]; ++ind_nnz_contribution){
                int& ind_index_col = col_local_contribution[ind_ele][ind_index_row][ind_nnz_contribution];
                for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][ind_index_row]; ++ind_nnz_row)
                    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][ind_index_col]; ++ind_nnz_col)
                        sp_pattern.add(transform_ind_global2local[ind_ele][ind_index_row][ind_nnz_row],
				       transform_ind_global2local[ind_ele][ind_index_col][ind_nnz_col]);
            }
    sp_pattern.compress();
    
    sp_matrix.reinit(sp_pattern);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele)
        for (int ind_index_row = 0; ind_index_row < n_index; ++ind_index_row)
            for (int ind_nnz_contribution = 0; ind_nnz_contribution < n_local_contribution[ind_ele][ind_index_row]; ++ind_nnz_contribution){
                int& ind_index_col = col_local_contribution[ind_ele][ind_index_row][ind_nnz_contribution];
                for (int ind_nnz_row = 0; ind_nnz_row < transform_n_global2local[ind_ele][ind_index_row]; ++ind_nnz_row)
                    for (int ind_nnz_col = 0; ind_nnz_col < transform_n_global2local[ind_ele][ind_index_col]; ++ind_nnz_col)
                        sp_matrix.add(transform_ind_global2local[ind_ele][ind_index_row][ind_nnz_row],
				      transform_ind_global2local[ind_ele][ind_index_col][ind_nnz_col],
				      val_local_contribution[ind_ele][ind_index_row][ind_nnz_contribution]
				      * transform_val_global2local[ind_ele][ind_index_row][ind_nnz_row]
				      * transform_val_global2local[ind_ele][ind_index_col][ind_nnz_col]);
            }    
    std::cerr << "construct mass type matrix for V\n";
}

TEMPLATE_TSEM
void THIS_TSEM::build_flag_bm_cube(valuetype bnd_left, valuetype bnd_right)
{ // build boundary mark for cubic region [Bnd_Left, Bnd_Right]^3

    flag_bm.resize(4); // boundary flag for all dimensional geometry
    flag_bm_dof.resize(n_dof_total, false);
    for (int ind = 0; ind <= 3; ++ind){
	flag_bm[ind].resize(n_geometry[ind], false);
        if (ind == 3) break;
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
            flag_bm[ind][ind_geo] = is_on_boundary(point_ref_mesh[ind][ind_geo], bnd_left, bnd_right);
	    if (!flag_bm[ind][ind_geo]) continue;
	    int location = ind_geo + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
	    int row_start = location_actualdof[location_geometry[location]];
	    for (unsigned int ind_dof = 0; ind_dof < n_dof_geometry[ind]; ++ind_dof)
		flag_bm_dof[row_start + ind_dof] = true;
	}
    }
    std::cerr << "build boundary flag for geometrys with boundary info: Bnd_Left = " << bnd_left << ", Bnd_Right = " << bnd_right << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::build_flag_bm_ball(valuetype radius)
{ // build boundary mark for ball region centering at (0, 0, 0) with radius

    flag_bm.resize(4); // boundary flag for all dimensional geometry
    flag_bm_dof.resize(n_dof_total, false);
    for (int ind = 0; ind <= 3; ++ind){
        flag_bm[ind].resize(n_geometry[ind], false);
        if (ind == 3) break;
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
	    switch (ind){
	    case 0:
		flag_bm[0][ind_geo] = is_on_boundary(point_ref_mesh[0][ind_geo], radius);
		break;
	    case 1:
		flag_bm[1][ind_geo] = flag_bm[0][number_node[0][ind_geo][0]] && flag_bm[0][number_node[0][ind_geo][1]];
		break;
	    case 2:
		flag_bm[2][ind_geo] = flag_bm[0][number_node[1][ind_geo][0]] && flag_bm[0][number_node[1][ind_geo][1]] && flag_bm[0][number_node[1][ind_geo][2]];
		break;
	    default:
		break;
	    }
	    if (!flag_bm[ind][ind_geo]) continue;
	    int location = ind_geo + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
	    int row_start = location_actualdof[location_geometry[location]];
	    for (unsigned int ind_dof = 0; ind_dof < n_dof_geometry[ind]; ++ind_dof)
		flag_bm_dof[row_start + ind_dof] = true;
	}
    }
    std::cerr << "build boundary flag for geometrys with boundary info: center = (0, 0, 0), radius = " << radius << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::build_flag_bm_cube(valuetype bnd_left, valuetype bnd_right, valuetype axis[])
{ // build boundary mark for cubic region [Bnd_Left, Bnd_Right]^3

    flag_bm.resize(4); // boundary flag for all dimensional geometry
    flag_bm_dof.resize(n_dof_total, false);
    std::vector<AFEPack::Point<3> > axis_coord(3);
    for (unsigned int ind_axis = 0; ind_axis < 3; ++ind_axis)
	for (unsigned int ind = 0; ind < 3; ++ind)
	    axis_coord[ind_axis][ind] = axis[ind_axis*3 + ind];
    for (int ind = 0; ind <= 3; ++ind){
	flag_bm[ind].resize(n_geometry[ind], false);
        if (ind == 3) break;
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
	    AFEPack::Point<3> point_now;
	    for (unsigned int ind_c = 0; ind_c < 3; ++ind_c)
		point_now[ind_c] = calc_inner_product(axis_coord[ind_c], point_ref_mesh[ind][ind_geo]);
            flag_bm[ind][ind_geo] = is_on_boundary(point_now, bnd_left, bnd_right);
	    if (!flag_bm[ind][ind_geo]) continue;
	    int location = ind_geo + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
	    int row_start = location_actualdof[location_geometry[location]];
	    for (unsigned int ind_dof = 0; ind_dof < n_dof_geometry[ind]; ++ind_dof)
		flag_bm_dof[row_start + ind_dof] = true;
	}
    }
    std::cerr << "build boundary flag for geometrys with boundary info: Bnd_Left = " << bnd_left << ", Bnd_Right = " << bnd_right << '\n';
}

TEMPLATE_TSEM
void THIS_TSEM::build_flag_bm_mesh(RegularMesh<3> &mesh)
{
    flag_bm.resize(4); // boundary flag for all dimensional geometry
    flag_bm_dof.resize(n_dof_total, false);
    for (unsigned int ind = 0; ind <= 3; ++ind){
	flag_bm[ind].resize(n_geometry[ind], false);
	if (ind == 3) break;
	for (unsigned int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo)
	    flag_bm[ind][ind_geo] = mesh.geometry(ind, ind_geo).boundaryMark() == 1;
    }
}

TEMPLATE_TSEM
void THIS_TSEM::calc_rhs(Vector<valuetype> &rhs, std::vector<std::vector<valuetype> > &value_f)
{
    rhs.reinit(n_dof_total);
    for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof) rhs(ind_dof) = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp);
        for (int ind_index = 0; ind_index < n_index; ++ind_index){
            valuetype cnt = 0;
            for (int p = 0; p < n_q_point[2]; ++p)
                cnt += Weight[2][p] * value_f[ind_ele][p] * basis_value_actual[p][ind_index];
            cnt *= val_volume[ind_ele];
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                rhs(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    += cnt * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        }
    }
    std::cerr << "calculate rhs\n";
}

TEMPLATE_TSEM
void THIS_TSEM::impose_zero_boundary_condition(SparseMatrix<valuetype> &sp_matrix)
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    
    // impose boundary condition
    for (int ind = 0; ind <= 2; ++ind)
        for (int ind_geometry = 0; ind_geometry < n_geometry[ind]; ++ind_geometry){
            // if (mesh.boundaryMark(ind, ind_geometry) != 1) continue;
            if (!flag_bm[ind][ind_geometry]) continue;
            int location = ind_geometry + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
            unsigned int row_begin = location_actualdof[location_geometry[location]];
            for (int i = 0; i < n_dof_geometry[ind]; ++i){
                unsigned int row = row_begin + i;
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = sp_matrix.begin(row);
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = sp_matrix.end(row);
                // for (; spm_ite != spm_end; ++spm_ite){
                //     int col = spm_ite->column();
		for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		    const unsigned int &col = col_nums[pos_whole];
                    if (col != row){
                        sp_matrix.set(row, col, 0);
                        sp_matrix.set(col, row, 0);
                    }
                }
            }
        }
    std::cerr << "impose boundary condition to matrix\n";
}

TEMPLATE_TSEM
void THIS_TSEM::impose_zero_boundary_condition(Vector<valuetype> &rhs)
{
    // impose zero boundary condition
    for (int ind = 0; ind <= 2; ++ind)
	for (int ind_geometry = 0; ind_geometry < n_geometry[ind]; ++ind_geometry){
	    if (!flag_bm[ind][ind_geometry]) continue;
	    int location = ind_geometry + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
	    int row_start = location_actualdof[location_geometry[location]];
	    for (int i = 0; i < n_dof_geometry[ind]; ++i)
		rhs(row_start + i) = 0;
	}
}

TEMPLATE_TSEM
void THIS_TSEM::impose_boundary_condition_rowOnly(SparseMatrix<valuetype> &sp_matrix)
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    
    // impose boundary condition
    for (int ind = 0; ind <= 2; ++ind)
        for (int ind_geometry = 0; ind_geometry < n_geometry[ind]; ++ind_geometry){
            if (!flag_bm[ind][ind_geometry]) continue;
            int location = ind_geometry + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
            unsigned int row_begin = location_actualdof[location_geometry[location]];
            for (int i = 0; i < n_dof_geometry[ind]; ++i){
                unsigned int row = row_begin + i;
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = sp_matrix.begin(row);
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = sp_matrix.end(row);
                // for (; spm_ite != spm_end; ++spm_ite){
                //     int col = spm_ite->column();
		for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		    const unsigned int &col = col_nums[pos_whole];
                    if (col != row)
                        sp_matrix.set(row, col, 0);
                }
            }
        }
    std::cerr << "impose boundary condition to matrix\n";
}

TEMPLATE_TSEM
void THIS_TSEM::impose_boundary_condition_rowOnly(SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &rhs, std::vector<std::vector<std::vector<valuetype> > > &value_bnd)
{
    
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    
    // impose boundary condition
    for (int ind = 0; ind <= 2; ++ind)
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
            // if (mesh.boundaryMark(ind, ind_geo) != 1) continue;
            if (!flag_bm[ind][ind_geo]) continue;
            int location = ind_geo + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
            unsigned int row_begin = location_actualdof[location_geometry[location]];
            std::vector<double> val(n_dof_geometry[ind], 0);
            // evaluate val in the same way to interpolation
            if (ind == 0) // 0 dimensional geometry, only 1 point
                val[0] = value_bnd[ind][ind_geo][0];
            if (ind == 1){ // 1 dimensional geometry
                int ind_point_s = number_node[0][ind_geo][0], ind_point_e = number_node[0][ind_geo][1];
                int location_s = location_actualdof[location_geometry[ind_point_s]], location_e = location_actualdof[location_geometry[ind_point_e]];
                double c_s = rhs(location_s) / sp_matrix.diag_element(location_s), c_e = rhs(location_e) / sp_matrix.diag_element(location_e);
                for (int l = 2; l <= M; ++l){ // dof locate on this edge
                    double count = 0;
                    for (int p = 0; p < n_q_point[0]; ++p)
                        count += Weight[0][p]
			    * (value_bnd[ind][ind_geo][p] - c_s*QPoint_Barycentric[0][p][1] - c_e*QPoint_Barycentric[0][p][0]) * basis_value_interp[0][p][l-2];
                    val[l-2] = count * -l * (2*l - 1) / (2 * (l-1));
                }
            }
            if (ind == 2){ // 2 dimensional geometry
                // traverse 2 dimensional geometry
                for (int l1 = 2; l1 <= M; ++l1)
                    for (int l2 = 1; l2 <= M-l1; ++l2){
                        int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1; // order of multiindex (l1-2, l2-1)
                        double count = 0;
                        for (int p = 0; p < n_q_point[1]; ++p){
                            double xp = QPoint_Barycentric[1][p][0], yp = QPoint_Barycentric[1][p][1], rp = QPoint_Barycentric[1][p][2];
                            double count_p = value_bnd[ind][ind_geo][p]; // the contribution from function u
                            for (int ind = 0; ind < 3; ++ind){ // substract the contribution from vertex: 0 -> 2, 1 -> 0, 2 -> 1 ((x + 2) % 3)
                                int& pos_vertex_global = location_actualdof[location_geometry[number_node[1][ind_geo][ind]]];
                                // double val_interp_vertex = rhs(pos_vertex_global) / stiff_matrix.diag_element(pos_vertex_global);
                                double val_interp_vertex = rhs(pos_vertex_global) / sp_matrix.diag_element(pos_vertex_global);
                                count_p -= val_interp_vertex * QPoint_Barycentric[1][p][(ind+2)%3];
                            }
                            int location_dof_edge[3]; // start location of dof on each edge
                            for (int ind_edge = 0; ind_edge < 3; ++ind_edge)
                                location_dof_edge[ind_edge] = location_actualdof[location_geometry[number_edge[ind_geo][ind_edge] + n_geometry[0]]];
                            std::vector<std::vector<double> > val_edgedof(3);
                            for (int ind_e = 0; ind_e < 3; ++ind_e){
                                val_edgedof[ind_e].resize(n_dof_edge);
                                for (int l = 2; l <= M; ++l){
                                    val_edgedof[ind_e][l-2] = rhs(location_dof_edge[ind_e] + l-2) / sp_matrix.diag_element(location_dof_edge[ind_e] + l-2);
                                    if (!flag_sameorder_edgeonface[ind_geo][ind_e] && l % 2 == 1)
                                        val_edgedof[ind_e][l-2] *= -1;
                                }
                            }
                            for (int l = 2; l <= M; ++l){ // substract the contribution from edge
                                count_p += 2 * (xp*yp * val_edgedof[0][l-2] * basis_value_addition[p][l-2][1] +
                                                yp*rp * val_edgedof[1][l-2] * basis_value_addition[p][l-2][1] +
                                                xp*rp * val_edgedof[2][l-2] * basis_value_addition[p][l-2][0]);
                            }
                            count += Weight[1][p] * count_p * basis_value_interp[1][p][ind_index];
                        }
                        val[ind_index] = count * -l1 * (2*l1-1) * (2*l1+2*l2-1) / (pow(2, l1) * (l1-1));
                    }
            }
	    if (ind == 2)
		 for (int i = 0; i < n_dof_geometry[ind]; ++i){
		     unsigned int row = row_begin + conversion[0][i];
		     rhs(row) = sp_matrix.diag_element(row) * val[i];
		 }
	    else
		for (int i = 0; i < n_dof_geometry[ind]; ++i){
		    unsigned int row = row_begin + i;
		    rhs(row) = sp_matrix.diag_element(row) * val[i];
		}
        }
    std::cerr << "impose boundary condition\n";
}

TEMPLATE_TSEM
void THIS_TSEM::impose_boundary_condition(SparseMatrix<valuetype> &sp_matrix, Vector<valuetype> &rhs, std::vector<std::vector<std::vector<valuetype> > > &value_bnd,
    bool flag_modify_matrix)
{
    const SparsityPattern &sp_pattern = sp_matrix.get_sparsity_pattern();
    const std::size_t *row_start = sp_pattern.get_rowstart_indices();
    const unsigned int *col_nums = sp_pattern.get_column_numbers();
    
    // impose boundary condition
    for (int ind = 0; ind <= 2; ++ind)
        for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
            // if (mesh.boundaryMark(ind, ind_geo) != 1) continue;
            if (!flag_bm[ind][ind_geo]) continue;
            int location = ind_geo + ((ind > 0) ? n_geometry[0] : 0) + ((ind > 1) ? n_geometry[1] : 0);
            unsigned int row_begin = location_actualdof[location_geometry[location]];
            std::vector<double> val(n_dof_geometry[ind], 0);
            // evaluate val in the same way to interpolation
            if (ind == 0) // 0 dimensional geometry, only 1 point
                val[0] = value_bnd[ind][ind_geo][0];
            if (ind == 1){ // 1 dimensional geometry
                int ind_point_s = number_node[0][ind_geo][0], ind_point_e = number_node[0][ind_geo][1];
                int location_s = location_actualdof[location_geometry[ind_point_s]], location_e = location_actualdof[location_geometry[ind_point_e]];
                double c_s = rhs(location_s) / sp_matrix.diag_element(location_s), c_e = rhs(location_e) / sp_matrix.diag_element(location_e);
                for (int l = 2; l <= M; ++l){ // dof locate on this edge
                    double count = 0;
                    for (int p = 0; p < n_q_point[0]; ++p)
                        count += Weight[0][p]
			    * (value_bnd[ind][ind_geo][p] - c_s*QPoint_Barycentric[0][p][1] - c_e*QPoint_Barycentric[0][p][0]) * basis_value_interp[0][p][l-2];
                    val[l-2] = count * -l * (2*l - 1) / (2 * (l-1));
                }
            }
            if (ind == 2){ // 2 dimensional geometry
                // traverse 2 dimensional geometry
                for (int l1 = 2; l1 <= M; ++l1)
                    for (int l2 = 1; l2 <= M-l1; ++l2){
                        int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1; // order of multiindex (l1-2, l2-1)
                        double count = 0;
                        for (int p = 0; p < n_q_point[1]; ++p){
                            double xp = QPoint_Barycentric[1][p][0], yp = QPoint_Barycentric[1][p][1], rp = QPoint_Barycentric[1][p][2];
                            double count_p = value_bnd[ind][ind_geo][p]; // the contribution from function u
                            for (int ind = 0; ind < 3; ++ind){ // substract the contribution from vertex: 0 -> 2, 1 -> 0, 2 -> 1 ((x + 2) % 3)
                                int& pos_vertex_global = location_actualdof[location_geometry[number_node[1][ind_geo][ind]]];
                                // double val_interp_vertex = rhs(pos_vertex_global) / stiff_matrix.diag_element(pos_vertex_global);
                                double val_interp_vertex = rhs(pos_vertex_global) / sp_matrix.diag_element(pos_vertex_global);
                                count_p -= val_interp_vertex * QPoint_Barycentric[1][p][(ind+2)%3];
                            }
                            int location_dof_edge[3]; // start location of dof on each edge
                            for (int ind_edge = 0; ind_edge < 3; ++ind_edge)
                                location_dof_edge[ind_edge] = location_actualdof[location_geometry[number_edge[ind_geo][ind_edge] + n_geometry[0]]];
                            std::vector<std::vector<double> > val_edgedof(3);
                            for (int ind_e = 0; ind_e < 3; ++ind_e){
                                val_edgedof[ind_e].resize(n_dof_edge);
                                for (int l = 2; l <= M; ++l){
                                    val_edgedof[ind_e][l-2] = rhs(location_dof_edge[ind_e] + l-2) / sp_matrix.diag_element(location_dof_edge[ind_e] + l-2);
                                    if (!flag_sameorder_edgeonface[ind_geo][ind_e] && l % 2 == 1)
                                        val_edgedof[ind_e][l-2] *= -1;
                                }
                            }
                            for (int l = 2; l <= M; ++l){ // substract the contribution from edge
                                count_p += 2 * (xp*yp * val_edgedof[0][l-2] * basis_value_addition[p][l-2][1] +
                                                yp*rp * val_edgedof[1][l-2] * basis_value_addition[p][l-2][1] +
                                                xp*rp * val_edgedof[2][l-2] * basis_value_addition[p][l-2][0]);
                            }
                            count += Weight[1][p] * count_p * basis_value_interp[1][p][ind_index];
                        }
                        val[ind_index] = count * -l1 * (2*l1-1) * (2*l1+2*l2-1) / (pow(2, l1) * (l1-1));
                    }
            }
            for (int i = 0; i < n_dof_geometry[ind]; ++i){
                unsigned int row = row_begin + (ind == 2 ? conversion[0][i] : i);
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_ite = sp_matrix.begin(row);
                // SparseMatrix<VALUETYPE_ITERATOR_TSEM>::iterator spm_end = sp_matrix.end(row);
                // for (; spm_ite != spm_end; ++spm_ite){
                //     int col = spm_ite->column();
		for (unsigned int pos_whole = row_start[row]; pos_whole < row_start[row+1]; ++pos_whole){
		    const unsigned int &col = col_nums[pos_whole];
                    if (col == row)
                        // rhs(row) = spm_ite->value() * val[i];
			// rhs(row) = sp_matrix.global_entry(pos_whole) * val[i];
			rhs(row) = sp_matrix.diag_element(row) * val[i];
                    else{
			// if (!flag_bm_dof[col])
			    rhs(col) -= sp_matrix.el(col, row) * val[i];
			if (flag_modify_matrix){
			    sp_matrix.set(row, col, 0);
			    sp_matrix.set(col, row, 0);
			}
                    }
                }
            }
        }
    std::cerr << "impose boundary condition\n";
}

TEMPLATE_TSEM
void THIS_TSEM::calc_coef_onElement(Vector<valuetype> &src, std::vector<valuetype> &dst, unsigned int ind_ele)
{ // calculate coeffcieint of sem solution on ind_ele-th element from source src to vector dst, suppose dst has size n_index
    // dst.resize(n_index);
    for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	dst[ind_index] = 0;
    for (int ind_index = 0; ind_index < n_index; ++ind_index)
	for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
	    dst[ind_index] += src(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
		* transform_val_global2local[ind_ele][ind_index][ind_nnz];
}

TEMPLATE_TSEM
void THIS_TSEM::calc_val_qp_onElement(Vector<valuetype> &src, std::vector<valuetype> &dst, unsigned int ind_ele)
{ // calculate value on quadrature point for sem solution on ind_ele-th element from source src to vector dst, suppose dst has size n_q_point[2]
    std::vector<valuetype> val_coef(n_index);
    calc_coef_onElement(src, val_coef, ind_ele);
    for (unsigned int p = 0; p < n_q_point[2]; ++p)
	dst[p] = 0;
    for (unsigned int p = 0; p < n_q_point[2]; ++p)
	for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	    dst[p] += val_coef[ind_index] * basis_value_actual[p][ind_index];
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_error(Vector<valuetype> &sol, std::vector<std::vector<valuetype> > &val_exc)
{
    // for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
    // 	std::cerr << sol(ind_dof) << ' ';
    // std::cerr << '\n';
    valuetype err_l2 = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef[ind_index] += sol(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
	// for (int ind_index = 0; ind_index < n_index; ++ind_index)
	//     std::cerr << val_coef[ind_index] << ' ';
	// std::cerr << '\n';

	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_sol = 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index)
                val_sol += val_coef[ind_index] * basis_value_actual[p][ind_index];
            // count the contribution
            count += Weight[2][p] * pow(val_sol - val_exc[ind_ele][p], 2);
	    // valuetype dif = fabs(val_sol - val_exc[ind_ele][p]);
	    // if (dif > 1.0e-8)
	    // 	std::cerr << "val_sol = " << val_sol << ", val_exc = " << val_exc[ind_ele][p] << ", dif = " << dif << ", count = " << count << ", Weight[2][p] = " << Weight[2][p] << '\n';
        }
        err_l2 += count * val_volume[ind_ele];
	// std::cerr << "ind_ele = " << ind_ele << ", count = " << count << '\n';
    }
    err_l2 = sqrt(err_l2);
    // std::cerr << "err_l2 = " << err_l2 << '\n';
    
    return err_l2;
}

TEMPLATE_TSEM
valuetype THIS_TSEM:: calc_l2_error_gradient(RegularMesh<3> &mesh,
					     Vector<valuetype> &sol, std::vector<std::vector<std::vector<valuetype> > > &val_g_exc)
{
    valuetype err = (valuetype) 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
	// valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef[ind_index] += sol(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];

	// prepare coordinates of element vertices
	valuetype xj_k[3][4];
	for (int k = 0; k < 4; ++k){
	    int ind_node = index_geometry_onelement[ind_ele][0][k];
	    for (int j = 0; j < 3; ++j){
		xj_k[j][k] = mesh.point(ind_node)[j];
		// std::cerr << "xj_k[" << j << "][" << k << "] = " << xj_k[j][k] << '\n';
	    }
	}
	valuetype d_jk[3][3];
	for (int j = 0; j < 3; ++j)
	    for (int k = 0; k < 3; ++k){
		d_jk[j][k] = xj_k[j][k+1] - xj_k[j][0];
		// std::cerr << "d_jk[" << j << "][" << k << "] = xj_k[" << j << "][" << k+1 << "] - xj_k[" << j << "][0] = " << xj_k[j][k+1] << "\t- " << xj_k[j][0] << " = " << d_jk[j][k] << '\n';
	    }
	valuetype val_V = d_jk[0][0]*d_jk[1][1]*d_jk[2][2] + d_jk[0][1]*d_jk[1][2]*d_jk[2][0] + d_jk[0][2]*d_jk[1][0]*d_jk[2][1]
	                - d_jk[0][0]*d_jk[1][2]*d_jk[2][1] - d_jk[0][1]*d_jk[1][0]*d_jk[2][2] - d_jk[0][2]*d_jk[1][1]*d_jk[2][0];
	// std::cerr << "ind_ele = " << ind_ele << ", val_V = " << val_V << '\n';
	// break;
	if (fabs(fabs(val_V) - 6*val_volume[ind_ele])/fabs(val_V) > tol_zero) std::cerr << "find different volume, ind_ele = " << ind_ele << ", volume = " << val_volume[ind_ele] << ", val_V = " << fabs(val_V/6) << ", dif = " << fabs(fabs(val_volume[ind_ele] - fabs(val_V/6))) << '\n';
		
	valuetype count = (valuetype) 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // calculate \partial x_j/\partial \hat{x}_k
	    valuetype parxj_parxk[3][3];
	    parxj_parxk[0][0] = (d_jk[1][1]*d_jk[2][2] - d_jk[1][2]*d_jk[2][1]) / val_V;
	    parxj_parxk[0][1] = (d_jk[0][2]*d_jk[2][1] - d_jk[0][1]*d_jk[2][2]) / val_V;
	    parxj_parxk[0][2] = (d_jk[0][1]*d_jk[1][2] - d_jk[0][2]*d_jk[1][1]) / val_V;
	    parxj_parxk[1][0] = (d_jk[1][2]*d_jk[2][0] - d_jk[1][0]*d_jk[2][2]) / val_V;
	    parxj_parxk[1][1] = (d_jk[0][0]*d_jk[2][2] - d_jk[0][2]*d_jk[2][0]) / val_V;
	    parxj_parxk[1][2] = (d_jk[0][2]*d_jk[1][0] - d_jk[0][0]*d_jk[1][2]) / val_V;
	    parxj_parxk[2][0] = (d_jk[1][0]*d_jk[2][1] - d_jk[1][1]*d_jk[2][0]) / val_V;
	    parxj_parxk[2][1] = (d_jk[0][1]*d_jk[2][0] - d_jk[0][0]*d_jk[2][1]) / val_V;
	    parxj_parxk[2][2] = (d_jk[0][0]*d_jk[1][1] - d_jk[0][1]*d_jk[1][0]) / val_V;
            // recover the value of solution
	    valuetype val_g_sol[3] = {0, 0, 0};
	    for (int ind = 0; ind < 3; ++ind)
		for (int ind_index = 0; ind_index < n_index; ++ind_index)
		    for (int indt = 0; indt < 3; ++indt)
			val_g_sol[ind] += parxj_parxk[indt][ind] * val_coef[ind_index] * basis_gradient_actual[p][ind_index][indt];
            // count the contribution
	    valuetype cnt_tmp = (valuetype) 0;
	    for (int ind = 0; ind < 3; ++ind)
		cnt_tmp += pow(val_g_sol[ind] - val_g_exc[ind_ele][p][ind], 2);
            count += Weight[2][p] * cnt_tmp;
	}
	err += count * val_volume[ind_ele];
    }
    
    err = sqrt(err);
    return err;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_density_difference(Vector<valuetype> &u, std::vector<std::vector<valuetype> > &v)
{
    valuetype dif_l2 = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef_u(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_u[ind_index] += u(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
       
	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_u = (valuetype) 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index)
                val_u += val_coef_u[ind_index] * basis_value_actual[p][ind_index];
            // count the contribution
            count += Weight[2][p] * pow(pow(val_u,2) - v[ind_ele][p], 2);
        }
        dif_l2 += count * val_volume[ind_ele];
    }
    dif_l2 = sqrt(dif_l2);
    
    return dif_l2;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_density_difference(Vector<valuetype> &u, Vector<valuetype> &v)
{
    valuetype dif_l2 = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef_u(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_u[ind_index] += u(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        std::vector<valuetype> val_coef_v(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_v[ind_index] += v(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];

	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_u = (valuetype) 0, val_v = (valuetype) 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index){
                val_u += val_coef_u[ind_index] * basis_value_actual[p][ind_index];
                val_v += val_coef_v[ind_index] * basis_value_actual[p][ind_index];
	    }
            // count the contribution
            count += Weight[2][p] * pow(pow(val_u,2) - pow(val_v,2), 2);
        }
        dif_l2 += count * val_volume[ind_ele];
    }
    dif_l2 = sqrt(dif_l2);
    
    return dif_l2;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_difference(Vector<valuetype> &u, Vector<valuetype> &v)
{
    valuetype dif_l2 = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef_u(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_u[ind_index] += u(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        std::vector<valuetype> val_coef_v(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_v[ind_index] += v(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];

	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_u = (valuetype) 0, val_v = (valuetype) 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index){
                val_u += val_coef_u[ind_index] * basis_value_actual[p][ind_index];
                val_v += val_coef_v[ind_index] * basis_value_actual[p][ind_index];
	    }
            // count the contribution
            count += Weight[2][p] * pow(val_u - val_v, 2);
        }
        dif_l2 += count * val_volume[ind_ele];
    }
    dif_l2 = sqrt(dif_l2);
    
    return dif_l2;
}


TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_difference(Vector<valuetype> &u, Vector<valuetype> &v, int polynomial_order, int flag)
{ // flag: 0 - each, 1 - <= polynomial order
    valuetype dif_l2 = 0;
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef_u(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_u[ind_index] += u(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        std::vector<valuetype> val_coef_v(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef_v[ind_index] += v(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
	for (int ind_index = 0; ind_index < n_index; ++ind_index){
	    Multiindex<3> index_now = correspondence.number2index(ind_index);
	    if (flag == 0 && index_now.sum() == polynomial_order) continue;
	    if (flag == 1 && index_now.sum() <=  polynomial_order) continue;
	    val_coef_u[ind_index] = 0;
	    val_coef_v[ind_index] = 0;
	}

	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_u = (valuetype) 0, val_v = (valuetype) 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index){
                val_u += val_coef_u[ind_index] * basis_value_actual[p][ind_index];
                val_v += val_coef_v[ind_index] * basis_value_actual[p][ind_index];
	    }
            // count the contribution
            count += Weight[2][p] * pow(val_u - val_v, 2);
        }
        dif_l2 += count * val_volume[ind_ele];
    }
    dif_l2 = sqrt(dif_l2);
    
    return dif_l2;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_l2_error_component(Vector<valuetype> &u, int polynomial_order_less, int polynomial_order_more)
{ // calcultate L2 error between polynomial_order_less and polynomial_order_large,
  // suppose polynomial_order_less < polynomial_order_more <= M
    valuetype err_l2 = 0;
    std::vector<bool> flag_less(n_index, true);
    std::vector<bool> flag_more(n_index, true);
    for (int ind_index = 0; ind_index < n_index; ++ind_index){
	Multiindex<3> index_now = correspondence.number2index(ind_index);
	if (index_now.sum() > polynomial_order_less) flag_less[ind_index] = false;
	if (index_now.sum() > polynomial_order_more) flag_more[ind_index] = false;
    }
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	// recover local coefficient on this element
        std::vector<valuetype> val_coef(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index)
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef[ind_index] += u(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];

	valuetype count = 0;
        for (int p = 0; p < n_q_point[2]; ++p){
            // recover the value of solution
            valuetype val_less = (valuetype) 0, val_more = (valuetype) 0;
            for (int ind_index = 0; ind_index < n_index; ++ind_index){
		if (flag_less[ind_index]) val_less += val_coef[ind_index] * basis_value_actual[p][ind_index];
		if (flag_more[ind_index]) val_more += val_coef[ind_index] * basis_value_actual[p][ind_index];
	    }
            // count the contribution
            count += Weight[2][p] * pow(val_less - val_more, 2);
        }
        err_l2 += count * val_volume[ind_ele];
    }
    err_l2 = sqrt(err_l2);
    
    return err_l2;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_density_l2_difference(std::vector<Vector<valuetype> > &u, std::vector<Vector<valuetype> > &v, std::vector<valuetype> &n_occupation)
{
    unsigned int n_orbital = u.size();
    valuetype dif_l2 = 0, count;
    std::vector<valuetype> val_qp_u(n_q_point[2]);
    std::vector<valuetype> val_qp_v(n_q_point[2]);
    std::vector<valuetype> val_dif(n_q_point[2]);
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // valuetype volume = fem_space.element(ind_ele).templateElement().volume();
        // AFEPack::Point<3> point_tmp;
        // for (int ind = 0; ind < 3; ++ind) point_tmp[ind] = 1.0/3;
        // valuetype jacobian = fem_space.element(ind_ele).local_to_global_jacobian(point_tmp); // the determinant of jacobian is fixed

	for (unsigned int p = 0; p < n_q_point[2]; ++p)
	    val_dif[p] = 0;

	for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    calc_val_qp_onElement(u[ind_orbital], val_qp_u, ind_ele);
	    calc_val_qp_onElement(v[ind_orbital], val_qp_v, ind_ele);
	    for (unsigned int p = 0; p < n_q_point[2]; ++p)
		val_dif[p] += n_occupation[ind_orbital] * (pow(val_qp_u[p],2) - pow(val_qp_v[p],2));
	}

	count = 0;
	for (unsigned int p = 0; p < n_q_point[2]; ++p)
	    count += Weight[2][p] * pow(val_dif[p], 2);

	dif_l2 += count * val_volume[ind_ele];
    }
    dif_l2 = sqrt(dif_l2);
    
    return dif_l2;
}


TEMPLATE_TSEM
void THIS_TSEM::calc_interpolation(Vector<valuetype> &interp, std::vector<std::vector<std::vector<valuetype> > > &val_interp)
{
    interp.reinit(n_dof_total);
    
    // traverse 0 dimensional geometry, correspond to vertex in fem dof
    for (int ind_point = 0; ind_point < n_geometry[0]; ++ind_point)
        interp(location_actualdof[location_geometry[ind_point]]) = val_interp[0][ind_point][0];

    // traverse 1 dimensional geometry
    for (int ind_edge = 0; ind_edge < n_geometry[1]; ++ind_edge){
	int ind_point_s = number_node[0][ind_edge][0], ind_point_e = number_node[0][ind_edge][1];
        double c_s = interp(location_actualdof[location_geometry[ind_point_s]]), c_e = interp(location_actualdof[location_geometry[ind_point_e]]);
        for (int l = 2; l <= M; ++l){ // dof locate on this edge
            int location_dof = location_actualdof[location_geometry[ind_edge+n_geometry[0]]] + l-2; // position of this dof
            valuetype count = 0;
            for (int p = 0; p < n_q_point[0]; ++p)
                count += Weight[0][p] * (val_interp[1][ind_edge][p] - c_s*QPoint_Barycentric[0][p][1] - c_e*QPoint_Barycentric[0][p][0]) * basis_value_interp[0][p][l-2];
            interp(location_dof) = count * -l * (2*l - 1) / (2 * (l-1));
        }
    }
    
    // traverse 2 dimensional geometry
    for (int ind_face = 0; ind_face < n_geometry[2]; ++ind_face){
        for (int l1 = 2; l1 <= M; ++l1)
            for (int l2 = 1; l2 <= M-l1; ++l2){
                int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1; // order of multiindex (l1-2, l2-1)
                int location_dof = location_actualdof[location_geometry[ind_face+n_geometry[0]+n_geometry[1]]] + ind_index;
                valuetype count = 0;
                for (int p = 0; p < n_q_point[1]; ++p){
                    valuetype xp = QPoint_Barycentric[1][p][0], yp = QPoint_Barycentric[1][p][1], rp = QPoint_Barycentric[1][p][2];
                    valuetype count_p = val_interp[2][ind_face][p]; // the contribution from function u
                    for (int ind = 0; ind < 3; ++ind) // substract the contribution from vertex: 0 -> 2, 1 -> 0, 2 -> 1 ((x + 2) % 3)
                        count_p -= interp(location_actualdof[location_geometry[number_node[1][ind_face][ind]]]) * QPoint_Barycentric[1][p][(ind+2)%3];
                    int location_dof_edge[3]; // start location of dof on each edge
                    for (int ind_edge = 0; ind_edge < 3; ++ind_edge)
                        location_dof_edge[ind_edge] = location_actualdof[location_geometry[number_edge[ind_face][ind_edge] + n_geometry[0]]];
                    std::vector<std::vector<valuetype> > val_edgedof(3);
                    for (int ind_e = 0; ind_e < 3; ++ind_e){
                        val_edgedof[ind_e].resize(n_dof_edge);
                        for (int l = 2; l <= M; ++l){
                            val_edgedof[ind_e][l-2] = interp(location_dof_edge[ind_e] + l-2);
                            if (!flag_sameorder_edgeonface[ind_face][ind_e] && l % 2 == 1)
                                val_edgedof[ind_e][l-2] *= -1;
                        }
                    }
                    for (int l = 2; l <= M; ++l){ // substract the contribution from edge
                        count_p += 2 * (xp*yp * val_edgedof[0][l-2] * basis_value_addition[p][l-2][1] +
                                        yp*rp * val_edgedof[1][l-2] * basis_value_addition[p][l-2][1] +
                                        xp*rp * val_edgedof[2][l-2] * basis_value_addition[p][l-2][0]);
                    }
                    count += Weight[1][p] * count_p * basis_value_interp[1][p][ind_index];
                }
                interp(location_dof) = count * -l1 * (2*l1-1) * (2*l1+2*l2-1) / (pow(2, l1) * (l1-1));
            }
    }
    
    // traverse fem element, assign value for interior dof
    for (int ind_ele = 0; ind_ele < n_element; ++ind_ele){
        // get local coefficients for vertex, edge and face
        std::vector<valuetype> val_coef(n_index, 0);
        for (int ind_index = 0; ind_index < n_index; ++ind_index){
            Multiindex<3> index_now = correspondence.number2index(ind_index);
            if (index_now.index[0] >= 2 && index_now.index[1] >= 1 && index_now.index[2] >= 1)
                continue;
            for (int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
                val_coef[ind_index] += interp(transform_ind_global2local[ind_ele][ind_index][ind_nnz])
                    * transform_val_global2local[ind_ele][ind_index][ind_nnz];
        }
        for (int l1 = 2; l1 <= M; ++l1)
            for (int l2 = 1; l2 <= M-l1; ++l2)
                for (int l3 = 1; l3 <= M-l1-l2; ++l3){
                    int ind_index = correspondence.index2number(Unitary_Multiindex[0] * (l1-2) + Unitary_Multiindex[1] * (l2-1) + Unitary_Multiindex[2] * (l3-1));
                    int location_dof = location_actualdof[location_geometry[ind_ele + n_geometry[0] + n_geometry[1] + n_geometry[2]]] + ind_index;
                    valuetype count = 0;
                    for (int p = 0; p < n_q_point[2]; ++p){
                        valuetype count_p = val_interp[3][ind_ele][p];
                        for (int j = 0; j < n_index; ++j){ // traverse multiindex for vertex, edge and face
                            Multiindex<3> index_tmp = correspondence.number2index(j);
                            if (index_tmp.index[0] >= 2 && index_tmp.index[1] >= 1 && index_tmp.index[2] >= 1) // the interior model function are orthogonal
                                continue;
                            count_p -= val_coef[j] * basis_value_actual[p][j];
                        }
                        count += Weight[2][p] * count_p *  basis_value_interp[2][p][ind_index];
                    }
                    // for interior dof, global coefficient is exactly local one
                    interp(location_dof) = count * -l1 * (2*l1-1) * (2*l1+2*l2-1) * (2*l1+2*l2+2*l3-1) / (pow(2, 2*l1+l2-5) * (l1-1) * 6);
                }
    }
    std::cerr << "interpolation exact solution\n";
}

TEMPLATE_TSEM
void THIS_TSEM::read_coef(Vector<valuetype> &dst, std::string filename)
{
    dst.reinit(n_dof_total);
    std::ifstream input_coef(filename);
    int M_interp, n_dof_total_interp;
    input_coef >> M_interp;
    input_coef >> n_dof_total_interp;
    // construct pointer
    std::vector<int> n_dof_geometry_interp(4); // number of dof location on each dimensional geoemtry for interpolation
    for (int ind = 0; ind <= 3; ++ind) // number of dof on ind dimensional geometry
	n_dof_geometry_interp[ind] = calc_binomial_coefficient(M_interp-1, ind);
    std::vector<int> location_actualdof_interp(n_geometry_total); // start index of geometry in actual discretized matrix for interpolation;
    location_actualdof_interp[0] = 0;
    for (int i = 1; i < n_geometry_total; ++i)
	location_actualdof_interp[i] = location_actualdof_interp[i-1] + n_dof_geometry_interp[geometry_dimension[i-1]];
    // read dof info
    Vector<valuetype> src(n_dof_total_interp);
    for (int i = 0; i < n_dof_total_interp; ++i)
	input_coef >> src(i);
    // initialize
    for (int i = 0; i < n_dof_total; ++i)
	dst(i) = 0;
    // interpolate
    for (int ind = 0; ind <= 3; ++ind)
	for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
	    int location = ind_geo + ((ind >= 1) ? n_geometry[0] : 0) + ((ind >= 2) ? n_geometry[1] : 0) + ((ind >= 3) ? n_geometry[2] : 0);
	    int position =        location_actualdof[location_geometry[location]];
	    int position_interp = location_actualdof_interp[location_geometry[location]];
	    for (int ind_index = 0; ind_index < n_dof_geometry_interp[ind]; ++ind_index)
		dst(position + ind_index) = src(position_interp + ind_index);
	}
    input_coef.close();
}

TEMPLATE_TSEM
void THIS_TSEM::read_coef(std::vector<Vector<valuetype> > &dst, std::string filename)
{ // suppose the length of dst and its element match the info in filename
    std::ifstream input_coef(filename);
    int M_interp, n_dof_total_interp;
    input_coef >> M_interp;
    input_coef >> n_dof_total_interp;
    // construct pointer
    std::vector<int> n_dof_geometry_interp(4); // number of dof location on each dimensional geoemtry for interpolation
    for (int ind = 0; ind <= 3; ++ind) // number of dof on ind dimensional geometry
	n_dof_geometry_interp[ind] = calc_binomial_coefficient(M_interp-1, ind);
    std::vector<int> location_actualdof_interp(n_geometry_total); // start index of geometry in actual discretized matrix for interpolation;
    location_actualdof_interp[0] = 0;
    for (int i = 1; i < n_geometry_total; ++i)
	location_actualdof_interp[i] = location_actualdof_interp[i-1] + n_dof_geometry_interp[geometry_dimension[i-1]];
    
    for (int ind_vec = 0; ind_vec < dst.size(); ++ind_vec){
	// read dof info
	Vector<valuetype> src(n_dof_total_interp);
	for (int i = 0; i < n_dof_total_interp; ++i)
	    input_coef >> src(i);
	// initialize
	for (int i = 0; i < n_dof_total; ++i)
	    dst[ind_vec](i) = 0;
	// interpolate
	for (int ind = 0; ind <= 3; ++ind)
	    for (int ind_geo = 0; ind_geo < n_geometry[ind]; ++ind_geo){
		int location = ind_geo + ((ind >= 1) ? n_geometry[0] : 0) + ((ind >= 2) ? n_geometry[1] : 0) + ((ind >= 3) ? n_geometry[2] : 0);
		int position =        location_actualdof[location_geometry[location]];
		int position_interp = location_actualdof_interp[location_geometry[location]];
		for (int ind_index = 0; ind_index < n_dof_geometry_interp[ind]; ++ind_index)
		    dst[ind_vec](position + ind_index) = src(position_interp + ind_index);
	    }
    }
    input_coef.close();
}

TEMPLATE_TSEM
void THIS_TSEM::write_coef(Vector<valuetype> &src, std::string filename)
{
    std::ofstream output_coef(filename);
    output_coef << M << '\t' << n_dof_total << '\n';
    for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
	output_coef << std::setprecision(14) << src(ind_dof) << '\n';
    output_coef.close();
}

TEMPLATE_TSEM
void THIS_TSEM::write_coef(std::vector<Vector<valuetype> > &src, std::string filename)
{
    std::ofstream output_coef(filename);
    output_coef << M << '\t' << n_dof_total << '\n';
    for (int ind_vec = 0; ind_vec < src.size(); ++ind_vec)
	for (int ind_dof = 0; ind_dof < n_dof_total; ++ind_dof)
	    output_coef << std::setprecision(14) << src[ind_vec](ind_dof) << '\n';
    output_coef.close();
}

TEMPLATE_TSEM
void THIS_TSEM::calc_sum_dof(std::vector<unsigned int> &sum_dof)
{
    // 0-d
    for (unsigned int ind_geo = 0; ind_geo < n_geometry[0]; ++ind_geo)
	sum_dof[location_actualdof[location_geometry[ind_geo]]] = 1;
    // 1-d
    for (unsigned int ind_geo = 0; ind_geo < n_geometry[1]; ++ind_geo){
	int &pos = location_actualdof[location_geometry[ind_geo + n_geometry[0]]];
	for (int l = 2; l <= M; ++l)
	    sum_dof[pos + l-2] = l;
    }
    // 2-d
    for (unsigned int ind_geo = 0; ind_geo < n_geometry[2]; ++ind_geo){
	int &pos = location_actualdof[location_geometry[ind_geo + n_geometry[0] + n_geometry[1]]];
	for (unsigned int l1 = 2; l1 <= M; ++l1)
	    for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
		int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
		sum_dof[pos + ind_index] = l1 + l2;
	    }
    }
    // 3-d
    for (unsigned int ind_geo = 0; ind_geo < n_geometry[3]; ++ind_geo){
	int &pos = location_actualdof[location_geometry[ind_geo + n_geometry[0] + n_geometry[1] + n_geometry[2]]];
	for (unsigned int l1 = 2; l1 <= M; ++l1)
	    for (unsigned int l2 = 1; l2 <= M-l1; ++l2)
		for (unsigned int l3 = 1; l3 <= M-l1-l2; ++l3){
		    Multiindex<3> index_now = Unitary_Multiindex[0]*(l1-2) + Unitary_Multiindex[1]*(l2-1) + Unitary_Multiindex[2]*(l3-1);
		    sum_dof[pos + correspondence.index2number(index_now)] = l1 + l2 + l3;
		}
    }
}

TEMPLATE_TSEM
void THIS_TSEM::calc_val_qpoint(Vector<valuetype> &src, int ind_element, std::vector<valuetype> &dst)
{ // recover function value on ind_element-th element in fem_space, generate dst from whole Vector src
    std::vector<valuetype> coef_local(n_index, 0);
    for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	for (unsigned int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_element][ind_index]; ++ind_nnz)
	    coef_local[ind_index] += src(transform_ind_global2local[ind_element][ind_index][ind_nnz])
		* transform_val_global2local[ind_element][ind_index][ind_nnz];

    // dst.resize(n_q_point[2]);
    for (unsigned int p = 0; p < n_q_point[2]; ++p){
	valuetype val_tmp = 0;
	for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	    val_tmp += coef_local[ind_index] * basis_value_actual[p][ind_index];
	dst[p] = val_tmp;
    }
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_val_inElement(Correspondence<3> &correspondence, RegularMesh<3> &mesh, int ind_ele, AFEPack::Point<3> &pos, std::vector<Vector<valuetype> > &psi)
{ // calculate the value of point with coordinate pos on ind_ele-th element
    int n_orbital = psi.size();
    // vertex
    for (unsigned int ind_p = 0; ind_p < 4; ++ind_p){
	int &ind_point = index_geometry_onelement[ind_ele][0][ind_p];
	if (calc_length_line(pos, mesh.point(ind_point)) < tol_zero){
	    valuetype rho_local = 0;
	    for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
		valuetype n_occupation = 2.0;
		int &loc = location_actualdof[location_geometry[ind_point]];
		rho_local += n_occupation * pow(psi[ind_orbital](loc), 2);
	    }
	    return rho_local;
	}
    }
    // edge
    for (unsigned int ind_e = 0; ind_e < 6; ++ind_e){
	int &ind_edge = index_geometry_onelement[ind_ele][1][ind_e];
	int &ind_point0 = number_node[0][ind_edge][0];
	int &ind_point1 = number_node[0][ind_edge][1];
	valuetype dis = calc_length_line(mesh.point(ind_point0), mesh.point(ind_point1));
	valuetype dis0 = calc_length_line(pos, mesh.point(ind_point0));
	valuetype dis1 = calc_length_line(pos, mesh.point(ind_point1));
	if (fabs(dis0+dis1 - dis) > tol_zero) continue;
	int &loc0 = location_actualdof[location_geometry[ind_point0]];
	int &loc1 = location_actualdof[location_geometry[ind_point1]];
	int &loc = location_actualdof[location_geometry[n_geometry[0] + ind_edge]];
	
	valuetype x = dis0 / dis, r = 1 - x;
	valuetype xi = 2 * x - 1;
	valuetype Jxi[M+1];
	Jxi[0] = 1; Jxi[1] = xi;
	for (unsigned int l1 = 1; l1 < M; ++l1)
	    Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
		/ calc_coefficient_a(-1, -1, 1, l1);
	
	valuetype rho_local = 0;
	for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    valuetype n_occupation = 2.0;
	    valuetype psi_local = psi[ind_orbital](loc0)*r + psi[ind_orbital](loc1)*x;
	    for (unsigned int l1 = 2; l1 <= M; ++l1)
		psi_local += 2 * Jxi[l1] * psi[ind_orbital](loc + l1-2);
	    rho_local += n_occupation * pow(psi_local, 2);
	}
	return rho_local;
    }
    // face
    for (unsigned int ind_f = 0; ind_f < 4; ++ind_f){
	int &ind_face = index_geometry_onelement[ind_ele][2][ind_f];
	int &ind_point0 = number_node[1][ind_face][0];
	int &ind_point1 = number_node[1][ind_face][1];
	int &ind_point2 = number_node[1][ind_face][2];
	valuetype area = calc_area_triangle(mesh.point(ind_point0), mesh.point(ind_point1), mesh.point(ind_point2));
	valuetype area0 = calc_area_triangle(pos, mesh.point(ind_point1), mesh.point(ind_point2));
	valuetype area1 = calc_area_triangle(mesh.point(ind_point0), pos, mesh.point(ind_point2));
	valuetype area2 = calc_area_triangle(mesh.point(ind_point0), mesh.point(ind_point1), pos);
	if (fabs(area0+area1+area2 - area) > tol_zero) continue;
	int &ind_edge0 = number_edge[ind_face][0];
	int &ind_edge1 = number_edge[ind_face][1];
	int &ind_edge2 = number_edge[ind_face][2];
	int &loc = location_actualdof[location_geometry[n_geometry[0] + n_geometry[1] + ind_face]];
	int &locp0 = location_actualdof[location_geometry[ind_point0]];
	int &locp1 = location_actualdof[location_geometry[ind_point1]];
	int &locp2 = location_actualdof[location_geometry[ind_point2]];
	int &loce0 = location_actualdof[location_geometry[n_geometry[0] + ind_edge0]];
	int &loce1 = location_actualdof[location_geometry[n_geometry[0] + ind_edge1]];
	int &loce2 = location_actualdof[location_geometry[n_geometry[0] + ind_edge2]];
	
	std::vector<valuetype> bas_val(n_dof_geometry[2], 0);
	valuetype x = area1 / area, y = area2 / area, r = 1 - x - y;
	valuetype xi = 2*x/(1-y)-1, eta = 2*y-1;
	valuetype Jxi[M+1], Jeta[M+1];
	Jxi[0] = Jeta[0] = 1; Jxi[1] = xi;
	for (unsigned int l1 = 1; l1 < M; ++l1)
	    Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
		/ calc_coefficient_a(-1, -1, 1, l1);
	for (unsigned int l1 = 2; l1 <= M-1; ++l1){
	    int aph2 = 2 * l1 - 1;
	    Jeta[1]  = calc_generalized_jacobi_polynomial( aph2, -1, 1, eta);
	    for (unsigned int l2 = 1; l2 < M-l1; ++l2)
		Jeta[l2+1]  = ((eta - calc_coefficient_a( aph2, -1, 2, l2)) *  Jeta[l2] - calc_coefficient_a( aph2, -1, 3, l2) *  Jeta[l2-1])
		    / calc_coefficient_a( aph2, -1, 1, l2);
	    for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
		unsigned int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
		bas_val[ind_index] = 2 * pow(1-y, l1) * Jxi[l1] * Jeta[l2];
	    }
	}
	std::vector<valuetype> bas_val_add_xi( n_dof_geometry[1], 0);
	std::vector<valuetype> bas_val_add_eta(n_dof_geometry[1], 0);
	Jxi[0] = Jeta[0] = 1;
	Jxi[1]  = calc_generalized_jacobi_polynomial(1, 1, 1, xi);
	Jeta[1] = calc_generalized_jacobi_polynomial(1, 1, 1, eta);
	for (unsigned int l = 1; l < M; ++l){
	    Jxi[ l+1] = ((xi  - calc_coefficient_a(1, 1, 2, l)) *  Jxi[l] - calc_coefficient_a(1, 1, 3, l) *  Jxi[l-1]) / calc_coefficient_a(1, 1, 1, l);
	    Jeta[l+1] = ((eta - calc_coefficient_a(1, 1, 2, l)) * Jeta[l] - calc_coefficient_a(1, 1, 3, l) * Jeta[l-1]) / calc_coefficient_a(1, 1, 1, l);
	}
	for (unsigned int l = 0; l < n_dof_geometry[1]; ++l){
	    bas_val_add_xi[l]  = Jxi[l] * pow(1-y, l);
	    bas_val_add_eta[l] = Jeta[l];
	}
	
	valuetype rho_local = 0;
	for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    valuetype n_occupation = 2.0;
	    valuetype psi_local;
	    psi_local = psi[ind_orbital](locp0) * r + psi[ind_orbital](locp1) * x + psi[ind_orbital](locp2) * y;
	    for (unsigned int l = 0; l < n_dof_geometry[1]; ++l)
		psi_local -= 2 * (x*y * psi[ind_orbital](loce0+l) * bas_val_add_eta[l] +
				  y*r * psi[ind_orbital](loce1+l) * bas_val_add_eta[l] +
				  x*r * psi[ind_orbital](loce2+l) * bas_val_add_xi[l]);
	    for (unsigned int l1 = 2; l1 <= M; ++l1)
		for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
		    unsigned int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
		    psi_local += bas_val[ind_index] * psi[ind_orbital](loc + ind_index);
		}
	    rho_local += n_occupation * pow(psi_local, 2);
	}
	return rho_local;
    }
    // interior
    int &ind_v0 = index_geometry_onelement[ind_ele][0][0];
    int &ind_v1 = index_geometry_onelement[ind_ele][0][1];
    int &ind_v2 = index_geometry_onelement[ind_ele][0][2];
    int &ind_v3 = index_geometry_onelement[ind_ele][0][3];
    valuetype volume = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), mesh.point(ind_v2), mesh.point(ind_v3));
    valuetype volume0 = calc_volume_tetrahedron(pos, mesh.point(ind_v1), mesh.point(ind_v2), mesh.point(ind_v3));
    valuetype volume1 = calc_volume_tetrahedron(mesh.point(ind_v0), pos, mesh.point(ind_v2), mesh.point(ind_v3));
    valuetype volume2 = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), pos, mesh.point(ind_v3));
    valuetype volume3 = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), mesh.point(ind_v2), pos);
    if (fabs(volume0+volume1+volume2+volume3 - volume) > tol_zero)
	std::cerr << "error: interpolation from SEM solution to FEM solution\n";

    valuetype x = volume1 / volume, y = volume2 / volume, z = volume3 / volume;
    valuetype xi = 2*x/(1-y-z)-1, eta = 2*y/(1-z)-1, zeta = 2*z-1;
    std::vector<valuetype> basis(n_index);
    // calculate immediate variable
    valuetype Jxi[M+1], Jeta[M+1], Jzeta[M+1];
    Jxi[0] = Jeta[0] = Jzeta[0] = 1;
    Jxi[1]  = xi;
    for (unsigned int l1 = 1; l1 < M; ++l1)
	Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
	    / calc_coefficient_a(-1, -1, 1, l1);
    for (unsigned int l1 = 0; l1 <= M; ++l1){
	int aph2 = 2 * l1 - 1;
	Jeta[1]  = calc_generalized_jacobi_polynomial( aph2, -1, 1, eta);
	for (unsigned int l2 = 1; l2 < M-l1; ++l2)
	    Jeta[l2+1]  = ((eta - calc_coefficient_a( aph2, -1, 2, l2)) *  Jeta[l2] - calc_coefficient_a( aph2, -1, 3, l2) *  Jeta[l2-1])
		/ calc_coefficient_a( aph2, -1, 1, l2);
	for (unsigned int l2 = 0; l2 <= M-l1; ++l2){
	    int aph3 = 2 * l1 + 2 * l2 - 1;
	    Jzeta[1]  = calc_generalized_jacobi_polynomial( aph3, -1, 1, zeta);
	    for (unsigned int l3 = 1; l3 < M-l1-l2; ++l3)
		Jzeta[l3+1]  = ((zeta - calc_coefficient_a( aph3, -1, 2, l3)) *  Jzeta[l3] - calc_coefficient_a( aph3, -1, 3, l3) *  Jzeta[l3-1])
		    / calc_coefficient_a( aph3, -1, 1, l3);
	    for (unsigned int l3 = 0; l3 <= M-l1-l2; ++l3){
		Multiindex<3> index_now = Unitary_Multiindex[0] * l1 + Unitary_Multiindex[1] * l2 + Unitary_Multiindex[2] * l3;
		int ind_index = correspondence.index2number(index_now);
		basis[ind_index] = pow(1-y-z, l1) * Jxi[l1] * pow(1-z, l2) * Jeta[l2] * Jzeta[l3];
	    }
	}
    }
    std::vector<valuetype> basis_actual(n_index, 0);
    for (int ind_index = 0; ind_index < n_index; ++ind_index)
	for (int ind_tl = 0; ind_tl < n_transform_local[ind_index]; ++ind_tl)
	    basis_actual[transform_local[ind_index][ind_tl]] += weight_transform_local[ind_index][ind_tl] * basis[ind_index];
    
    valuetype rho_local = 0;
    for (int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	valuetype n_occupation = 2.0;
	// calculate function value
	std::vector<valuetype> coef_local(n_index, 0);
	for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	    for (unsigned int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_ele][ind_index]; ++ind_nnz)
		coef_local[ind_index] += psi[ind_orbital](transform_ind_global2local[ind_ele][ind_index][ind_nnz])
		    * transform_val_global2local[ind_ele][ind_index][ind_nnz];

	valuetype psi_local = 0;
	for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
	    psi_local += coef_local[ind_index] * basis_actual[ind_index];
	// add contribution
	rho_local += n_occupation * pow(psi_local, 2);
    }
    return rho_local;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_val_point(Correspondence<3> &correspondence, RegularMesh<3> &mesh, AFEPack::Point<3> &pos, std::vector<Vector<valuetype> > &psi, std::vector<valuetype> &n_occupation)
{
    int n_orbital = psi.size();
    
    // traverse 0-d geometries
    for (unsigned int ind_p = 0; ind_p < n_geometry[0]; ++ind_p)
	if (calc_length_line(mesh.point(ind_p), pos) < tol_zero){
	    int &loc = location_actualdof[location_geometry[ind_p]];
	    valuetype rho_local = 0;
	    for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
		// valuetype n_occupation = 2.0;
		rho_local += n_occupation[ind_orbital] * pow(psi[ind_orbital](loc), 2);
	    }
	    if (rho_local > 1.0e4) std::cerr << "rho_local = " << rho_local << ", ind_p = " << ind_p << '\n';
	    return rho_local;
	}
    // std::cerr << "searched point\n";

    // traverse 1-d geometries
    for (unsigned int ind_e = 0; ind_e < n_geometry[1]; ++ind_e){
	int &ind_point0 = number_node[0][ind_e][0];
	int &ind_point1 = number_node[0][ind_e][1];
	valuetype dis = calc_length_line(mesh.point(ind_point0), mesh.point(ind_point1));
	valuetype dis0 = calc_length_line(mesh.point(ind_point0), pos);
	valuetype dis1 = calc_length_line(mesh.point(ind_point1), pos);
	if (fabs(dis0+dis1 - dis) > tol_zero) continue;
	int &loc0 = location_actualdof[location_geometry[ind_point0]];
	int &loc1 = location_actualdof[location_geometry[ind_point1]];
	int &loc = location_actualdof[location_geometry[n_geometry[0] + ind_e]];
	
	valuetype x = dis0 / dis, r = 1 - x;
	valuetype xi = 2 * x - 1;
	valuetype Jxi[M+1];
	Jxi[0] = 1; Jxi[1] = xi;
	for (unsigned int l1 = 1; l1 < M; ++l1)
	    Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
		/ calc_coefficient_a(-1, -1, 1, l1);
	
	valuetype rho_local = 0;
	for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    // valuetype n_occupation = 2.0;
	    valuetype psi_local = psi[ind_orbital](loc0)*r + psi[ind_orbital](loc1)*x;
	    for (unsigned int l1 = 2; l1 <= M; ++l1)
		psi_local += 2 * Jxi[l1] * psi[ind_orbital](loc + l1-2);
	    rho_local += n_occupation[ind_orbital] * pow(psi_local, 2);
	}
	if (rho_local > 1.0e4) std::cerr << "rho_local = " << rho_local << ", ind_edge = " << ind_e << '\n';
	return rho_local;
    }
    // std::cerr << "searched edge\n";
    
    // traverse 2-d geometries
    for (unsigned int ind_f = 0; ind_f < n_geometry[2]; ++ind_f){
	int &ind_point0 = number_node[1][ind_f][0];
	int &ind_point1 = number_node[1][ind_f][1];
	int &ind_point2 = number_node[1][ind_f][2];
	valuetype area = calc_area_triangle(mesh.point(ind_point0), mesh.point(ind_point1), mesh.point(ind_point2));
	valuetype area0 = calc_area_triangle(pos, mesh.point(ind_point1), mesh.point(ind_point2));
	valuetype area1 = calc_area_triangle(mesh.point(ind_point0), pos, mesh.point(ind_point2));
	valuetype area2 = calc_area_triangle(mesh.point(ind_point0), mesh.point(ind_point1), pos);
	if (fabs(area0+area1+area2 - area) > tol_zero) continue;
	int &ind_edge0 = number_edge[ind_f][0];
	int &ind_edge1 = number_edge[ind_f][1];
	int &ind_edge2 = number_edge[ind_f][2];
	int &loc = location_actualdof[location_geometry[n_geometry[0] + n_geometry[1] + ind_f]];
	int &locp0 = location_actualdof[location_geometry[ind_point0]];
	int &locp1 = location_actualdof[location_geometry[ind_point1]];
	int &locp2 = location_actualdof[location_geometry[ind_point2]];
	int &loce0 = location_actualdof[location_geometry[n_geometry[0] + ind_edge0]];
	int &loce1 = location_actualdof[location_geometry[n_geometry[0] + ind_edge1]];
	int &loce2 = location_actualdof[location_geometry[n_geometry[0] + ind_edge2]];
	
	std::vector<valuetype> bas_val(n_dof_geometry[2], 0);
	valuetype x = area1 / area, y = area2 / area, r = 1 - x - y;
	valuetype xi = 2*x/(1-y)-1, eta = 2*y-1;
	valuetype Jxi[M+1], Jeta[M+1];
	Jxi[0] = Jeta[0] = 1; Jxi[1] = xi;
	for (unsigned int l1 = 1; l1 < M; ++l1)
	    Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
		/ calc_coefficient_a(-1, -1, 1, l1);
	for (unsigned int l1 = 2; l1 <= M-1; ++l1){
	    int aph2 = 2 * l1 - 1;
	    Jeta[1]  = calc_generalized_jacobi_polynomial( aph2, -1, 1, eta);
	    for (unsigned int l2 = 1; l2 < M-l1; ++l2)
		Jeta[l2+1]  = ((eta - calc_coefficient_a( aph2, -1, 2, l2)) *  Jeta[l2] - calc_coefficient_a( aph2, -1, 3, l2) *  Jeta[l2-1])
		    / calc_coefficient_a( aph2, -1, 1, l2);
	    for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
		unsigned int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
		bas_val[ind_index] = 2 * pow(1-y, l1) * Jxi[l1] * Jeta[l2];
	    }
	}
	std::vector<valuetype> bas_val_add_xi( n_dof_geometry[1], 0);
	std::vector<valuetype> bas_val_add_eta(n_dof_geometry[1], 0);
	Jxi[0] = Jeta[0] = 1;
	Jxi[1]  = calc_generalized_jacobi_polynomial(1, 1, 1, xi);
	Jeta[1] = calc_generalized_jacobi_polynomial(1, 1, 1, eta);
	for (unsigned int l = 1; l < M; ++l){
	    Jxi[ l+1] = ((xi  - calc_coefficient_a(1, 1, 2, l)) *  Jxi[l] - calc_coefficient_a(1, 1, 3, l) *  Jxi[l-1]) / calc_coefficient_a(1, 1, 1, l);
	    Jeta[l+1] = ((eta - calc_coefficient_a(1, 1, 2, l)) * Jeta[l] - calc_coefficient_a(1, 1, 3, l) * Jeta[l-1]) / calc_coefficient_a(1, 1, 1, l);
	}
	for (unsigned int l = 0; l < n_dof_geometry[1]; ++l){
	    bas_val_add_xi[l]  = Jxi[l] * pow(1-y, l);
	    bas_val_add_eta[l] = Jeta[l];
	}
	
	valuetype rho_local = 0;
	for (unsigned int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    // valuetype n_occupation = 2.0;
	    valuetype psi_local;
	    psi_local = psi[ind_orbital](locp0) * r + psi[ind_orbital](locp1) * x + psi[ind_orbital](locp2) * y;
	    for (unsigned int l = 0; l < n_dof_geometry[1]; ++l)
		psi_local -= 2 * (x*y * psi[ind_orbital](loce0+l) * bas_val_add_eta[l] +
				  y*r * psi[ind_orbital](loce1+l) * bas_val_add_eta[l] +
				  x*r * psi[ind_orbital](loce2+l) * bas_val_add_xi[l]);
	    for (unsigned int l1 = 2; l1 <= M; ++l1)
		for (unsigned int l2 = 1; l2 <= M-l1; ++l2){
		    unsigned int ind_index = (1+l1+l2-3) * (l1+l2-3) / 2 + l2-1;
		    psi_local += bas_val[ind_index] * psi[ind_orbital](loc + ind_index);
		}
	    rho_local += n_occupation[ind_orbital] * pow(psi_local, 2);
	}
	if (rho_local > 1.0e4) std::cerr << "rho_local = " << rho_local << ", ind_f = " << ind_f << '\n';
	return rho_local;
    }
    // std::cerr << "searched face\n";
    
    // traverse 3-d geometries
    for (unsigned int ind_e = 0; ind_e < n_geometry[3]; ++ind_e){
	int &ind_v0 = index_geometry_onelement[ind_e][0][0];
	int &ind_v1 = index_geometry_onelement[ind_e][0][1];
	int &ind_v2 = index_geometry_onelement[ind_e][0][2];
	int &ind_v3 = index_geometry_onelement[ind_e][0][3];
	valuetype volume = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), mesh.point(ind_v2), mesh.point(ind_v3));
	valuetype volume0 = calc_volume_tetrahedron(pos, mesh.point(ind_v1), mesh.point(ind_v2), mesh.point(ind_v3));
	valuetype volume1 = calc_volume_tetrahedron(mesh.point(ind_v0), pos, mesh.point(ind_v2), mesh.point(ind_v3));
	valuetype volume2 = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), pos, mesh.point(ind_v3));
	valuetype volume3 = calc_volume_tetrahedron(mesh.point(ind_v0), mesh.point(ind_v1), mesh.point(ind_v2), pos);
	if (fabs(volume0+volume1+volume2+volume3 - volume) > tol_zero) continue;

	valuetype x = volume1 / volume, y = volume2 / volume, z = volume3 / volume;
	valuetype xi = 2*x/(1-y-z)-1, eta = 2*y/(1-z)-1, zeta = 2*z-1;
	std::vector<valuetype> basis(n_index);
	// calculate immediate variable
	valuetype Jxi[M+1], Jeta[M+1], Jzeta[M+1];
	Jxi[0] = Jeta[0] = Jzeta[0] = 1;
	Jxi[1]  = xi;
	for (unsigned int l1 = 1; l1 < M; ++l1)
	    Jxi[l1+1]  = ((xi - calc_coefficient_a(-1, -1, 2, l1)) *  Jxi[l1] - calc_coefficient_a(-1, -1, 3, l1) *  Jxi[l1-1])
		/ calc_coefficient_a(-1, -1, 1, l1);
	for (unsigned int l1 = 0; l1 <= M; ++l1){
	    int aph2 = 2 * l1 - 1;
	    Jeta[1]  = calc_generalized_jacobi_polynomial( aph2, -1, 1, eta);
	    for (unsigned int l2 = 1; l2 < M-l1; ++l2)
		Jeta[l2+1]  = ((eta - calc_coefficient_a( aph2, -1, 2, l2)) *  Jeta[l2] - calc_coefficient_a( aph2, -1, 3, l2) *  Jeta[l2-1])
		    / calc_coefficient_a( aph2, -1, 1, l2);
	    for (unsigned int l2 = 0; l2 <= M-l1; ++l2){
		int aph3 = 2 * l1 + 2 * l2 - 1;
		Jzeta[1]  = calc_generalized_jacobi_polynomial( aph3, -1, 1, zeta);
		for (unsigned int l3 = 1; l3 < M-l1-l2; ++l3)
		    Jzeta[l3+1]  = ((zeta - calc_coefficient_a( aph3, -1, 2, l3)) *  Jzeta[l3] - calc_coefficient_a( aph3, -1, 3, l3) *  Jzeta[l3-1])
			/ calc_coefficient_a( aph3, -1, 1, l3);
		for (unsigned int l3 = 0; l3 <= M-l1-l2; ++l3){
		    Multiindex<3> index_now = Unitary_Multiindex[0] * l1 + Unitary_Multiindex[1] * l2 + Unitary_Multiindex[2] * l3;
		    int ind_index = correspondence.index2number(index_now);
		    basis[ind_index] = pow(1-y-z, l1) * Jxi[l1] * pow(1-z, l2) * Jeta[l2] * Jzeta[l3];
		}
	    }
	}
	std::vector<valuetype> basis_actual(n_index, 0);
	for (int ind_index = 0; ind_index < n_index; ++ind_index)
	    for (int ind_tl = 0; ind_tl < n_transform_local[ind_index]; ++ind_tl)
		basis_actual[transform_local[ind_index][ind_tl]] += weight_transform_local[ind_index][ind_tl] * basis[ind_index];
    
	valuetype rho_local = 0;
	for (int ind_orbital = 0; ind_orbital < n_orbital; ++ind_orbital){
	    // valuetype n_occupation = 2.0;
	    // calculate function value
	    std::vector<valuetype> coef_local(n_index, 0);
	    for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
		for (unsigned int ind_nnz = 0; ind_nnz < transform_n_global2local[ind_e][ind_index]; ++ind_nnz)
		    coef_local[ind_index] += psi[ind_orbital](transform_ind_global2local[ind_e][ind_index][ind_nnz])
			* transform_val_global2local[ind_e][ind_index][ind_nnz];

	    valuetype psi_local = 0;
	    for (unsigned int ind_index = 0; ind_index < n_index; ++ind_index)
		psi_local += coef_local[ind_index] * basis_actual[ind_index];
	    // add contribution
	    rho_local += n_occupation[ind_orbital] * pow(psi_local, 2);
	}
	if (rho_local > 1.0e4) std::cerr << "rho_local = " << rho_local << ", ind_ele = " << ind_e << '\n';
	return rho_local;
    }

    std::cerr << "error, cant find position of given pos in current SEM mesh, pos = (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")\n";
    return 0;
}


TEMPLATE_TSEM
valuetype THIS_TSEM::calc_generalized_jacobi_polynomial(int alpha, int beta, int k, valuetype x)
{// calculate k-th generalized jocobi polynomial with index (alpha, beta) at point x, where $alpha, beta >= -1$
    valuetype ans;
    if (k == 0){
	ans = (valuetype) 1;
	return ans;
    }
    if (alpha + beta == -2){
        if (k == 1) ans = x;
        else ans = ((valuetype) 0.25) * (x-1) * (x+1) * calc_generalized_jacobi_polynomial(1, 1, k-2, x);
	return ans;
    }
    if (alpha == -1){
        ans = (k+beta) * (x-1) / (k*2) * calc_generalized_jacobi_polynomial(1, beta, k-1, x);
	return ans;
    }
    if (beta == -1){
        ans = (k+alpha) * (x+1) / (k*2) * calc_generalized_jacobi_polynomial(alpha, 1, k-1, x);
	return ans;
    }
    valuetype tmp_power = (valuetype) 1;
    ans = (valuetype) 0;
    for (int j = 0; j <= k; ++j){
        valuetype factor = (valuetype) 1;
        for (int i = 0; i < k-j; ++i)
            factor *= ((valuetype) (alpha+j+1 + i)) / (i+1);
        for (int i = 0; i < j; ++i)
            factor *= ((valuetype) (k+alpha+beta+1 + i)) / (i+1);
        ans += factor * tmp_power;
        tmp_power *= (x - 1) / 2;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_a(int alpha, int beta, int ind, int k)
{// calculate coefficient $a_{ind, k}^{alpha, beta}$
    valuetype ans;
    if (k < 0){
	ans = (valuetype) 0;
	return ans;
    }
    switch (ind){
    case 1:
	ans = ((valuetype) (2 * (k+1) * (k+alpha+beta+1))) / ((2*k+alpha+beta+1) * (2*k+alpha+beta+2));
        if (alpha == -1 && beta == -1){
            switch (k){
            case 0: ans = (valuetype) 1;   break;
            case 1: ans = (valuetype) 4;   break;
            case 2: ans = (valuetype) 0.5; break;
            }
	    break;
	}
        if (k == 0 && alpha + beta != -2){
            ans = ((valuetype) 2) / (alpha + beta + 2);
	    break;
	}
	break;
    case 2:
        if (0 <= k && k <= 2 && alpha == -1 && beta == -1){
	    ans = (valuetype) 0;
	    break;
	}
        if (k == 0 && alpha + beta != -2){
            ans = ((valuetype) (beta - alpha)) / (alpha + beta + 2);
	    break;
	}
	ans = ((valuetype) (pow(beta,2)-pow(alpha,2))) / ((2*k+alpha+beta) * (2*k+alpha+beta+2));
	break;
    case 3:
	ans = ((valuetype) (2 * (k+alpha) * (k+beta))) / ((2*k+alpha+beta) * (2*k+alpha+beta+1));
        if (alpha == -1 && beta == -1){
            if (k == 0) ans = (valuetype) 0;
            if (k == 1) ans = (valuetype) 1;
            if (k == 2) ans = (valuetype) 0;
	    break;
        }
        if (k == 0 && alpha+beta != -2){
            ans = (valuetype) 0;
	    break;
	}
	break;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_b(int alpha, int beta, int ind, int k)
{// calculate coefficient $b_{ind, k}^{alpha, beta}$
    valuetype ans;
    if (k < 0){
	ans = (valuetype) 0;
	return ans;
    }
    switch (ind){
    case 1:
        if (k == 0 && alpha >= -1 && beta >= -1){
	    ans = (valuetype) 1;
	    break;
	}
        if (k == 1 && alpha == -1 && beta == -1){
	    ans = (valuetype) 2;
	    break;
	}
	ans = ((valuetype) (k+alpha+beta+1)) / (2*k+alpha+beta+1);
	break;
    case 2:
        if (k == 0 && alpha >= -1 && beta >= -1){
	    ans = (valuetype) 0;
	    break;
	}
        if (k == 1 && alpha == -1 && beta == -1){
	    ans = (valuetype) -1;
	    break;
	}
        ans = ((valuetype) -(k+beta)) / (2*k+alpha+beta+1);
        break;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_c(int alpha, int beta, int ind, int k)
{// calculate coefficient $c_{ind, k}^{alpha, beta}$
    valuetype ans;
    switch (ind){
    case 1:
	ans = calc_coefficient_b(alpha, beta, 1, k) * calc_coefficient_b(alpha+1, beta, 1, k);
	break;
    case 2:
	ans = calc_coefficient_b(alpha, beta, 1, k) * calc_coefficient_b(alpha+1, beta, 2, k) +
	      calc_coefficient_b(alpha, beta, 2, k) * calc_coefficient_b(alpha+1, beta, 1, k-1);
	break;
    case 3:
	ans = calc_coefficient_b(alpha, beta, 2, k) * calc_coefficient_b(alpha+1, beta, 2, k-1);
        break;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_d(int alpha, int beta, int k)
{// calculate coefficient $d_k^{alpha, beta}$
    valuetype ans;
    if (k <= 0){
	ans = (valuetype) 0;
	return ans;
    }
    if (k == 1 && alpha == -1 && beta == -1){
	ans = (valuetype) 1;
	return ans;
    }
    ans = ((valuetype) (k+alpha+beta+1)) / 2;
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_e(int alpha, int beta, int ind, int k)
{// calculate coefficient $e_{ind, k}^{alpha, beta}$
    valuetype ans;
    if (k < 0){
	ans = (valuetype) 0;
	return ans;
    }
    switch (ind){
    case 1:
        if (k == 0 && alpha == -1 && beta == -1){
	    ans = (valuetype) 0.5;
	    break;
	}
        if (k == 1 && alpha == -1 && beta == -1){
	    ans = (valuetype) 0;
	    break;
	}
        ans = ((valuetype) (k+alpha+1)) / (2*k+alpha+beta+2);
	break;
    case 2:
        if (k == 0 && alpha == -1 && beta == -1){
	    ans = (valuetype) -0.5;
	    break;
	}
        if (k == 1 && alpha == -1 && beta == -1){
	    ans = (valuetype) -1;
	    break;
	}
        ans = ((valuetype) -(k+1)) / (2*k+alpha+beta+2);
        break;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_g(int alpha, int beta, int ind, int k)
{// calculate coefficient $g_{ind, k}^{alpha, beta}$
    valuetype ans;
    switch (ind){
    case 1:
	ans = calc_coefficient_e(alpha+1, beta, 2, k) * calc_coefficient_e(alpha, beta, 2, k+1);
	break;
    case 2:
	ans = calc_coefficient_e(alpha+1, beta, 1, k) * calc_coefficient_e(alpha, beta, 2, k) +
	      calc_coefficient_e(alpha+1, beta, 2, k) * calc_coefficient_e(alpha, beta, 1, k+1);
	break;
    case 3:
	ans = calc_coefficient_e(alpha+1, beta, 1, k) * calc_coefficient_e(alpha, beta, 1, k);
        break;
    }
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_rho(Multiindex<3> index)
{// calculate coefficient $\rho_{index}^{-\mathds{1}}$
    valuetype ans;
    int l1 = index.index[0], l2 = index.index[1];
    ans = 2 * calc_coefficient_d(-1, -1, l1) * calc_coefficient_e(-1, 0, 1, l1-1)
        - l1 * calc_coefficient_b(-1, -1, 2, l1);
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_kappa(Multiindex<3> index)
{// calculate coefficient $\kappa_{index}^{-\mathds{1}}$
    valuetype ans;
    int l1 = index.index[0], l2 = index.index[1];
    ans = l1 * calc_coefficient_b(-1, -1, 2, l1)
        - 2 * calc_coefficient_d(-1, -1, l1) * calc_coefficient_e(-1, 0, 1, l1-1);
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_theta(Multiindex<3> index)
{// calculate coefficient $\theta_{index}^{-\mathds{1}}$
    valuetype ans;
    int l1 = index.index[0], l2 = index.index[1];
    ans = 2 * calc_coefficient_d(2*l1-1, -1, l2) * calc_coefficient_e(-1, 2*l1, 1, l2-1)
        - l2 * calc_coefficient_b(2*l1-1, -1, 2, l2);
    return ans;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_coefficient_D(int ind_derivative, int ind_variation, Multiindex<3> index)
{// calculate coefficient for the derivative of generalized jacobi polynomial
    valuetype ans;
    int l1 = index.index[0], l2 = index.index[1], l3 = index.index[2];
    int aph2 = 2 * l1, aph3 = 2 * l1 + 2 * l2;
    switch (ind_derivative){
    case 0: // $\partial x1$,               or multiindex (2, 0, 0)
        ans = 2 * calc_coefficient_d(-1, -1, l1);
	break;
    case 1: // $\partial x2 - \partial x1$, or multiindex (1, 1, 0)
        switch (ind_variation){
        case 0: // $D^{21}_0 =  D^{2}_0$
            ans = 2 * calc_coefficient_d(aph2-1, -1, l2) * calc_coefficient_b(-1, -1, 1, l1);
	    break;
        case 1: // $D^{21}_1 = -D^{2}_1$
            ans =  -((2*calc_coefficient_d(-1,-1,l1)*calc_coefficient_e(-1,0,1,l1-1) - l1*calc_coefficient_b(-1,-1,2,l1)) * calc_coefficient_b(aph2-1,-1,1,l2)
		     + 2*calc_coefficient_d(aph2-1,-1,l2)*calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-1,0,2,l2-1)) / calc_coefficient_b(aph2-2,0,1,l2);
	    break;
        }
	break;
    case 2: // $\partial x1 - \partial x3$, or multiindex (1, 0, 1)
        switch (ind_variation){
        case 0: // $D^{31}_{0,0} = -D^3_{0,0}$
            ans = -2 * calc_coefficient_d(aph3-1,-1,l3) * calc_coefficient_b(-1,-1,1,l1) * calc_coefficient_b(aph2-1,-1,1,l2);
	    break;
        case 1: // $D^{31}_{1,0} =  D^3_{1,0}$
            ans = 2 * calc_coefficient_d(aph3-1,-1,l3) * calc_coefficient_b(-1,-1,2,l1) * calc_coefficient_e(aph2-2,-1,2,l2);
	    break;
        case 2: // $D^{31}_{0,1} = -D^3_{0,1}$
            ans = -(calc_coefficient_b(-1,-1,1,l1)*calc_coefficient_theta(index)*calc_coefficient_b(aph3-1,-1,1,l3)
		    + 2*calc_coefficient_d(aph3-1,-1,l3)*calc_coefficient_b(-1,-1,1,l1)*calc_coefficient_b(aph2-1,-1,2,l2)
		    *calc_coefficient_e(aph3-1,0,2,l3-1)) / calc_coefficient_b(aph3-2,0,1,l3);
	    break;
        case 3:
            ans = ((calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-1,-1,2,l2-1)*calc_coefficient_theta(index) - calc_coefficient_kappa(index))
		   * calc_coefficient_b(aph3-1,-1,1,l3) + 2*calc_coefficient_b(aph2-2,-1,1,l2)*calc_coefficient_d(aph3-1,-1,l3)
		   *calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-2,-1,1,l2)*calc_coefficient_e(aph3-1,0,2,l3-1))
                / (calc_coefficient_b(aph2-2,-1,1,l2)*calc_coefficient_b(aph3-2,0,1,l3));
	    break;
        }
	break;
    case 3: // $\partial x2$,               or multiindex (0, 2, 0)
        switch (ind_variation){
        case 0:
	    ans = 2 * calc_coefficient_d(aph2-1, -1, l2) * calc_coefficient_b(-1, -1, 1, l1);
	    break;
        case 1:
	    ans = ((2*calc_coefficient_d(-1,-1,l1)*calc_coefficient_e(-1,0,1,l1-1) - l1*calc_coefficient_b(-1,-1,2,l1)) * calc_coefficient_b(aph2-1,-1,1,l2)
		   + 2*calc_coefficient_d(aph2-1,-1,l2)*calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-1,0,2,l2-1)) / calc_coefficient_b(aph2-2,0,1,l2);
	    break;
        }
	break;
    case 4: // $\partial x3 - \partial x2$, or multiindex (0, 1, 1)
        switch (ind_variation){
        case 0:
	    ans = 2 * calc_coefficient_d(aph3-1,-1,l3) * calc_coefficient_b(aph2-1,-1,1,l2);
	    break;
        case 1:
            ans = ((l2*calc_coefficient_b(-1,aph2-1,2,l2) - 2*calc_coefficient_d(aph2-1,-1,l2)*calc_coefficient_e(aph2-1,0,1,l2-1))
		   * calc_coefficient_b(aph3-1,-1,1,l3) - 2*calc_coefficient_d(aph3-1,-1,l3)*calc_coefficient_b(-1,aph2-1,2,l2)
		   *calc_coefficient_e(aph3-1,0,2,l3-1)) / calc_coefficient_b(aph3-2,0,1,l3);
	    break;
        }
	break;
    case 5: // $\partial x3$,               or multiindex (0, 0, 2)
        switch (ind_variation){
        case 0:
	    ans = 2 * calc_coefficient_d(aph3-1,-1,l3) * calc_coefficient_b(-1,-1,1,l1) * calc_coefficient_b(aph2-1,-1,1,l2);
	    break;
        case 1:
	    ans = 2 * calc_coefficient_d(aph3-1,-1,l3) * calc_coefficient_b(-1,-1,2,l1) * calc_coefficient_e(aph2-2,-1,2,l2);
	    break;
        case 2:
            ans = (calc_coefficient_b(-1,-1,1,l1)*calc_coefficient_theta(index)*calc_coefficient_b(aph3-1,-1,1,l3)
		   + 2*calc_coefficient_d(aph3-1,-1,l3)*calc_coefficient_b(-1,-1,1,l1)*calc_coefficient_b(aph2-1,-1,2,l2)
		   *calc_coefficient_e(aph3-1,0,2,l3-1)) / calc_coefficient_b(aph3-2,0,1,l3);
	    break;
        case 3:
            ans = ((calc_coefficient_rho(index) + calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-1,-1,2,l2-1)*calc_coefficient_theta(index))
		   * calc_coefficient_b(aph3-1,-1,1,l3) + 2*calc_coefficient_b(aph2-2,-1,1,l2)*calc_coefficient_d(aph3-1,-1,l3)
		   *calc_coefficient_b(-1,-1,2,l1)*calc_coefficient_e(aph2-2,-1,1,l2)
		   *calc_coefficient_e(aph3-1,0,2,l3-1)) / (calc_coefficient_b(aph2-2,-1,1,l2)*calc_coefficient_b(aph3-2,0,1,l3));
	    break;
        }
	break;
    }
    return ans;
}

TEMPLATE_TSEM
bool THIS_TSEM::is_on_boundary(AFEPack::Point<3> &p, valuetype bnd_left, valuetype bnd_right)
{// judge whether point p is on boundary of [Bnd_Left, Bnd_Right]^3
    bool flag = false;
    for (int ind = 0; ind < 3; ++ind){
        bool flag_onleftbnd  = fabs(p[ind] - bnd_left)  < tol_zero;
        bool flag_onrightbnd = fabs(p[ind] - bnd_right) < tol_zero;
        if (flag_onleftbnd || flag_onrightbnd) flag = true;
    }
    // if (fabs(p[0] - 1.0/3) < 1.0e-8 && fabs(p[1] - 1.0/3) < 1.0e-8 && fabs(p[2] - 1.0/3) < 1.0e-8)
    //     flag = true; // for tetrahedron
    return flag;
}

TEMPLATE_TSEM
bool THIS_TSEM::is_on_boundary(AFEPack::Point<3> &p, valuetype radius)
{// judge whether point p is on boundary of ball centering at (0,0,0) with radius
    // std::cerr << "p = (" << p[0] << ", " << p[1] << ", " << p[2] << "), dis = " << fabs(radius-sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])) << '\n';
    // return fabs(sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) - radius) < radius * 0.1;
    return fabs(sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) - radius) < tol_zero;
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_length_line(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1)
{
    return sqrt(pow(p0[0]-p1[0], 2) + pow(p0[1]-p1[1], 2) + pow(p0[2]-p1[2], 2));
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_area_triangle(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1, AFEPack::Point<3> &p2)
{
    return 0.5 * sqrt(pow((p1[1]-p0[1])*(p2[2]-p0[2]) - (p1[2]-p0[2])*(p2[1]-p0[1]), 2) +
		      pow((p1[0]-p0[0])*(p2[2]-p0[2]) - (p1[2]-p0[2])*(p2[0]-p0[0]), 2) +
		      pow((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]), 2));
}

TEMPLATE_TSEM
valuetype THIS_TSEM::calc_volume_tetrahedron(AFEPack::Point<3> &p0, AFEPack::Point<3> &p1, AFEPack::Point<3> &p2, AFEPack::Point<3> &p3)
{
    return fabs(((p1[0] - p0[0])*(p2[1] - p0[1])*(p3[2] - p0[2])
		 + (p1[1] - p0[1])*(p2[2] - p0[2])*(p3[0] - p0[0])
		 + (p1[2] - p0[2])*(p2[0] - p0[0])*(p3[1] - p0[1])
		 - (p1[0] - p0[0])*(p2[2] - p0[2])*(p3[1] - p0[1])
		 - (p1[1] - p0[1])*(p2[0] - p0[0])*(p3[2] - p0[2])
                 - (p1[2] - p0[2])*(p2[1] - p0[1])*(p3[0] - p0[0]))/6.);
}



#endif

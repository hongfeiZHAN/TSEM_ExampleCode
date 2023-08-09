/*
 * realization of Multiindex
 * declare Unitary_Multiindex at the end of this file
 */
#ifndef __Multiindex_
#define __Multiindex_

#include "Option.h"

template <int dim> class Multiindex
{// dim dimensional multiindex
    
public:
    
    int index[dim];
    
    bool operator <= (const Multiindex<dim> &multiindex){ // overload relation <= for multiindex
        for (int i = 0; i < dim; ++i) if (this->index[i] > multiindex.index[i]) return false;
        return true;
    }
    bool operator == (const Multiindex<dim> &multiindex){ // overload relation <= for multiindex
        for (int i = 0; i < dim; ++i) if (this->index[i] != multiindex.index[i]) return false;
        return true;
    }
    bool operator != (const Multiindex<dim> &multiindex){ // overload relation != for multiindex
        for (int i = 0; i < dim; ++i) if (this->index[i] != multiindex.index[i]) return true;
        return false;}
    void operator = (const Multiindex<dim> &multiindex){ // overload operation = for multiindex
        for (int i = 0; i < dim; ++i) this->index[i] = multiindex.index[i];}
    Multiindex<dim> operator + (const Multiindex<dim> &multiindex){ // overload operation + for multiindex
        Multiindex<dim> ans;
        for (int i = 0; i < dim; ++i) ans.index[i] = this->index[i] + multiindex.index[i];
        return ans;}
    Multiindex<dim> operator - (const Multiindex<dim> &multiindex){ // overload operation + for multiindex
        Multiindex<dim> ans;
        for (int i = 0; i < dim; ++i) ans.index[i] = this->index[i] - multiindex.index[i];
        return ans;}
    Multiindex<dim> operator * (int n){ // overload operation * for multiindex
        Multiindex<dim> ans;
        for (int i = 0; i < dim; ++i) ans.index[i] = this->index[i] * n;
        return ans;}
    int sum(); // summation of all components
    int n_nonzero(){ // number of nonzero components
        int ans = 0;
        for (int i = 0; i < dim; ++i) if (this->index[i] != 0) ans++;
        return ans;}

private:
    
};

#define TEMPLATE_MUL template<int dim>
#define THIS_MUL Multiindex<dim>

TEMPLATE_MUL
int THIS_MUL::sum()
{
    int ans = 0;
    for (int i = 0; i < dim; ++i)
	ans += this->index[i];
    return ans;
}



int calc_binomial_coefficient_multiindex(Multiindex<1> &alpha, Multiindex<1> &beta)
{// calculate binomial coefficient for multiindex $\binom{\alpha}{\beta}$
    return calc_binomial_coefficient(alpha.index[0], beta.index[0]);
}
int calc_binomial_coefficient_multiindex(Multiindex<2> &alpha, Multiindex<2> &beta)
{// calculate binomial coefficient for multiindex $\binom{\alpha}{\beta}$
    int dim = 2;
    int val = 1;
    for (int ind = 0; ind < dim; ++ind) // for $\beta$ or $\alpha-\beta$ with negative component, return 0
        if (beta.index[ind] < 0 || beta.index[ind] > alpha.index[ind])
            return 0;
    for (int ind = 0; ind < dim; ++ind)
        val *= calc_binomial_coefficient(alpha.index[ind], beta.index[ind]);
    return val;
}
int calc_binomial_coefficient_multiindex(Multiindex<3> &alpha, Multiindex<3> &beta)
{// calculate binomial coefficient for multiindex $\binom{\alpha}{\beta}$
    int dim = 3;
    int val = 1;
    for (int ind = 0; ind < dim; ++ind) // for $\beta$ or $\alpha-\beta$ with negative component, return 0
        if (beta.index[ind] < 0 || beta.index[ind] > alpha.index[ind])
            return 0;
    for (int ind = 0; ind < dim; ++ind)
        val *= calc_binomial_coefficient(alpha.index[ind], beta.index[ind]);
    return val;
}


static Multiindex<3> Unitary_Multiindex[3];
void init_unitary_multiindex()
{
    for (int ind = 0; ind < 3; ++ind){
	for (int ind2 = 0; ind2 < 3; ++ind2)
	    Unitary_Multiindex[ind].index[ind2] = 0;
	Unitary_Multiindex[ind].index[ind] = 1;
    }
}
static Multiindex<3> Zero_Multiindex;
void init_zero_multiindex()
{
    for (int ind = 0; ind < 3; ++ind)
	Zero_Multiindex.index[ind] = 0;
}

#endif

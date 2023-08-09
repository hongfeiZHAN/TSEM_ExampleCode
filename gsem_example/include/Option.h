#ifndef __Option_
#define __Option_


#include "AFEPack/Geometry.h"
/* #define DIM 3 */
/* #define PI (4.0*atan(1.0)) */


bool is_onboundary(AFEPack::Point<3> &p, const double &Bnd_Left, const double &Bnd_Right)
{// judge whether point p is on boundary of [Bnd_Left, Bnd_Right]^3
    bool flag = false;
    for (int ind = 0; ind < 3; ++ind){
        bool flag_onleftbnd  = fabs(p[ind] - Bnd_Left)  < 1.0e-8;
        bool flag_onrightbnd = fabs(p[ind] - Bnd_Right) < 1.0e-8;
        if (flag_onleftbnd || flag_onrightbnd) flag = true;
    }
    // if (fabs(p[0] - 1.0/3) < 1.0e-8 && fabs(p[1] - 1.0/3) < 1.0e-8 && fabs(p[2] - 1.0/3) < 1.0e-8)
    //     flag = true; // for tetrahedron
    return flag;
}

AFEPack::Point<3> calc_cross_product(AFEPack::Point<3> p1, AFEPack::Point<3> p2)
{// calculate the cross product between 0p1 and 0p2
    AFEPack::Point<3> ans;
    ans[0] =  p1[1] * p2[2] - p1[2] * p2[1];
    ans[1] = -p1[0] * p2[2] + p1[2] * p2[0];
    ans[2] =  p1[0] * p2[1] - p1[1] * p2[0];
    return ans;
}

double calc_inner_product(AFEPack::Point<3> p1, AFEPack::Point<3> p2)
{// calculate the inner product between two vectors, whose enties are given by points p1 and p2
    return p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2];
}

int calc_delta(int i, int j)
{// calculate the delta function $\delta_{ij}$
    if (i == j) return 1;
    return 0;
}

int calc_binomial_coefficient(int n, int m)
{// calculate binomial coefficient $\binom{n}{m}$
    int val = 1;
    if (n > 0 && n >= m){
        if (n-m < m) m = n-m;
        for (int i = 0; i < m; ++i){val *= (n-i); val /= (i+1);}}
    return val;
}


#endif

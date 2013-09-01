/*
 *
 * This is to define some shared-variables
 * used in both lbfgs and owlqn algorithm.
 *
 */

#ifndef __QUASI_NEWTON_
#define __QUASI_NEWTON_

namespace maxent {

using namespace math;

const static int    QN_M = 10;
const static double LINE_SEARCH_ALPHA = 0.1;
const static double LINE_SEARCH_BETA  = 0.5;

const static int LBFGS_MAX_ITER = 300;
const static int OWLQN_MAX_ITER = 300;

const static double MIN_GRAD_NORM = 0.0001;

/*
 * For approximating the (Hessian .* gradient)
 */
static Vec approximate_Hg(
        const int iter,
        const Vec& grad,
        const Vec S[],
        const Vec Y[],
        const double z[])
{
    int offset, bound;
    if (iter <= QN_M)
    {
        bound = iter; offset = 0;
    }
    else
    {
        bound = QN_M; offset = iter - QN_M;
    }

    Vec q = grad;
    double alpha[QN_M], beta[QN_M];
    for (int i = bound - 1; i >= 0; --i)
    {
        const int j = (i + offset) % QN_M;
        alpha[i] = z[j] * dot_product(S[j], q);
        q += -alpha[i] * Y[j];
    }
    if (iter > 0)
    {
        const int j = (iter - 1) % QN_M;
        const double gamma = ((1.0 / z[j]) / dot_product(Y[j], Y[j]));
        q *= gamma;
    }

    for (int i = 0; i <= bound - 1; i++) {
        const int j = (i + offset) % QN_M;
        beta[i]     = z[j] * dot_product(Y[j], q);
        q          += S[j] * (alpha[i] - beta[i]);
    }

    return q;
}

}

#endif

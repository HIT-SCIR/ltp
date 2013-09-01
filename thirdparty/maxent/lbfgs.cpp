#include "maxent.h"
#include "opmath.h"
#include "quasi_newton.h"
#include <cmath>
#include <cstdio>

namespace maxent {

using namespace std;
using namespace math;

double ME_Model::backtracking_line_search(
        const Vec& x0,
        const Vec& grad0,
        const double f0,
        const Vec& dx,
        Vec& x,
        Vec& grad1)
{
    double t = 1.0 / LINE_SEARCH_BETA;
    double f;

    do {
        t *= LINE_SEARCH_BETA;
        x = x0 + t * dx;
        f = l2_regularized_func_gradient(x.stl_vec(), grad1.stl_vec());
    } while (f > f0 + LINE_SEARCH_ALPHA * t * dot_product(dx, grad0));

    return f;
}

void ME_Model::perform_LBFGS()
{
    const size_t dim = _vec_lambda.size();
    Vec x = _vec_lambda;

    Vec grad(dim), dx(dim);
    double f = l2_regularized_func_gradient(x.stl_vec(), grad.stl_vec());

    Vec S[QN_M], Y[QN_M];
    double z[QN_M]; // rho

    for (int iter = 0; iter < LBFGS_MAX_ITER; iter++)
    {
        fprintf(stderr, "%3d  obj(err) = %f (%6.4f)", iter+1, -f, _train_error);
        if (_param.nheldout > 0)
        {
            const double heldout_logl = heldout_likelihood();
            fprintf(stderr, "  heldout_logl(err) = %f (%6.4f)", heldout_logl, _heldout_error);
        }
        fprintf(stderr, "\n");

        if (sqrt(dot_product(grad, grad)) < MIN_GRAD_NORM) break; // stopping criteria

        dx = -1 * approximate_Hg(iter, grad, S, Y, z);

        Vec x1(dim), grad1(dim);

        f = backtracking_line_search(x, grad, f, dx, x1, grad1);

        S[iter % QN_M] = x1 - x;
        Y[iter % QN_M] = grad1 - grad;
        z[iter % QN_M] = 1.0 / dot_product(Y[iter % QN_M], S[iter % QN_M]);
        x = x1;

        grad = grad1;
    }

    for (size_t i = 0; i < x.size(); ++i)
        _vec_lambda[i] = x[i];
}

}

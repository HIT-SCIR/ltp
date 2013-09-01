#include "maxent.h"
#include "opmath.h"
#include "quasi_newton.h"
#include <cmath>
#include <cstdio>

namespace maxent {

using namespace std;
using namespace math;

static Vec pseudo_gradient(
        const Vec& x,
        const Vec& grad0,
        const double C)
{
    Vec grad = grad0;
    for (size_t i = 0; i < x.size(); i++)
    {
        if (x[i] != 0)
        {
            grad[i] += C * sign(x[i]);
            continue;
        }

        const double gm = grad0[i] - C;
        if (gm > 0)
        {
            grad[i] = gm;
            continue;
        }

        const double gp = grad0[i] + C;
        if (gp < 0)
        {
            grad[i] = gp;
            continue;
        }

        grad[i] = 0;
    }

    return grad;
}

double ME_Model::constrained_line_search(
                const Vec& x0,
                const Vec& grad0,
                const double f0,
                const Vec& dx,
                Vec& x,
                Vec& grad1)
{
    /* compute the orthant to explore */
    Vec orthant = x0;
    for (size_t i = 0; i < orthant.size(); i++)
    {
        if (orthant[i] == 0) orthant[i] = -grad0[i];
    }

    double t = 1.0 / LINE_SEARCH_BETA;

    double f;
    do {
        t *= LINE_SEARCH_BETA;
        x = x0 + t * dx;
        x.project(orthant);
        f = l1_regularized_func_gradient(x, grad1);
    } while (f > f0 + LINE_SEARCH_ALPHA * dot_product(x - x0, grad0));

    return f;
}

void ME_Model::perform_OWLQN()
{
    const size_t dim = _vec_lambda.size();
    Vec x = _vec_lambda;

    Vec grad(dim), dx(dim);
    double f = l1_regularized_func_gradient(x, grad);

    Vec S[QN_M], Y[QN_M];
    double z[QN_M];  // rho

    for (int iter = 0; iter < OWLQN_MAX_ITER; iter++)
    {
        Vec pg = pseudo_gradient(x, grad, _param.l1_reg);

        fprintf(stderr, "%3d  obj(err) = %f (%6.4f)", iter+1, -f, _train_error);
        if (_param.nheldout > 0)
        {
            const double heldout_logl = heldout_likelihood();
            fprintf(stderr, "  heldout_logl(err) = %f (%6.4f)", heldout_logl, _heldout_error);
        }
        fprintf(stderr, "\n");

        if (sqrt(dot_product(pg, pg)) < MIN_GRAD_NORM)
            break;

        dx = -1 * approximate_Hg(iter, pg, S, Y, z);
        if (dot_product(dx, pg) >= 0)
            dx.project(-1 * pg);

        Vec x1(dim), grad1(dim);
        f = constrained_line_search(x, pg, f, dx, x1, grad1);

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

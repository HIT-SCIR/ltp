#include "maxent.h"
#include "opmath.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace maxent {

using namespace std;
using namespace math::minifunc;

void ME_Model::sgd_apply_penalty(
        const int i,
        const double u,
        vector<double>& q)
{
    double& wi = _vec_lambda[i];
    const double z = wi;
    double& qi = q[i];

    if (wi > 0)
    {
        wi = max(.0, wi - (u + qi));
    }
    else if (wi < 0)
    {
        wi = min(.0, wi + (u - qi));
    }
    qi += (wi - z);
}

void ME_Model::perform_SGD()
{
    const double l1_reg = _param.l1_reg;
    const double eta0   = _param.sgd_eta0;
    const double alpha  = _param.sgd_alpha;
    const int num_iter  = _param.sgd_iter;

    const int num_mef = _set_mefeature.size();

    vector<int> rand_idx(_training_data.size()); // randomized index of training samples
    for (size_t i = 0; i < rand_idx.size(); ++i)
        rand_idx[i] = i;

    vector<double> grad(num_mef);
    int iter_sample = 0;

    double u = 0;
    vector<double> q(num_mef, 0);

    for (int iter = 0; iter < num_iter; ++iter)
    {
        random_shuffle(rand_idx.begin(), rand_idx.end());

        double log_likelihood = 0;
        int n_error = 0, n_total = 0;
        for (size_t i = 0; i < _training_data.size(); ++i, n_total++, iter_sample++)
        {
            const Sample& s = _training_data[rand_idx[i]];

            vector<double> vec_prob(_num_classes);
            const int plabel = classify(s, vec_prob);

            // think it would be negative iter_number/size
            const double eta = eta0 * pow(alpha, (double)iter_sample / _training_data.size());
            u += eta * l1_reg;

            log_likelihood += log(vec_prob[s.label]);
            if (plabel != s.label) n_error++;

            /* binary features */
            for (vector<int>::const_iterator j = s.features.begin(); j != s.features.end(); ++j)
            {
                for (vector<int>::const_iterator k = _feature2mef[*j].begin(); k != _feature2mef[*j].end(); ++k)
                {
                    // model expectation, for one sample
                    const double me = vec_prob[_set_mefeature.feature(*k).label()];
                    // empirical expectation, for one sample
                    const double ee = (_set_mefeature.feature(*k).label() == s.label ? 1.0 : 0);

                    const double grad = me - ee;
                    _vec_lambda[*k] -= eta * grad;
                    sgd_apply_penalty(*k, u, q);
                }
            }

            /* real-valued features */
            for (vector< pair<int, double> >::const_iterator j = s.rvfeatures.begin(); j != s.rvfeatures.end(); ++j)
            {
                for (vector<int>::const_iterator k = _feature2mef[j->first].begin(); k != _feature2mef[j->first].end(); ++k)
                {
                    const double me = vec_prob[_set_mefeature.feature(*k).label()];
                    const double ee = (_set_mefeature.feature(*k).label() == s.label ? 1.0 : 0);
                    const double grad = (me - ee) * j->second;
                    _vec_lambda[*k] -= eta * grad;
                    sgd_apply_penalty(*k, u, q);
                }
            }
        }

        log_likelihood /= _training_data.size();

        if (l1_reg > 0)
        {
            const double l1 = l1norm(_vec_lambda);
            log_likelihood -= l1_reg * l1;
        }

        /*
         * Note that _train_error cannot be used here,
         * since it is batch-implemented.
         */
        fprintf(stderr, "%3d  obj(err) = %f (%6.4f)", iter+1, log_likelihood, (double)n_error/n_total);

        if (_param.nheldout > 0)
        {
            double heldout_logl = heldout_likelihood();
            fprintf(stderr, "  heldout_logl(err) = %f (%6.4f)", heldout_logl, _heldout_error);
        }
        fprintf(stderr, "\n");
    }
}

}

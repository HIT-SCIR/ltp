/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * lbfgstrainer.cpp  -  trainer for conditional ME with L-BFGS method
 *
 * this class utilizes the Fortran implementation of L-BFGS described in
 *
 *   "Limited Memory BFGS Method for Large Scale Optimization"
 *                      by Jorge Nocedal
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jun-2003
 * Last Change : 24-Dec-2004.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/timer.hpp>
#include <boost/scoped_array.hpp>
#include "lbfgstrainer.hpp"
#include "display.hpp"
#include "finite.h"
#include "lbfgs.h"

namespace maxent{
using namespace std;

void LBFGSTrainer::init_trainer() {
    assert(m_params->size() > 0);
    assert(m_es->size() > 0);

    m_observed_expects.reset(new double[m_n_theta]);
    std::fill(m_observed_expects.get(), m_observed_expects.get() + m_n_theta, 0.0);

    m_N = 0;
    // calculate observed feature expectations
    // a hash map to hold the value of feature <pred, outcome> pair occured in event list
    // which is the sum of active feature f_i(a,b) in the training set
    typedef hash_map <pair<size_t, size_t>, float, featid_hasher> FeatSumMap;
    FeatSumMap feat_sum;
    for (vector<Event>::const_iterator it = m_es->begin();
            it != m_es->end(); ++it) {
        size_t len = it->context_size();
        for (size_t i = 0; i < len; ++i) {
            feat_sum[make_pair(it->m_context[i].first,it->m_outcome)] +=
                it->m_count * it->m_context[i].second;
        }
        m_N += it->m_count;
    }

    // Get the observed expectations of the features. Strictly speaking,
    // we should divide the counts by the number of Tokens, but because of
    // the way the model's expectations are approximated in the
    // implementation, this is cancelled out when we compute the next
    // iteration of a parameter, making the extra divisions wasteful.

    FeatSumMap::iterator it;
    for (size_t pid = 0; pid < m_params->size(); ++pid) {
        vector<pair<size_t, size_t> >& param = (*m_params)[pid];
        for (size_t j = 0; j < param.size(); ++j) {
            it = feat_sum.find(make_pair(pid,param[j].first));
            assert(it != feat_sum.end());
            if (it == feat_sum.end())
                throw runtime_error("broken training data: some <pid, oid> in params not found in training data");

            m_observed_expects[param[j].second] = -(it->second);
        }
    }

}

// test accuracy of current model on heldout events
double LBFGSTrainer::heldout_accuracy() const {
    size_t correct = 0;
    size_t total = 0;
    vector<double> q(m_n_outcomes); // q(y|x)
    for (vector<Event>::iterator it = m_heldout_es->begin();
            it != m_heldout_es->end(); ++it) {
        total += it->m_count;
        size_t best_oid = eval(it->m_context,it->context_size(), q);
        if (best_oid == it->m_outcome)
            correct += it->m_count;
    }
    return double(correct)/total;
}

void LBFGSTrainer::train(size_t iter, double eps) {
    if (!m_params || !m_es)
        throw runtime_error("Can not train on an empty model");

    init_trainer();

    const double LOG_ZERO = log(DBL_MIN);
    vector<double> q(m_n_outcomes); // q(y|x)
    int n = m_n_theta;
    int m = 5;
    boost::scoped_array<double> grad(new double[n]);
    double* g = grad.get();
    double* x = m_theta.get();
    fill(x, x+n, 0.0);

    double f     = 0.0;
    size_t correct;
    double heldout_acc = -1.0;
    boost::timer t;

    lbfgs_t* opt = lbfgs_create(n, m, eps);
    if (!opt)
        throw runtime_error("fail to initlize L-BFGS optimizer");

    display("");
    display("Starting L-BFGS iterations...");
    display("Number of Predicates:  %d", m_params->size());
    display("Number of Outcomes:    %d", m_n_outcomes);
    display("Number of Parameters:  %d", n);
    display("Number of Corrections: %d", m);
    display("Tolerance:             %E", eps);
    display("Gaussian Penalty:      %s", (m_sigma2?"on":"off"));
#if defined(NDEBUG)
    display("Optimized version");
#endif

    display("iter  eval     loglikelihood  training accuracy   heldout accuracy");
    display("==================================================================");

    for (;opt->niter < (int)iter;) {
        // calculate loglikehood and gradient
        correct = 0;
        f = 0.0;
        std::copy(m_observed_expects.get(), m_observed_expects.get() + n, g);

        for (vector<Event>::iterator it = m_es->begin();
                it != m_es->end(); ++it) {
            size_t best_oid = eval(it->m_context,it->context_size(), q);
            if (best_oid == it->m_outcome)
                correct += it->m_count;
            for (size_t i = 0; i < it->context_size(); ++i) {
                size_t pid = it->m_context[i].first;
                float fval = it->m_context[i].second;
                vector<pair<size_t, size_t> >& param = (*m_params)[pid];
                for (size_t j = 0; j < param.size(); ++j) {
                    size_t oid = param[j].first;
                    size_t fid = param[j].second;
                    g[fid] += it->m_count * q[oid] * fval;
                }
            }

            assert(finite(q[it->m_outcome]));
            double t = log(q[it->m_outcome]);
            if (finite(t))
                f -= it->m_count * t;
            else
                f -= it->m_count * LOG_ZERO;
        }

        if (m_sigma2) { // applying Gaussian penality
            for (size_t i = 0;i < m_n_theta; ++i) {
                double penality = x[i]/m_sigma2[i];
                g[i] += penality;
                f += (penality * x[i]) / 2;
            }
        }

        if (m_heldout_es && m_heldout_es->size() > 0)
            heldout_acc = heldout_accuracy();

        int iflag = lbfgs_run(opt, x, &f, g);

        if (iflag < 0) {
            lbfgs_destory(opt);
            throw runtime_error("lbfgs routine stops with an error");
        } else if (iflag == 0) {
            display("Training terminats succesfully in %.2f seconds", t.elapsed());
            break;
        } else {
            // continue evaluations
            double acc = correct/double(m_N);
            if (m_heldout_es && m_heldout_es->size() > 0) {
                display("%3d    %3d\t%E\t  %.3f%%\t     %.3f%%" , opt->niter ,
                        opt->nfuns , (-f/m_N) , (acc*100) ,  (heldout_acc * 100));
            } else {
                display("%3d    %3d\t%E\t  %.3f%%\t     %s", opt->niter, opt->nfuns ,
                        (-f/m_N) , (acc*100) ,  "N/A");
            }
        }
    }

    if (opt->niter >= (int)iter)
        display("Maximum numbers of %d iterations reached in %.2f seconds", iter
                , t.elapsed());
    display("Highest log-likelihood: %E", (-f/m_N));

    lbfgs_destory(opt);
}

} // namespace maxent


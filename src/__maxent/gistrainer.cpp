/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * gistrainer.cpp  -  a trainer for conditional ME model with GIS algorithm
 *
 * An implementation of Generalized Iterative Scaling.  The reference paper
 * for this implementation was Adwait Ratnaparkhi's tech report at the
 * University of Pennsylvania's Institute for Research in Cognitive Science,
 * and is available at ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z
 *
 * This C++ implementation is originally based on java maxent implementation,
 * with the help of developers from java maxent.
 * see http://maxent.sf.net
 * 
 * Current implementation implements the "Correction Free" GIS algorithm with
 * Gaussian prior smoothing described in [Curran and Clark, 2003]:
 * "Investigating GIS and Smoothing for Maximum Entropy Taggers".
 *
 * Without the computation of correction parameter, the new GIS algorithm is
 * much faster, making the already simple algorithm simpler.
 *
 * Copyright (C) 2002 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 31-Dec-2002
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
#include <cstdlib>
#include <limits>
#include <boost/timer.hpp>
#include "gistrainer.hpp"
#include "display.hpp"
#include "finite.h"

namespace maxent{

void GISTrainer::init_trainer() {
    assert(m_params->size() > 0);
    assert(m_es->size() > 0);

    // (bool) is needed for BCC5.5
    if ((bool)m_heldout_es && m_heldout_es->size() > 0) {
        cerr << "calculating heldout accuracy is not supported in GIS trainer yet." << endl;
    }

    m_modifiers.reset(new vector<vector<double> >
            (m_params->size(), vector<double>(0)) );
    m_observed_expects.reset(new vector<vector<double> >
            (m_params->size(), vector<double>(0)) );

    // init all thetas to 0.0
    for (size_t i = 0; i < m_params->size(); ++i) {
        std::vector<pair<size_t, size_t> >& param = (*m_params)[i];
        (*m_modifiers)[i].resize(param.size());
        (*m_observed_expects)[i].resize(param.size());
        for (size_t j = 0; j < param.size(); ++j) {
            m_theta[param[j].second] = 0.0;
            assert((*m_modifiers)[i][j] == 0.0);
        }
    }

    // determine the correction constant
    // C = max sum_{x,y} f_i(x, y)
    // m_correct_constant = (*m_es)[0].context_size();
    m_correct_constant = -999;
    for (size_t i = 0; i < m_es->size(); ++i) {
        // size_t len = (*m_es)[i].context_size();
        double t = 0.0;
        Event& e = (*m_es)[i];
        // assume no duplicated features
        for (size_t j = 0; j < e.context_size(); ++j) {
            assert (e.m_context[j].second >= 0.0);
            t += e.m_context[j].second;
        }

        if (t > m_correct_constant)
            m_correct_constant = t;
    }

    m_N = 0;
    for (vector<Event>::iterator it = m_es->begin();
            it != m_es->end(); ++it) {
        // XXX: is the calculation of active features correct?
        m_N += it->m_count;
    }

    // calculate observed feature expectations
    // a hash map to hold the value of feature <pred,outcome> pair occured in event list
    // which is the sum of active feature f_i(a,b) in the training set
    typedef hash_map <pair<size_t, size_t>, float, featid_hasher> FeatSumMap;
    FeatSumMap feat_sum;
    for (vector<Event>::const_iterator it = m_es->begin();
            it != m_es->end(); ++it) {
        size_t len = it->context_size();
        for (size_t i = 0; i < len; ++i) {
        // check for feature values, current implementation only supports
        // binary features
//            if (it->m_context[i].second != 1.0)
//                throw runtime_error("Current GIS implementation only supports binary features, use L-BFGS instead");
            feat_sum[make_pair(it->m_context[i].first,it->m_outcome)] +=
                (it->m_count * it->m_context[i].second);
        }
    }

    // Get the observed expectations of the features. Strictly speaking,
    // we should divide the counts by the number of tokens, but because of
    // the way the model's expectations are approximated in the
    // implementation, this is cancelled out when we compute the next
    // iteration of a parameter, making the extra divisions wasteful.
    // Because we need no division of N in Ep<f_i> and Eq<f_i>
    // when calculating delta of update paramater:
    // lambda(n+1) = lambda(n) + (1/C)*[log(Ep<f_i>) - log(Eq<f_i>)]

    FeatSumMap::iterator it;
    for (size_t pid = 0; pid < m_params->size(); ++pid) {
        vector<pair<size_t, size_t> >& param = (*m_params)[pid];
        vector<double>& observ = (*m_observed_expects)[pid];
        for (size_t j = 0; j < param.size(); ++j) {
            it = feat_sum.find(make_pair(pid,param[j].first));
            assert(it != feat_sum.end());
            if (it == feat_sum.end())
                throw runtime_error("broken training data: some <pid, oid> in params not found in training data");

            observ[j] = it->second; // Ep<f_i> = sum C(f_i)*f_i
        }
    }

    if (!m_sigma2) {
        // We are not using a prior, so we can save log(E_ref) instead 
        // to avoid unnecessary log(*) operation during gis parameter updates

        const double LOG_ZERO = log(numeric_limits<double>::min());
        for (size_t pid = 0; pid < m_params->size(); ++pid) {
            vector<pair<size_t, size_t> >& param = (*m_params)[pid];
            vector<double>& observ = (*m_observed_expects)[pid];
            for (size_t j = 0; j < param.size(); ++j) {
                observ[j] = (observ[j] == 0.0) ? LOG_ZERO : log(observ[j]);
            }
        }
    }
}

void GISTrainer::train(size_t iter, double tol) {
    if (!m_params || !m_es)
        throw runtime_error("Can not train on an empty model");

    init_trainer();

    // now enter iterations
    const double LOG_ZERO = log(numeric_limits<double>::min());
    double old_loglikelihood = 99999;
    double new_loglikelihood = 0.0;
    double acc;
    size_t correct;
    size_t best_oid;
    vector<double> q(m_n_outcomes); // q(y|x)
    boost::timer t;
    size_t niter = 0;

    display("");
    display("Starting GIS iterations...");
    display("Number of Predicates: %d", m_params->size());
    display("Number of Outcomes:   %d", m_n_outcomes);
    display("Number of Parameters: %d", m_n_theta);
    display("Tolerance:            %E", tol);
    display("Gaussian Penalty:     %s", (m_sigma2?"on":"off"));
#if defined(NDEBUG)
    display("Optimized version");
#endif
    display("iters   loglikelihood    training accuracy   heldout accuracy");
    display("=============================================================");

    for (; niter < iter;) {
        new_loglikelihood = 0.0;
        correct = 0;

        // computer modifiers for all features from training data
        for (vector<Event>::iterator it = m_es->begin();
                it != m_es->end(); ++it) {
            best_oid = eval(it->m_context, it->context_size(), q);
            if (best_oid == it->m_outcome)
                correct += it->m_count;
            // TODO:optimize the code
            // calculate Eq<f_i> = \sum q(y|x) * Count(f_i) * f_i(x, y) 
            // (need not being divided by N)
            for (size_t i = 0; i < it->context_size(); ++i) {
                size_t pid = it->m_context[i].first;
                double fval = it->m_context[i].second;
                vector<pair<size_t, size_t> >& param = (*m_params)[pid];
                for (size_t j = 0; j < param.size(); ++j) {
                    size_t oid = param[j].first;
                    (*m_modifiers)[pid][j] += q[oid] * it->m_count * fval;
                    // binary case: (*m_modifiers)[pid][j] += q[oid] * it->m_count;
                }
            }
            assert(finite(q[it->m_outcome]));
            double t = log(q[it->m_outcome]);
            new_loglikelihood += (finite(t) ? t : LOG_ZERO) * it->m_count;
            assert(finite(new_loglikelihood));
        }
        acc = correct/double(m_N);

        // compute the new parameter values
        if (m_sigma2) { // applying Gaussian penality
            for (size_t pid = 0; pid < m_params->size(); ++pid) {
                vector<pair<size_t, size_t> >& param = (*m_params)[pid];
                for (size_t i = 0; i < param.size(); ++i) {
                    size_t fid = param[i].second;
                    m_theta[fid] += newton((*m_modifiers)[pid][i],
                            (*m_observed_expects)[pid][i], fid);
                    (*m_modifiers)[pid][i] = 0.0; // clear modifiers for next iteration
                }
            }
        } else {
            for (size_t pid = 0; pid < m_params->size(); ++pid) {
                vector<pair<size_t, size_t> >& param = (*m_params)[pid];
                for (size_t i = 0; i < param.size(); ++i) {
                    if ((*m_modifiers)[pid][i] != 0.0) { 
                        m_theta[param[i].second] += 
                            ((*m_observed_expects)[pid][i] -
                             log((*m_modifiers)[pid][i])) / m_correct_constant;
                        (*m_modifiers)[pid][i] = 0.0; // clear modifiers for next iteration
                    } else {
                        // E_q == 0 means feature value is 0, which means
                        // update for this parameter will always be zero,
                        // hence can be ignored.
                    }
                }
            }
        }

        ++niter;
        display("%3d\t%E\t  %.3f%%\t     %s",
                niter , (new_loglikelihood/m_N) , (acc*100) ,  "N/A");
        if (fabs((old_loglikelihood - new_loglikelihood)/old_loglikelihood) < tol) {
            display("Training terminats succesfully in %.2f seconds", t.elapsed());
            break;
        }
        old_loglikelihood = new_loglikelihood;
    }
    if (niter >= iter)
        display("Maximum numbers of %d iterations reached in %.2f seconds", iter , t.elapsed());

    // kill a bunch of these big objects now that we don't need them
    m_modifiers.reset();
    m_observed_expects.reset();
}

// Calculate the ith GIS parameter updates with Gaussian prior
// using Newton-Raphson method
// the update rule is the solution of the following equation:
//                                   lambda_i + delta_i
// E_ref = E_q * exp (C * delta_i) + ------------------ * N
//                                       sigma_i^2
// note: E_ref and E_q were not divided by N
double GISTrainer::newton(double f_q, double f_ref, size_t i, double tol) {
    size_t maxiter = 50;
    double x0 = 0.0;
    double x = 0.0;

    for (size_t iter = 1; iter <= maxiter; ++iter) {
        double t = f_q * exp(m_correct_constant * x0);
        double fval = t + m_N * (m_theta[i] + x0) / m_sigma2[i] - f_ref;
        double fpval = t * m_correct_constant + m_N / m_sigma2[i];
        if (fpval == 0) {
            cerr <<
                "Warning: zero-derivative encountered in newton() method." 
                << endl;
            return x0;
        }
        x = x0 - fval/fpval;
        if (abs(x-x0) < tol)
            return x;
        x0 = x;
    }
    throw runtime_error("Failed to converge after 50 iterations in newton() method");
}

} // namespace maxent


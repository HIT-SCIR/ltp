/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * trainer.cpp  -  abstract Trainer interface for conditional ME trainers
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
#include <cstdlib>
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/progress.hpp>
#include <boost/tokenizer.hpp>
#include "trainer.hpp"
#include "modelfile.hpp"
#include "finite.h"

namespace maxent{
using namespace std;
using namespace me;

// Event file format:
// <event_freq> <outcome_id> <context_size> (<context_precidate_id> <feature_value>)+
// only load events, no events merging
void
load_events_txt(const string& filename, MEEventSpace& es) {
    ifstream f(filename.c_str());
    if (!f)
        throw runtime_error("Unable to open event file to read");

    es.clear();

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep(" \t");
    string s;
    tokenizer tokens(s,sep);

    size_t count;
    string outcome;
    string pred;
    float  fval;
    vector<pair<string, float> > context;

    while (getline(f, s)) {
        tokens.assign(s);
        tokenizer::iterator it = tokens.begin();
        count = atoi(it->c_str());
        ++it;
        outcome = it->c_str();
        ++it;
        size_t n = atoi(it->c_str());
        ++it;
        context.clear();
        for (size_t i = 0; i < n; ++i) {
            pred = it->c_str(); ++it;
            fval = atof(it->c_str()); ++it;
            context.push_back(make_pair(pred, fval));
        }
        es.add_event(context, count, outcome);
    }
}

void save_events_txt(const string& filename, const MEEventSpace& es) {
    if (es.empty())
        throw runtime_error("es is empty");

    ofstream f(filename.c_str());
    if (!f)
        throw runtime_error("Unable to open event file to write");

    for (vector<Event>::const_iterator it = es.begin();
            it != es.end(); ++it) {
        f << it->m_count << ' ';
        f << it->m_outcome << ' ';
        f << it->context_size() << ' ';
        for (size_t i = 0; i < it->context_size(); ++i) {
            f << it->m_context[i].first << ' ';
            f << it->m_context[i].second << ' ';
        }
        f << endl;
    }
}

/*
void Trainer::save_param(const string& model, bool binary) const {
    if (!m_params)
        throw runtime_error("Empty trainer (no model)");

    if (binary)
        throw runtime_error("binary param file not support yet");
    else
        save_param_txt(model);
}

void Trainer::save_param_txt(const string& model) const {
    string file = model + ".param";
    ofstream f(file.c_str());
    if (!f)
        throw runtime_error("unable to open param file to write");

    f << *m_correct_constant << endl;
    f << *m_correct_param << endl;
    for (size_t i = 0; i < m_params->size(); ++i) {
        std::vector<pair<size_t, double> >& param = (*m_params)[i];
        for (size_t j = 0; j < param.size(); ++j) {
            f << param[j].second << endl;
        }
    }
}
*/

/*
 * Load training data.
 *
 * @param events the file name of
 */
void Trainer::load_training_data(const string& events, const string& model) {
    shared_ptr<MEEventSpace> e(new MEEventSpace);
    load_events_txt(events, *e);

    MaxentModelFile f;
    f.load(model);
    f.params(m_params, m_n_theta, m_theta);

    m_es = e;
}

/**
 * Setting training data directly.
 *
 * @param events A vector of Event objects consist of training event space.
 * @param params The internal params.
 * @param n_theta The number of \f$\theta_i \f$ parameters.
 * @param sigma2  Global variance \f$\sigma^2 \f$ in Gaussian Prior Smoothing.
 * @param correct_constant Correct constant used in GIS algorithm.
 * @param n_outcomes Number of outcomes.
 * @param heldout_events A vector of Event objects consist of heldout event
 *        space.  this parameter can be safely ignored.
 */
void
Trainer::set_training_data(shared_ptr<MEEventSpace> events,
        shared_ptr<ParamsType> params,
        size_t n_theta,
        shared_array<double> theta,
        shared_array<double> sigma2,
        size_t n_outcomes,
        shared_ptr<MEEventSpace> heldout_events
        ) {
    assert(events && params);
    assert(n_outcomes > 0);

      m_es         = events;
      m_heldout_es = heldout_events;
      m_params     = params;
      m_n_theta    = n_theta;
      m_theta      = theta;
      m_sigma2      = sigma2;
      m_n_outcomes = n_outcomes;
}

// return the oid of best outcome
size_t
Trainer::eval(const Event::context_type* context, size_t len,
        vector<double>& probs) const{
    assert(m_params && m_theta);

    if (probs.size() != m_n_outcomes)
        probs.resize(m_n_outcomes);

    fill(probs.begin(), probs.end(), 0.0);

    for (size_t i = 0;i < len; ++i) {
        vector<pair<size_t, size_t> >& param = (*m_params)[context[i].first];
        float fval = context[i].second;
        for (size_t j = 0; j < param.size(); ++j) {
            size_t oid = param[j].first;
            probs[oid] += m_theta[param[j].second] * fval;
        }
    }

    // normalize
    size_t best_oid = 0;
    double max_prob = -1;
    double sum = 0.0;
    for (size_t oid = 0; oid < m_n_outcomes; ++oid) {
        probs[oid] = exp(probs[oid]);
        if (!finite(probs[oid]))
            probs[oid] = numeric_limits<double>::max();// DBL_MAX;
        sum += probs[oid];
        if (probs[oid] >= max_prob) {
            max_prob = probs[oid];
            best_oid = oid;
        }
    }

    for (size_t oid = 0; oid < m_n_outcomes; ++oid)  {
        probs[oid] /= sum;
    }

    return best_oid;
}

} // namespace maxent

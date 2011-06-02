/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * maxentmodel.cpp  -  a Conditional Maximun Entropy Model
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jan-2003
 * Last Change : 18-Mar-2005.
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
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <set>
#include <map>

#include "hash_map.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/timer.hpp>

#include "display.hpp"
#include "gistrainer.hpp"
#ifdef HAVE_FORTRAN
    #include "lbfgstrainer.hpp"
#endif
#include "modelfile.hpp"
#include "maxentmodel.hpp"
#include "finite.h"

namespace maxent {

using namespace boost;
using namespace std;
using namespace me;

/**
 * Default constructor for MaxentModel.
 *
 * Construct an empty MaxentModel instance
 */
MaxentModel::MaxentModel():
    m_n_theta(0) {
//#ifdef NDEBUG 
//        cerr << "Optimized Version" << endl;
//#endif
    }

/**
 * Signal the begining of adding event (the start of training).
 *
 * This method must be called before adding any event to the model.
 * It informs the model the beginning of training. After the last event is
 * added \sa end_add_event() must be called to indicate the ending of
 * adding events.
 */
void MaxentModel::begin_add_event() {
    m_es.reset(new MEEventSpace);
    m_pred_map = m_es->feat_map();
    m_outcome_map = m_es->outcome_map();
    m_heldout_es.reset(new MEEventSpace(m_pred_map, m_outcome_map));
    m_params.reset(new ParamsType);
    m_timer.reset(new boost::timer());
}

/**
 * Add an event (context, outcome, count) to current model for training later.
 *
 * add_event() should be called after calling \sa begin_add_event().
 * @param context A std::vector of pair<std::string, float> to indicate
 *                the context predicates and their values (must be >= 0)
 *                occured in the event.
 * @param outcome A std::string indicates the outcome label.
 * @param count   How many times this event occurs in training set. default = 1
 */
void MaxentModel::add_event(
        const context_type& context,
        const outcome_type& outcome,
        size_t count) {
    assert(m_es);
    assert(m_heldout_es);
    assert(m_pred_map);
    assert(m_outcome_map);
    m_es->add_event(context, count, outcome);
}

/**
 * Add an event (context, outcome, count) to model for training later.
 *
 * This function is a thin wrapper for the above \sa eval(), with all feature
 * values omitted (default to 1.0, which is binary feature case).
 *
 * add_event() should be called after calling \sa begin_add_event().
 * @param context A list string names of the context predicates occure in the
 *                event. The feature value defaults to 1.0 (binary feature)
 * @param outcome A std::string indicates the outcome label.
 * @param count   How many times this event occurs in training set. default = 1
 */
void MaxentModel::add_event(
        const vector<string>& context,
        const outcome_type& outcome,
        size_t count) {
    context_type c(context.size());
    for (size_t i = 0;i < context.size(); ++i) {
        c[i].first = context[i];
        c[i].second = 1.0;
    }
    return add_event(c, outcome, count);
}

// TODO: test and document this function
void MaxentModel::add_heldout_event(
        const context_type& context,
        const outcome_type& outcome,
        size_t count) {
    assert(m_es);
    assert(m_heldout_es);
    assert(m_pred_map);
    assert(m_outcome_map);
    m_heldout_es->add_event(context, count, outcome);

}

void MaxentModel::add_heldout_event(
        const vector<string>& context,
        const outcome_type& outcome,
        size_t count) {
    context_type c(context.size());
    for (size_t i = 0;i < context.size(); ++i) {
        c[i].first = context[i];
        c[i].second = 1.0;
    }
    return add_heldout_event(c, outcome, count);
}

/**
 * Signal the ending of adding events.
 *
 * This method must be called after adding of the last event to inform the
 * model the ending of the adding events.
 *
 * @param cutoff Event cutoff, all events that occurs less than cutoff times
 * will be discussed. Default = 1 (remain all events). Please note this is
 * different from the usual sense of *feature cutoff*.
 */
void MaxentModel::end_add_event(size_t cutoff) {
    assert(m_es && m_heldout_es);

    //remove duplicate events and update event count
    display("Total %d training events and %d heldout events added in %.2f s",
            m_es->size(), m_heldout_es->size(), m_timer->elapsed());
    display("Reducing events (cutoff is %d)...", cutoff);

    m_es->merge_events(cutoff);

    display("Reduced to %d training events", m_es->size());

    // merge the held out events
    if (m_heldout_es->size() > 0) {
        m_heldout_es->merge_events(cutoff);
        display("Reduced to %d heldout events", m_heldout_es->size());
    }

//        cerr << build_params(m_params, m_n_theta) << endl;
//        cerr << build_params2(m_params, m_n_theta) << endl;

    //TODO: the value of 150 needs more testing
    if (m_outcome_map->size() < 150) 
        build_params(m_params, m_n_theta);
    else
        build_params2(m_params, m_n_theta);
    m_theta.reset(new double[m_n_theta]);
    // m_sigma.reset(0);
}

// TODO: test & document this function
void MaxentModel::dump_events(const string& model, bool binary) const {
    if (!m_es || m_es->size() == 0)
        throw runtime_error("empty model, no events to dump");

    if (binary)
        throw runtime_error("binary events file not supported yet");

    display("Dumping events to %s.ev%s", model.c_str() , (binary?".bin":""));
    if (binary) {
        string file = model + ".ev.bin";
        // save_events_bin(f, *m_es);
    } else {
        string file = model + ".ev";
        save_events_txt(file, *m_es);
    }

    // save model
    MaxentModelFile f;
    f.set_pred_map(m_pred_map);
    f.set_outcome_map(m_outcome_map);
    f.set_params(m_params, m_n_theta, m_theta);
    f.save(model, binary);

    if (m_heldout_es->size() > 0) {
        display("Dumping heldout events to %s.heldout.ev%s", model.c_str() , (binary?".bin":""));
        if (binary) {
            string file = model + ".heldout.ev.bin";
            // save_events_bin(f, *m_heldout_es);
        } else {
            string file = model + ".heldout.ev";
            save_events_txt(file, *m_heldout_es);
        }
    }
}

double MaxentModel::build_params(shared_ptr<ParamsType>& params,
        size_t& n_theta) const {
    boost::timer t;
    assert(m_es);
    typedef hash_map <pair<size_t, size_t>, bool, featid_hasher> FeatMap;

    FeatMap feat_map;
    size_t len;
    for (MEEventSpace::const_iterator it = m_es->begin();
            it != m_es->end(); ++it) {
        len = it->context_size();
        for (size_t i = 0; i < len; ++i) {
            feat_map[make_pair(it->m_context[i].first,it->m_outcome)] = true;
        }
    }
    params.reset(new ParamsType(m_pred_map->size()));

    n_theta = 0;
    for (size_t pid = 0; pid < m_pred_map->size(); ++pid) {
        vector<pair<size_t, size_t> >& param = (*params)[pid];
        for (size_t oid = 0; oid < m_outcome_map->size(); ++oid)
            if (feat_map.find(make_pair(pid, oid)) != feat_map.end())
                param.push_back(make_pair(oid, n_theta++));
    }
    return t.elapsed();
}

// the same as build_params() but specially designed for large outcome set.
// You should use build_params() when working on problem with small outcome
// set
double MaxentModel::build_params2(shared_ptr<ParamsType>& params, 
        size_t& n_theta) const {
    boost::timer t;
    assert(m_es);
    //map: predicate -> set of outcomes
    //i.e., pred1->{out1, out2, out3, ...}
    typedef map<size_t, set<size_t> > FeatMap;

    FeatMap feat_map;
    size_t len;
    for (MEEventSpace::const_iterator it = m_es->begin();
            it != m_es->end(); ++it) {
        len = it->context_size();
        for (size_t i = 0; i < len; ++i) {
            //Find (add if necessary) the set of outcomes for this predicate
            FeatMap::iterator outsetIt = feat_map.find(it->m_context[i].first);
            if(outsetIt == feat_map.end())
            {
                outsetIt = feat_map.insert(make_pair(
                            it->m_context[i].first,set<size_t>() )).first;
                // outsetIt = feat_map.find(it->m_context[i].first);
            }
            //Add the outcome to the set for this predicate
            outsetIt->second.insert(it->m_outcome);
        }
    }
    params.reset(new ParamsType(m_pred_map->size()));

    n_theta = 0;
    for(FeatMap::const_iterator pidIt = feat_map.begin();
            pidIt != feat_map.end();pidIt++)
    {
        vector<pair<size_t, size_t> >& param = (*params)[pidIt->first];
        for(set<size_t>::const_iterator oidIt = pidIt->second.begin();
                oidIt != pidIt->second.end(); oidIt++)
            param.push_back(make_pair(*oidIt, n_theta++));
    }
    return t.elapsed();
}

/**
 * Evaluates a context, return the conditional distribution of the context.
 *
 * This method calculates the conditional probability p(y|x) for each possible
 * outcome tag y.
 *
 * @param context A list of pair<string, double> indicates the contextual
 *                predicates and their values (must be >= 0) which are to be
 *                evaluated together.
 * @param outcomes An array of the outcomes paired with it's probability
 *        predicted by the model (the conditional distribution).
 * @param sort_result Whether or not the returned outcome array is sorted
 *                    (larger probability first). Default is true.
 *
 * TODO:  need optimized for large number of outcomes
 *
 * \sa eval()
 */
void MaxentModel::eval_all(const context_type& context,
        std::vector<pair<outcome_type, double> >& outcomes,
        bool sort_result) const {
    assert(m_params);

    //TODO:static?
    static vector<double> probs;
    if (probs.size() != m_outcome_map->size())
        probs.resize(m_outcome_map->size());
    fill(probs.begin(), probs.end(), 0.0);

    size_t pid;
    for (size_t i = 0; i < context.size(); ++i) {
        pid = m_pred_map->id(context[i].first);
        if (pid != m_pred_map->null_id) {
            std::vector<pair<size_t, size_t> >& param = (*m_params)[pid];
            float fval = context[i].second;
            for(size_t j = 0;j < param.size(); ++j)
                probs[param[j].first] += m_theta[param[j].second] * fval;
        } else {
            //#warning how to deal with unseen predicts?
            //m_debug.debug(0,"Predict id %d not found.",i);
        }
    }

    double sum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = exp(probs[i]);
        sum += probs[i];
    }

    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    outcomes.resize(m_outcome_map->size());
    for (size_t i = 0;i < outcomes.size(); ++i) {
        outcomes[i].first = (*m_outcome_map)[i];
        outcomes[i].second = probs[i];
    }


    if (sort_result)
        sort(outcomes.begin(),outcomes.end(), cmp_outcome());
}


/**
 * Evaluates a context, return the conditional probability p(y|x).
 *
 * This method calculates the conditional probability p(y|x) for given x and y.
 *
 * @param context A list of pair<string, double> indicates names of 
 *        the contextual predicates and their values which are to be
 *        evaluated together.
 * @param outcome The outcome label for which the conditional probability is
 *        calculated.
 * @return The conditional probability of p(outcome|context).
 * \sa eval_all()
 */
double MaxentModel::eval(const context_type& context,
                const outcome_type& outcome) const{

    size_t oid = m_outcome_map->id(outcome);

    if (oid == m_outcome_map->null_id) {
        cerr << "[MaxentModel::eval()] unknown outcome id:" << oid << endl;
        return 0.0;
    }

    static vector<double> probs;
    if (probs.size() != m_outcome_map->size())
        probs.resize(m_outcome_map->size());
        fill(probs.begin(), probs.end(), 0.0);

    size_t pid;
    for (size_t i = 0; i < context.size(); ++i) {
        pid = m_pred_map->id(context[i].first);
        if (pid != m_pred_map->null_id) {
            std::vector<pair<size_t, size_t> >& param = (*m_params)[pid];
            float fval = context[i].second;
            for(size_t j = 0;j < param.size(); ++j)
                probs[param[j].first] += m_theta[param[j].second] * fval;
        } else {
            //#warning how to deal with unseen predicts?
            //m_debug.debug(0,"Predict id %d not found.",i);
        }
    }

    double sum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = exp(probs[i]);
        if (!finite(probs[i]))
            probs[i] = numeric_limits<double>::max();// DBL_MAX;
        sum += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    return probs[oid];
}


/**
 * Evaluates a context, return the most possible outcome y for given context x
 * 
 * This function is a thin wrapper for \sa eval_all().
 * @param context A list of String names of the contextual predicates
 *        which are to be evaluated together.
 * @return The most possible outcome label for given context.
 * \sa eval_all()
 */
MaxentModel::outcome_type MaxentModel::predict(const context_type& context) const {
    std::vector<pair<outcome_type, double> > outcomes;
    eval_all(context, outcomes, false);
    vector<pair<outcome_type, double> >::iterator it;
    vector<pair<outcome_type, double> >::iterator max_it;
    max_it = it = outcomes.begin();
    ++it;
    for (; it != outcomes.end(); ++it){
        if (it->second > max_it->second)
            max_it = it;
    }
    return max_it->first;
}

/**
 * Load a MaxentModel from a file.
 *
 * @param model The name of the model to load
 */
void MaxentModel::load(const string& model) {
    MaxentModelFile f;
    cout << "begin load " << model << endl;
    f.load(model);
    cout << "load " << model << " over!" << endl;
    m_pred_map = f.pred_map();
    m_outcome_map = f.outcome_map();
    f.params(m_params, m_n_theta, m_theta);
}

/**
 * Save a MaxentModel to a file.
 *
 * @param model The name of the model to save.
 * @param binary If true, the file is saved in binary format, which is usually
 * smaller (if compiled with libz) and much faster to load.
 */
void MaxentModel::save(const string& model, bool binary) const {
    if (!m_params)
        throw runtime_error("no model to save (empty model)");
    MaxentModelFile f;
    f.set_pred_map(m_pred_map);
    f.set_outcome_map(m_outcome_map);
    f.set_params(m_params, m_n_theta, m_theta);
    f.save(model, binary);
}

/**
 * Train a ME model using selected training method.
 *
 * This is a wapper function for the underline Trainer class. It will create
 * corresponding Trainer object to train a Conditional MaxentModel. Currently
 * L-BFGS and GIS are implemented.
 *
 * @param iter  Specify how many iterations are need for iterative methods.
 *         Default is 15 iterations.
 *
 * @param method  The training method to use. Can be "lbfgs" or "gis".
 *         L-BFGS is used as the default training method.
 *
 * @param sigma2 Global variance \f$\sigma^2\f$ in Gaussian prior smoothing. Default is 0, which
 *              turns off Gaussian smoothing.
 *
 * @param tol Tolerance for detecting model convergence. A model is regarded as
 *         convergence when \f$|\frac{Log-likelihood(\theta_2) -
 *         Log-likelihood(\theta_1)}{Log-likelihood(\theta_1)}|<tol\f$.
 *         Default tol = 1-E05
 */
void MaxentModel::train(size_t iter, const std::string& method, 
        double sigma2, double tol) {
    if (!m_es)
        throw runtime_error("unable to train an emtpy model");

    scoped_ptr<Trainer> t;
    if (method == "lbfgs") {
#ifdef HAVE_FORTRAN
        t.reset(new LBFGSTrainer);
#else
        cerr << "LBFGS module not compiled in, use GIS instead" << endl;
        t.reset(new GISTrainer);
#endif
    }
    else if (method == "gis")
        t.reset(new GISTrainer);
    else
        throw runtime_error("training method not supported");

    shared_array<double> gaussian;
    if (sigma2 != 0.0) { // use Gaussian prior
        gaussian.reset(new double[m_n_theta]);
        fill(gaussian.get(), gaussian.get() + m_n_theta, sigma2);
    }

    t->set_training_data(m_es, m_params, m_n_theta,
            m_theta, gaussian, m_outcome_map->size(), m_heldout_es);
    t->train(iter, tol);
}

// The following functions are wrapper call for the corresponding functions
// with the omittion of feature values (default to 1.0) for binary feature
// cases.
//
// These functions are provided for conviences, for binary feature cases are
// quit common.

/**
 * Evaluates a context, return the conditional distribution of given context.
 *
 * This method calculates the conditional probability p(y|x) for each possible
 * outcome tag y.
 *
 * This function is a thin warpper for the above \sa eval_all()
 * feature values are omitted (defualt to 1.0) for binary feature value case.
 *
 * @param context A list of string names of the contextual predicates
 *        which are to be evaluated together.
 * @param outcomes an array of the outcomes paired with it's probability
 *        predicted by the model (the conditional distribution).
 * @param sort_result  Whether or not the returned outcome array is sorted
 *                     (larger probability first). Default is true.
 *
 * TODO:  need optimized for large number of outcomes
 *
 * \sa eval()
 */
void MaxentModel::eval_all(const vector<string>& context,
        std::vector<pair<outcome_type, double> >& outcomes,
        bool sort_result) const {
    context_type c(context.size());
    for (size_t i = 0;i < context.size(); ++i) {
        c[i].first = context[i];
        c[i].second = 1.0;
    }
    eval_all(c, outcomes, sort_result);
}

/**
 * Evaluates a context, return the conditional probability p(y|x).
 *
 * This method calculates the conditional probability p(y|x) for given x and y.
 *
 * This is a wrapper function for the above \sa eval(), omitting feature
 * values in paramaters (default to 1.0, treated as  binary case)
 *
 * @param context A list of string names of the contextual predicates to be
 *                evaluated together.
 * @param outcome The outcome label for which the conditional probability is
 *        calculated.
 * @return The conditional probability of p(outcome|context).
 * \sa eval_all()
 */
double MaxentModel::eval(const vector<string>& context,
                const outcome_type& outcome) const{
    context_type c(context.size());
    for (size_t i = 0;i < context.size(); ++i) {
        c[i].first = context[i];
        c[i].second = 1.0;
    }
    return eval(c, outcome);
}

/**
 * Evaluates a context, return the most possible outcome y for given context x
 * 
 * This function is a thin wrapper for \sa predict() for binary value case
 * (omitting feature values which default to 1.0)
 *
 * @param context A list of String names of the contextual predicates
 *        which are to be evaluated together.
 * @return The most possible outcome label for given context.
 * \sa eval_all()
 */
MaxentModel::outcome_type MaxentModel::predict(const vector<string>& context) const {
    context_type c(context.size());
    for (size_t i = 0;i < context.size(); ++i) {
        c[i].first = context[i];
        c[i].second = 1.0;
    }
    return predict(c);
}

// for python __str__() binding
const char* MaxentModel::__str__() const {
    if (!m_params)
        return "Conditional Maximum Entropy Model (C++ version) [empty]";
    else {
        size_t n = 0;
        for (size_t i = 0; i < m_params->size(); ++i)
            n += (*m_params)[i].size();

        static char buf[300];
        sprintf(buf, 
"Conditional Maximum Entropy Model (C++ version)\n"
"Number of context predicates  : %d\n"
"Number of outcome             : %d\n"
"Number of paramaters(features): %d" ,  m_pred_map->size()
, m_outcome_map->size() , n);
        return buf;
    }
}

} // namespace maxent


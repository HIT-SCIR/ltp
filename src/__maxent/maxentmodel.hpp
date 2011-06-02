/**
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * maxentmodel.hpp  -  A Conditional Maximun Entropy Model
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jan-2003
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

#ifndef MAXENTMODEL_H
#define MAXENTMODEL_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <utility>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <ostream>
#include <iostream>

#include "itemmap.hpp"
#include "meevent.hpp"

namespace boost {
    class timer;
}

/**
 * All classes and functions are placed in the namespace maxent.
 */
namespace maxent {
using namespace std;
using boost::shared_ptr;
using boost::shared_array;

extern int verbose;  // set this to 0 if you do not want verbose output

struct maxent_pickle_suite;
/**
 * This class implements a conditional Maximun Entropy Model.
 *
 * A conditional Maximun Entropy Model (also called log-linear model)has the
 * form:
 * \f$p(y|x)=\frac{1}{Z(x)} \exp \left[\sum_{i=1}^k\lambda_if_i(x,y) \right]\f$
 * Where x is a context and y is the outcome tag and p(y|x) is the conditional
 * probability.
 *
 * Normally the context x is composed of a set of contextual predicates.
 */
class MaxentModel /*: TODO: we need copyable? boost::noncopyable*/  {
    friend struct maxent_pickle_suite;

    // private:
    // virtual ~MaxentModel();

    public:
//    typedef std::string feature_type;
//    typedef std::string outcome_type;
    typedef me::feature_type feature_type;
    typedef me::feature_type outcome_type;
    typedef std::vector<pair<feature_type, float> > context_type;

    MaxentModel();

    void load(const string& model);

    void save(const string& model, bool binary = false) const;

    double eval(const context_type& context, const outcome_type& outcome) const;

    void eval_all(const context_type& context,
            std::vector<pair<outcome_type, double> >& outcomes,
            bool sort_result = true) const;

    outcome_type predict(const context_type& context) const;

    void begin_add_event();

    void add_event(const context_type& context,
            const outcome_type& outcome,
            size_t count = 1);

    void add_heldout_event(const context_type& context,
            const outcome_type& outcome,
            size_t count = 1);

    // wrapper functions for binary feature cases, provided for conviences
    void add_event(const vector<string>& context,
            const outcome_type& outcome,
            size_t count = 1);

    void add_heldout_event(const vector<string>& context,
            const outcome_type& outcome,
            size_t count = 1);

    double eval(const vector<string>& context, 
            const outcome_type& outcome) const;

    void eval_all(const vector<string>& context,
            std::vector<pair<outcome_type, double> >& outcomes,
            bool sort_result = true) const;

    outcome_type predict(const vector<string>& context) const;

    /**
     * Add a set of events indicated by range [begin, end).
     * the value type of Iterator must be pair<context_type, outcome_type>
     */
    template <typename Iterator>
        void add_events(Iterator begin, Iterator end) {
            for (Iterator it = begin; it != end; ++it)
                this->add_event(it->first, it->second);
        }


    void end_add_event(size_t cutoff = 1);

    void train(size_t iter = 15, const std::string& method = "lbfgs",
            double sigma2 = 0.0, // non-zero enables Gaussian prior smoothing (global variance sigma^2)
            double tol = 1E-05);

     void dump_events(const string& model, bool binary = false) const;

    const char* __str__() const; // python __str__() 

    // Python binding related functions {{{
#if defined(PYTHON_MODULE)

    // return the whole probabistic distribution [(outcome1, prob1),
    // (outcome2, prob2), ...] for a given context
    std::vector<pair<outcome_type, double> > py_eval(const context_type& context) const {
        static std::vector<pair<outcome_type, double> > outcomes;
        eval_all(context, outcomes);
        return outcomes;
    }
#endif
    // end py binding }}}

    private:
    double build_params(shared_ptr<me::ParamsType>& params, 
            size_t& n_theta) const;
    double build_params2(shared_ptr<me::ParamsType>& params, 
            size_t& n_theta) const;

#if !defined(_STLPORT_VERSION) && defined(_MSC_VER) && (_MSC_VER >= 1300)
            // for MSVC7's hash_map declaration
            class featid_hasher : public stdext::hash_compare<pair<size_t, size_t> > {
                public:
                    size_t operator()(const pair<size_t, size_t>& p) const {
                        return p.first + p.second;
                    }

                    bool operator()(const pair<size_t, size_t>& k1,
                            const  pair<size_t, size_t>& k2) {
                        return k1 < k2;
                    }
            };
#else
            // for hash_map of GCC & STLPORT
            struct featid_hasher {
                size_t operator()(const pair<size_t, size_t>& p) const {
                    return p.first + p.second;
                }
            };

#endif

    struct cutoffed_event {
        cutoffed_event(size_t cutoff):m_cutoff(cutoff) {}
        bool operator()(const me::Event& ev) const {
            return ev.m_count < m_cutoff;
        }
        size_t m_cutoff;
    };

    struct cmp_outcome {
        bool operator()(const pair<outcome_type, double>& lhs,
                const pair<outcome_type, double>& rhs) const {
            return lhs.second > rhs.second;
        }
    };

    size_t m_n_theta;
    shared_ptr<me::MEEventSpace> m_es;
    shared_ptr<me::MEEventSpace> m_heldout_es;
    shared_ptr<me::PredMapType> m_pred_map;
    shared_ptr<me::OutcomeMapType> m_outcome_map;
    shared_ptr<me::ParamsType> m_params;
    shared_array<double> m_theta; // feature weights

    shared_ptr<boost::timer> m_timer;

    struct param_hasher {
        size_t operator()(const pair<size_t,size_t>& v) const {
            return size_t(~(v.first<< 1) + v.second);
        }
    };
};

#if defined(OLD_PYTHON_MODULE) //{{{ old python pickle support through Boost.Python
struct maxent_pickle_suite : boost::python::pickle_suite {
    static boost::python::tuple getstate(const MaxentModel& m)
    {
        if (!m.m_params)
            throw runtime_error("can not get state from empty model");
        using namespace boost::python;
        boost::python::list state;
        size_t i;

        shared_ptr<me::PredMapType> pred_map = m.m_pred_map;
        shared_ptr<me::OutcomeMapType> outcome_map = m.m_outcome_map;
        shared_ptr<me::ParamsType> params = m.m_params;
        size_t n_theta = m.m_n_theta;
        shared_array<double> theta = m.m_theta;

        // save pred_map
        state.append(pred_map->size());
        for (i = 0;i < pred_map->size(); ++i)
            state.append((*pred_map)[i]);

        // save outcome_map
        state.append(outcome_map->size());
        for (i = 0;i < outcome_map->size(); ++i)
            state.append((*outcome_map)[i]);

        // save params
        state.append(n_theta);
        assert(params->size() == pred_map->size());
        for (i = 0;i < params->size(); ++i) {
            boost::python::list oids;
            boost::python::list t;
            const std::vector<pair<size_t, size_t> >& a = (*params)[i];
            for (size_t j = 0; j < a.size(); ++j) {
                oids.append(a[j].first);
                t.append(a[j].second);
            }
            state.append(make_tuple(oids, t));
        }
        // save theta
        for (i = 0;i < n_theta; ++i)
            state.append(theta[i]);
        return boost::python::tuple(state);
    }

    static void setstate(MaxentModel& m, boost::python::tuple state)
    {
        using namespace boost::python;
        assert (!m.m_pred_map);
        assert (!m.m_outcome_map);
        assert (!m.m_params);
        assert (len(state) > 0);

        shared_ptr<me::PredMapType> pred_map(new me::PredMapType);
        shared_ptr<me::OutcomeMapType> outcome_map(new me::OutcomeMapType);
        shared_ptr<me::ParamsType> params(new me::ParamsType);
        size_t n_theta;
        shared_array<double> theta;

        size_t count;
        size_t i;
        size_t index = 0;

        // load pred_map
        count = extract<size_t>(state[index++]);
        for (i = 0; i < count; ++i)
            pred_map->add(extract<std::string>(state[index++]));

        // load outcome_map
        count = extract<size_t>(state[index++]);
        for (i = 0; i < count; ++i)
            outcome_map->add(extract<std::string>(state[index++]));

        // load params
        n_theta = extract<size_t>(state[index++]);
        for (i = 0; i < pred_map->size(); ++i) {
            tuple tmp(state[index++]);
            boost::python::list oids(tmp[0]);
            boost::python::list t(tmp[1]);
            std::vector<pair<size_t, size_t> > a;

            size_t k = extract<size_t>(oids.attr("__len__")());
            assert (k == len(t));

            for (size_t j = 0; j < k; ++j) {
                size_t oid = extract<size_t>(oids[j]);
                size_t fid = extract<size_t>(t[j]);
                a.push_back(std::make_pair(oid, fid));
            }
            params->push_back(a);
        }
        // extract theta
        theta.reset(new double[n_theta]);
        for (i = 0;i < n_theta; ++i)
            theta[i] = extract<double>(state[index++]);
        m.m_pred_map = pred_map;
        m.m_outcome_map = outcome_map;
        m.m_params = params;
        m.m_n_theta = n_theta;
        m.m_theta = theta;
    }
};
#endif // PYTHON_MODULE }}}

} // namespace maxent
#endif /* ifndef MAXENTMODEL_H */


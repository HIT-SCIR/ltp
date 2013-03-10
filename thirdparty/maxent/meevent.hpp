/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * meevent.hpp  -  define the Event (samples) the Maxent framework
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jan-2003
 * Last Change : 26-Jun-2004.
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

#ifndef EVENT_H
#define EVENT_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ext_algorithm.hpp"

#include "eventspace.hpp"

namespace maxent {
using namespace std;

namespace me {
    // Maxent Event
struct Event {
    public:
    typedef size_t    pred_id_type;
    typedef size_t outcome_id_type;
    // to save memory, feature value is defined as float rather than double
    typedef pair<pred_id_type, float> context_type;

    Event():m_context(0), m_context_size(0),m_outcome(0), m_count(0){}
    Event( context_type* context,
            size_t context_size,
            size_t count):
        m_context(context), m_context_size(context_size),
        m_outcome(0), m_count(count) {}

    context_type*    m_context;
    size_t           m_context_size;
    outcome_id_type  m_outcome;
    size_t           m_count;

    size_t context_size() const { return m_context_size; }

    //first compare context (including fvalues) then outcome
    bool operator<(const Event& rhs) const {
        int ret = lexicographical_compare_3way(m_context, m_context +
                m_context_size, rhs.m_context, rhs.m_context +
                rhs.m_context_size);
        if (ret == 0)
            return m_outcome < rhs.m_outcome;
        else
            return ret < 0;
    }

    bool operator>(const Event& rhs) const {
        return rhs < *this;
    }

    // two events are equal if they have same context and outcome
    bool operator==(const Event& rhs) const {
        return m_outcome == rhs.m_outcome && is_same_context(rhs);
    }

    bool is_same_context(const Event& rhs) const {
            return (lexicographical_compare_3way(m_context, m_context +
                        m_context_size, rhs.m_context, rhs.m_context +
                        rhs.m_context_size) == 0); 
    }

    void set_outcome(outcome_id_type oid) { m_outcome = oid; }
    void set_prior(const double p) {}
};

typedef size_t outcome_id_type;
typedef std::string feature_type;
typedef std::string outcome_type;

typedef EventSpace<me::Event> MEEventSpace;
typedef MEEventSpace::featmap_type FeatMapType;

typedef ItemMap<feature_type> PredMapType;
typedef ItemMap<outcome_type> OutcomeMapType;
// ParamsType: each vector[pred_id] is a pair<outcome_id, feat_id> 
typedef std::vector<std::vector<pair<size_t, size_t> > > ParamsType;

} // namespace me
} // namespace maxent

#endif /* ifndef EVENT_H */


/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * rfevent.hpp  -  define the Event (sample) in the RandomField framework
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jan-2003
 * Last Change : 10-Mar-2004.
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

#ifndef RFEVENT_H
#define RFEVENT_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ext_algorithm.hpp>
#include "eventspace.hpp"

namespace maxent {
using namespace std;

namespace rf {
    // RandomField Event
struct Event {
    public:
    typedef size_t feat_id_type;
    typedef size_t outcome_id_type;
    // to save memory, feature value is defined as float rather than double
    // a context is a set of <feature, fvalue> pair
    typedef pair<feat_id_type, float> context_type;

    Event():m_context(0), m_context_size(0), m_count(0), m_prior(1.0) {}

    Event(context_type* context, size_t context_size, size_t count):
        m_context(context), m_context_size(context_size),
    m_count(count), m_prior(1.0){}

    context_type*   m_context;
    size_t       m_context_size;
    size_t       m_count;
    double       m_prior;    // store prior(x), if the prior is uniform 
                             // then the model becomes Maxent Model

    size_t context_size() const { return m_context_size; }

    //compare features (including fvalues)
    bool operator<(const Event& rhs) const {
        int ret = lexicographical_compare_3way(m_context,
                m_context + m_context_size,
                rhs.m_context, rhs.m_context + rhs.m_context_size);
        return ret < 0;
    }

    bool operator>(const Event& rhs) const {
        return rhs < *this;
    }

    // two events are equal if they have the same features
    bool operator==(const Event& rhs) const {
        bool eq = is_same_context(rhs);
        if (eq) assert(m_prior == rhs.m_prior);
        return eq;
    }

    bool is_same_context(const Event& rhs) const {
            return (lexicographical_compare_3way(m_context,
                m_context + m_context_size,
                rhs.m_context, rhs.m_context + rhs.m_context_size) == 0);
    }

    void set_outcome(outcome_id_type oid) {}

    void set_prior(const double p) {m_prior = p;}
};

typedef EventSpace<rf::Event> RFEventSpace;
typedef RFEventSpace::featmap_type featmap_type;
typedef std::string feature_type;

} // namespace rf
} // namespace maxent

#endif /* ifndef RFEVENT_H */


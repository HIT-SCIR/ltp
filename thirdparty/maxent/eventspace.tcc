/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * event.cpp  -  EventSpace used in Maxent/RandomField framework
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 29-Oct-2003
 * Last Change : 12-Jun-2006.
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

#include <algorithm>

namespace maxent {
using namespace std;

// create an event space.
//
// If a valid featmap is given, only feature in that featmap will be used.
// All other features will be discarded during \sa add_event().
//
// If no valid featmap is given, a new feature map will be created and all
// unknown features will be added to that feature map during \sa add_event().
//
// This behavior can be changed by setting \sa m_newfeat_mode
template<typename Ev>
EventSpace<Ev>::EventSpace(boost::shared_ptr<featmap_type> featmap, 
        boost::shared_ptr<outcomemap_type> outcomemap) {
    if (!featmap)
        m_feat_map.reset(new featmap_type);
    else
        m_feat_map = featmap;

    if (!outcomemap) 
        m_outcome_map.reset(new outcomemap_type);
    else 
        m_outcome_map = outcomemap;

    m_newfeat_mode = (!featmap && !outcomemap) ? true : false;
    // cerr << "EventSpace():new feat mode: " <<  m_newfeat_mode << endl;
}

template<typename Ev>
EventSpace<Ev>::~EventSpace() {
    for (size_t i = 0; i < this->size(); ++i) {
        delete[] (*this)[i].m_context;
    }
}


//Add an event (context, count, prior) to current event space.
//
//@param context A std::vector of pair<std::string, float> to indicate
//               the features and their values (must be >= 0)
//               occure in the event.
//@param outcome A std::string of outcome label, default is "", ignored in
//random field model
//@param count   How many time this event occurs in training set. Default = 1
//@param prior   the prior  P_0(x)
template<typename Ev>
void EventSpace<Ev>::add_event( const context_type& context,
        size_t count,
        const std::string& outcome, 
        double prior) {
    if (m_newfeat_mode) {
        // TODO: how to handle duplicate predicates?
        typename Ev::context_type* c =
            new typename Ev::context_type[context.size()];
        for (size_t i = 0; i < context.size(); ++i) {
            c[i].first = m_feat_map->add(context[i].first);
            c[i].second = context[i].second;

//            if (c[i].second < 0)
//                throw runtime_error("feature value must be non-negative in ME model");
        }

        sort(c,c + context.size());
        Ev e(c, context.size(), count);
        e.set_outcome(m_outcome_map->add(outcome));
        e.set_prior(prior);
        this->push_back(e);
    } else {
    size_t oid = m_outcome_map->id(outcome);

    // all outcomes in events must have been seen in training events
//    cerr << oid << " " << m_outcome_map->null_id << endl;
    if (oid == m_outcome_map->null_id) {
        cerr << "Invalid outcome:" << outcome 
            << " in heldout events, ignored." << endl;
        return;
    }
        // only previously seen features will be retained
        // unseen features are discarded
        vector<typename Ev::context_type> c;
        for (size_t i = 0; i < context.size(); ++i) {
            size_t fid = m_feat_map->id(context[i].first);
            if (fid != m_feat_map->null_id) {
//                if (context[i].second < 0)
//                    throw runtime_error("feature value must be non-negative in ME model");
                c.push_back(make_pair(fid, context[i].second));
            }
        }

        sort(c.begin(), c.end());

        Ev e;
        e.m_context = new typename Ev::context_type[c.size()];
        e.m_context_size = c.size();
        copy(c.begin(), c.end(), e.m_context);
        e.m_count   = count;
        e.set_outcome(oid);
        e.set_prior(prior);
        this->push_back(e);
    }
}

// merge events with the same context and remove events whose count < cutoff
template <typename Ev>
void EventSpace<Ev>::merge_events(size_t cutoff) {
    sort(this->begin(), this->end());

    //merge training events
    typename EventSpace<Ev>::iterator it = this->begin();
    typename EventSpace<Ev>::iterator end = this->end();
    typename EventSpace<Ev>::iterator p;
    while (it != end) {
        p = it + 1;
        while (p != end && *p == *it) {
            it->m_count += p->m_count;
            p->m_count = 0;
            ++p;
        }
        it = p;
    }

    // Apply cutoff and remove events with zero freq (merged events)
    //
    // I hate writing explicit delete operation.
    // But I have to do so since these elements will not be 
    // destoried in ~EventSpace() 
    // And they must be freed before calling remove_if() since remove_if()
    // simply overwrites these elements to be erased with remained elements.
    //
    // Is there a better way to handle memory without losing efficiency?
    // (vector<size_t> is too heavy)
    for (it = this->begin(); it != this->end(); ++it) {
        if (it->m_count < cutoff) {
             delete[] it->m_context;
            it->m_context = 0;
        }
    }
    this->erase(remove_if(this->begin(), this->end(),
                cutoffed_event(cutoff)), this->end());
}

} // namespace maxent

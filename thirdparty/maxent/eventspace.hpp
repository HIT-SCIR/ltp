/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * eventspace.hpp  -  An event space is made up of events and feature mapping
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Mar-2004
 * Last Change : 10-Mar-2004.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser GPL (LGPL) as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.
 */

#ifndef EVENTSPACE_H
#define EVENTSPACE_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include <vector>
#include <utility>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include "itemmap.hpp"

namespace maxent {
using namespace std;

// this class is responsible to free memory allocated for context on destory
// it's noncopyable, so can only has one instance.  When deleting elements
// from this vector, the caller is responsible to free those elements. Anyway
// it can be used in boost::shared_ptr<>.
template<typename Ev>
class EventSpace : public std::vector<Ev>, public boost::noncopyable {
    public:
        typedef ItemMap<std::string> featmap_type;
        typedef ItemMap<std::string> outcomemap_type;
        typedef std::vector<pair<std::string, float> > context_type;

        EventSpace(boost::shared_ptr<featmap_type> featmap =
                boost::shared_ptr<featmap_type>(), 
                boost::shared_ptr<outcomemap_type> outcomemap =
                boost::shared_ptr<outcomemap_type>()
                );

        virtual ~EventSpace();

        void add_event(const context_type& context, size_t count = 1,
                const std::string& outcome = "", double prior = 1.0);

        void add_event(const context_type& context, size_t count = 1,
                double prior = 1.0) { add_event(context, count, "", prior);}

        void set_newfeat_mode(bool f) { m_newfeat_mode = f; }

        bool newfeat_mode() const { return m_newfeat_mode; }

        void merge_events(size_t cutoff);

        boost::shared_ptr<featmap_type> feat_map() {
            return m_feat_map;
        }

        boost::shared_ptr<outcomemap_type> outcome_map() {
            return m_outcome_map;
        }

    private:
        struct cutoffed_event {
            cutoffed_event(size_t cutoff):m_cutoff(cutoff) {}
            bool operator()(const Ev& ev) const {
                return ev.m_count < m_cutoff;
            }
            size_t m_cutoff;
        };
        boost::shared_ptr<featmap_type> m_feat_map;
        boost::shared_ptr<outcomemap_type> m_outcome_map;
        bool m_newfeat_mode;  // whether new feature can be added
};

} // namespace maxent

#include "eventspace.tcc"
#endif /* ifndef EVENTSPACE_H */


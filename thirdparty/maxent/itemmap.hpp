/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * itemmap.hpp  -  generic item <--> id map class
 *
 * Copyright (C) 2002 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 31-Dec-2002
 * Last Change : 25-Dec-2004.
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

#ifndef ITEMMAP_H
#define ITEMMAP_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include "hash_map.hpp"

#if !defined(_STLPORT_VERSION) && defined(_MSC_VER) && ((_MSC_VER >= 1300) || defined(__INTEL_COMPILER))
    // workaround for MSVC7's hash_map declaration
    template <typename T, typename HashFunc = hash_compare<T, _STD less<T> >, typename EqualKey = _STD allocator< _STD pair<const T, size_t> > >
#else
    template <typename T, typename HashFunc = hash<T>, typename EqualKey = std::equal_to<T> >
#endif
class ItemMap {
    public:
        typedef T      item_type;
        typedef size_t id_type;
        typedef hash_map <T, id_type, HashFunc, EqualKey>         hash_map_type;
        // static const size_t null_id = ~(ItemMap::id_type)0;
        static const size_t null_id;
        typedef typename std::vector<T>::iterator       iterator;
        typedef typename std::vector<T>::const_iterator const_iterator;

        ItemMap(){}

        ~ItemMap();

        iterator begin() { return m_index.begin(); }

        iterator end() { return m_index.end(); }

        const_iterator begin() const { return m_index.begin(); }

        const_iterator end() const { return m_index.end(); }

        size_t size() const { return m_index.size(); }

        bool empty() const { return m_index.empty(); }

        void clear();

        /**
         * add a item into dict return new item's id
         * if the item already exists simply return its id
         */
        id_type add(const T& f);

        /**
         * get a item's id (index in dict)
         * if the item does not exist return null_id
         */
        id_type id(const T& f) const {
            typename hash_map_type::const_iterator it = m_hashdict.find(f);
            if (it == m_hashdict.end())
                return null_id;
            return it->second;
            // return has_item(f) ? m_hashdict[f] : null_id;
        }

        bool has_item(const T& f) const {
            return m_hashdict.find(f) != m_hashdict.end();
        }

        const T& operator[](id_type id) const {
            return m_index[id];
        }

    private:
        mutable hash_map_type m_hashdict;
        std::vector<T>        m_index;
};

template <typename T, typename HashFunc, typename EqualKey >
const size_t ItemMap<T, HashFunc, EqualKey>::null_id =
~(typename ItemMap<T, HashFunc, EqualKey>::id_type)(0); // -1 is null_id
// (typename ItemMap<T, HashFunc, EqualKey>::id_type)(-1); // -1 is null_id
#include "itemmap.tcc"
#endif /* ifndef ITEMMAP_H */

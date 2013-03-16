/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * itemmap.cpp  -  description
 *
 * Copyright (C) 2002 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 31-Dec-2002
 * Last Change : 11-Mar-2004.
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

#include <cassert>

#include <fstream>
#include <stdexcept>

using namespace std;

//ItemMap<T, HashFunc, typename EqualKey>::ItemMap(const string& filename)
//{
//    load(filename);
//}

template<typename T, typename HashFunc, typename EqualKey>
ItemMap<T, HashFunc, EqualKey>::~ItemMap() {
    clear();
}

template<typename T, typename HashFunc, typename EqualKey>
void ItemMap<T, HashFunc, EqualKey>::clear() {
    m_index.clear();
    m_hashdict.clear();
}

/**
 * load feature map of T from a binary file.
 */
//void ItemMap<T, HashFunc, typename EqualKey>::load(const string& filename) {
//    assert(!filename.empty());
//
//    ifstream in(filename.c_str(),ios::binary);
//    if (!in)
//        throw runtime_error("unable to open featmap file to read");
//
//    load(in);
//}

/*
template<>
void ItemMap<string>::load(istream& is) {
    clear();
    size_t n;
    is.read((char*)&n,sizeof(n));

    char buf[4000];
    size_t len;
    string feat;
    id_type  index = 0;
    for (size_t i = 0;i < n; ++i) {
        is.read((char*)&len,sizeof(len));
        if (len >= 4000)
            throw runtime_error("buffer overflow when loading");
        is.read((char*)buf,sizeof(char) * len);
        buf[len] = '\0';
        //feat = buf;
        m_hashdict[buf] = index;
        m_index.push_back(buf);
        ++index;
    }
}
*/

/**
 * save feature map of T to given file one word per line
 * the feature should can be write through << operator
 */
/*
template<>
void ItemMap<string>::save(ostream& os) {
    size_t n = size();
    os.write((char*)&n, sizeof(n));

    for (size_t i = 0;i < n; ++i) {
        string& s = m_index[i];
        size_t len = s.size();
        os.write((char*)&len,sizeof(len));
        os.write((char*)s.c_str(),sizeof(char) * len);
    }
}
*/

/**
 * save feature map of T to a binary file
 */
//void ItemMap<T, HashFunc, typename EqualKey>::save(const string& filename) {
//    assert(!filename.empty());
//
//    ofstream out(filename.c_str(),ios::binary);
//    if (!out)
//        throw runtime_error("unable to open wordmap file to write");
//    save(out);
//}

template<typename T, typename HashFunc, typename EqualKey>
typename ItemMap<T, HashFunc, EqualKey>::id_type ItemMap<T, HashFunc, EqualKey>::add(const T& f) {
    typename hash_map_type::const_iterator it = m_hashdict.find(f);
    if (it != m_hashdict.end())
        return it->second;

    id_type id = m_index.size();
    m_hashdict[f] = id;
    m_index.push_back(f);
    return id;
}


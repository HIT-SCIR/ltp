/*
 * LTP - Language Technology Platform
 *
 * Copyright (C) 2011-2013 HIT-SCIR
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */

/*
 * A library for mapping string to user specified type.
 * Modify from @file CharArrayHashFunc.h and @file CharArrayEqualFunc.h
 *  
 *  @author:    Mihai Surdeanu
 *  @modifier:  LI, Zhenhua
 *  @modifier:  LIU, Yijia
 */

#ifndef __STRING_MAP_HPP__
#define __STRING_MAP_HPP__

#ifdef _WIN32
#include <hash_map>
#else
#include <tr1/unordered_map>
#endif

#include <string.h>
#include <stdlib.h>

namespace ltp {
namespace utility {

typedef struct CharArrayEqualFunc {
    bool operator() (const char * s1, const char * s2) const {
        return (strcmp(s1, s2) == 0);
    }
} char_array_equal;

typedef struct CharArrayHashFunc {
#ifdef _WIN32
    : public stdext::hash_compare<const char *>
#endif
    /*
     * hash function from Mihai
     */
    size_t operator() (const char * s) const {
        /*int hashTemp = 0;

        for (unsigned int i = 0; s[i]; ++ i) {
            if (0 > hashTemp) hashTemp = (hashTemp << 1) + 1;
            else hashTemp = hashTemp << 1;
            hashTemp ^= s[i];
        }*/

        unsigned int hashTemp = 0;
        while (*s) {
            hashTemp = hashTemp * 101 + *s ++;
        }
        return (size_t(hashTemp));
    }

    bool operator() (const char * s1, const char * s2) const {
        return (strcmp(s1, s2) < 0);
    }

} char_array_hash;

template <class T>
class StringMap {
public:
#ifdef _WIN32
    typedef stdext::hash_map<const char *, T, char_array_hash> internal_map_t;
#else
    typedef std::tr1::unordered_map<const char *, T, char_array_hash, char_array_equal> internal_map_t;
#endif  // end for _WIN32
    typedef typename internal_map_t::iterator       iterator;
    typedef typename internal_map_t::const_iterator const_iterator;

    StringMap() {
    }

    ~StringMap() {
        clear();
    }

    void clear() {
        const char * last = NULL;
        for (iterator it = _map.begin(); it != _map.end(); ++ it) {
            if (last != NULL) {
                free( (void *)last );
            }
            last = it->first;
        }
        if (last != NULL) {
            free( (void *)last );
        }
        _map.clear();
    }

    bool set( const char * key, const T &val ) {
        if (contains(key)) {
            return false;
        }

        int len = 0;
        for (; key[len] != 0; ++ len);

        char * new_key = (char *) malloc( (len + 1) * sizeof(char) );
        for (int i = 0; i < len; ++ i) {
            new_key[i] = key[i];
        }

        new_key[len] = 0;
        _map[new_key] = val;

        return true;
    }

    void unsafe_set(const char * key, const T &val ) {
        int len = 0;
        for (; key[len] != 0; ++ len);

        char * new_key = (char *) malloc( (len + 1) * sizeof(char) );
        for (int i = 0; i < len; ++ i) {
            new_key[i] = key[i];
        }

        new_key[len] = 0;
        _map[new_key] = val;

    }

    bool overwrite( const char * key, const T &val ) {
        if (contains(key)) {
            iterator it = _map.find(key);
            it->second = val;
            return true;
        } else {
            return set(key, val);
        }
        return false;
    }

    bool get( const char * key, T& val) const {
        const_iterator it;

        if ((it = _map.find(key)) != _map.end()) {
            val = it->second;
            return true;
        }

        return false;
    }

    T* get(const char * key) {
        iterator it = _map.find(key);

        if (it != _map.end()) {
            return &(it->second);
        }

        return NULL;
    }

    void unsafe_get( const char * key, T& val) {
        val = _map.find(key)->second;
    }

    bool contains( const char * key ) const {
        if (_map.find(key) != _map.end()) {
            return true;
        }

        return false;
    }

    size_t size() const {
        return _map.size();
    }

    bool empty() const {
        return _map.empty();
    }

    const_iterator begin() const {
        return _map.begin();
    }

    const_iterator end() const {
        return _map.end();
    }

    iterator mbegin() {
        return _map.begin();
    }

    iterator mend() {
        return _map.end();
    }

protected:
    internal_map_t _map;
};  // end for class StringMap

}   // end for namespace utility
}   // end for namespace ltp


#endif  // end for __STRING_MAP_HPP__

/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * hash_map.hpp  -  wrapper header as a workaround for several different ways
 * of using hash_map/hash_set since this is not ISO standard.
 *
 * After inclusion of this file hash and hash_map are exported into the global
 * namespace.
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 26-Jun-2004
 * Last Change : 25-Dec-2004.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HASH_MAP_HPP
#define HASH_MAP_HPP

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string>

#if defined(_STLPORT_VERSION)
    #include <hash_map>
    #include <hash_set>
    using std::hash;
    using std::hash_map;
    using std::hash_set;
#else // not using STLPORT

    #ifdef __GNUC__
        #if __GNUC__ >= 3
            #include <ext/hash_map>
            #include <ext/hash_set>
            namespace __gnu_cxx {
                template <>
                struct hash<std::string> {
                    size_t operator()(const std::string& s) const {
                        unsigned long __h = 0;
                        for (unsigned i = 0;i < s.size();++i)
                            __h ^= (( __h << 5) + (__h >> 2) + s[i]);

                        return size_t(__h);
                    }
                };
            };
            using __gnu_cxx::hash_map;
            using __gnu_cxx::hash;
        #else // GCC 2.x
            #include <hash_map>
            #include <hash_set>
            namespace std {
                struct hash<std::string> {
                    size_t operator()(const std::string& s) const {
                        unsigned long __h = 0;
                        for (unsigned i = 0;i < s.size();++i)
                            __h ^= (( __h << 5) + (__h >> 2) + s[i]);

                        return size_t(__h);
                    }
                };
            };
            using std::hash_map;
            using std::hash_set;
            using std::hash;
        #endif // end GCC >= 3
    #elif defined(_MSC_VER) && ((_MSC_VER >= 1300) || defined(__INTEL_COMPILER))
        // we only support MSVC7+ and Intel C++ 8.0
        #include <hash_map>
        #include <hash_set>
        namespace stdext {
            inline size_t hash_value(const std::string& s) {
                unsigned long __h = 0;
                for (unsigned i = 0;i < s.size();++i)
                    __h ^= (( __h << 5) + (__h >> 2) + s[i]);

                return size_t(__h);
            }
        }
        //using std::hash_map; // _MSC_EXTENSIONS, though DEPRECATED
        //using std::hash_set;
        using stdext::hash_map;
        using stdext::hash_set;
        using stdext::hash_compare;
    #else
        #error unknown compiler
    #endif //GCC or MSVC7+
#endif // end STLPORT

#endif /* ifndef HASH_MAP_HPP */


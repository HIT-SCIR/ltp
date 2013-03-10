/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * ext_algorithm.hpp  -  wrapper header as a workaround for several different ways
 * of using non-standard STL algorithms, since they are not in ISO standard.
 *
 * After inclusion of this file non-standard STL algorithms like
 * lexicographical_compare_3way are exported into the global namespace.
 *
 * Copyright (C) 2004 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 26-Jun-2004
 * Last Change : 01-Jul-2004.
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

#ifndef EXT_ALGORITHM_HPP
#define EXT_ALGORITHM_HPP

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if defined(_STLPORT_VERSION) || (defined (__GNUC__) && __GNUC__ <= 2)
    #include <algorithm>
    using std::copy_n;
    using std::lexicographical_compare_3way;
#else 
    // not using STLPORT, not GCC 2.9x
    #if (defined(__GNUC__) && __GNUC__ >= 3)
        #include <ext/algorithm>
        using __gnu_cxx::copy_n;
        using __gnu_cxx::lexicographical_compare_3way;
    #elif defined(_MSC_VER) && (_MSC_VER >= 1300) 
        // we only support MSVC7+
        // the following non-standard functions are modified from STLPORT
        // lexicographical_compare_3way
        template <class _InputIter1, class _InputIter2>
        int lexicographical_compare_3way(_InputIter1 __first1, _InputIter1 __last1,
                                           _InputIter2 __first2, _InputIter2 __last2)
        {
          while (__first1 != __last1 && __first2 != __last2) {
            if (*__first1 < *__first2)
              return -1;
            if (*__first2 < *__first1)
              return 1;
            ++__first1;
            ++__first2;
          }
          if (__first2 == __last2) {
            return !(__first1 == __last1);
          }
          else {
            return -1;
          }
        }

        // copy_n
        #include <utility>
        template <class _InputIter, class _Size, class _OutputIter>
        std::pair<_InputIter, _OutputIter> copy_n(_InputIter __first, _Size __count,
                                               _OutputIter __result) {
          for ( ; __count > 0; --__count) {
            *__result = *__first;
            ++__first;
            ++__result;
          }
          return std::pair<_InputIter, _OutputIter>(__first, __result);
        }
    #else 
        #error unknown compiler
    #endif // end GCC >= 3
#endif // end STLPORT or GCC 2.9x

#endif /* ifndef EXT_ALGORITHM_HPP */


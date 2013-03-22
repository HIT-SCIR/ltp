// Stream iterators over text lines
// Modified from istream_iterator implementation from GNU ISO C++ Library
// Typical code:
//    token_mem_iterator<> line(cin);
//    token_mem_iterator<> lend;
//    for (; line != lend; ++line) {
//        cout << string(line->first, line->second - line->first) << endl;
//    }
//    here the iterator is pair<const char*, const char*> indicating a range
//    of buffer

// Copyright (C) 2001 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING.  If not, write to the Free
// Software Foundation, 59 Temple Place - Suite 330, Boston, MA 02111-1307,
// USA.

// As a special exception, you may use this file as part of a free software
// library without restriction.  Specifically, if other files instantiate
// templates or use macros or inline functions from this file, or you compile
// this file and link it with other files to produce an executable, this
// file does not by itself cause the resulting executable to be covered by
// the GNU General Public License.  This exception does not however
// invalidate any other reasons why the executable file might be covered by
// the GNU General Public License.

/** @file stream_iterator.h
 *  This is an internal header file, included by other library headers.
 *  You should not attempt to use it directly.
 */

#ifndef token_mem_iterator_H
#define token_mem_iterator_H


#include <string>
#include <iostream>
#include <iterator>
#include <utility>

using namespace std;
template<typename _Tp = pair<const char*, const char*>, typename _CharT = char, 
    typename _Traits = char_traits<_CharT>, typename _Dist = ptrdiff_t> 
    class token_mem_iterator 
    : public iterator<input_iterator_tag, _Tp, _Dist, const _Tp*, const _Tp&>
{
    public:
        typedef _CharT                         char_type;
        typedef _Traits                        traits_type;

    private:
        const char* _M_data_begin;
        const char* _M_data_end;
        _Tp 		_M_value;
        bool 		_M_ok;
        bool        _M_flag[255];

    public:      
        token_mem_iterator() : _M_data_begin(0), _M_data_end(0), _M_ok(false) {}

        token_mem_iterator(const char* begin, const char* end, const char* delim = " \t\f\v\r") :
            _M_data_begin(begin), _M_data_end(end) { 
            _M_value.first = begin;
            _M_value.second = begin;
            memset(_M_flag, false, 255);
            for (const char* p = delim; *p != '\0'; ++p)
                _M_flag[(unsigned char)*p] = true;
            _M_read();
        }

        // compiler generated copy constructor is ok
        // token_mem_iterator(const token_mem_iterator& __obj) 

        const _Tp& operator*() const { return _M_value; }

        const _Tp* operator->() const { return &(operator*()); }

        token_mem_iterator& operator++() 
            { _M_read(); return *this; }

        token_mem_iterator operator++(int)  
            {
                token_mem_iterator __tmp = *this;
                _M_read();
                return __tmp;
            }

        bool _M_equal(const token_mem_iterator& __x) const
            { return (_M_ok == __x._M_ok) && 
                (!_M_ok || (_M_data_begin == __x._M_data_begin &&
                            _M_data_end == __x._M_data_end));}

    private:      
        void _M_read()  // advance one token
            {
                _M_ok = (_M_value.first < _M_data_end) ? true : false;
                if (_M_ok) 
                {
                    const char* p = _M_value.second;
                    while (p < _M_data_end && _M_flag[(unsigned char)*p])
                        ++p;
                    _M_value.first = p;
                    while (p < _M_data_end && !_M_flag[(unsigned char)*p])
                        ++p;
                    _M_value.second = p;
                    _M_ok = (_M_value.first < _M_data_end) ? true : false;
                }
            }
};

template<typename _Tp, typename _CharT, typename _Traits, typename _Dist>
inline bool 
operator==(const token_mem_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
        const token_mem_iterator<_Tp, _CharT, _Traits, _Dist>& __y) 
{ return __x._M_equal(__y); }

template <class _Tp, class _CharT, class _Traits, class _Dist>
inline bool 
operator!=(const token_mem_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
        const token_mem_iterator<_Tp, _CharT, _Traits, _Dist>& __y) 
{ return !__x._M_equal(__y); }


#endif


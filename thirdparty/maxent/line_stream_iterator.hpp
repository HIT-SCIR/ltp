// Stream iterators over text lines
// Modified from istream_iterator implementation from GNU ISO C++ Library
// Typical code:
//    line_stream_iterator<> line(cin);
//    line_stream_iterator<> lend;
//    for (; line != lend; ++line) {
//        cout << *line;
//    }

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

#ifndef LINE_STREAM_ITERATOR_H
#define LINE_STREAM_ITERATOR_H


#include <string>
#include <iostream>
#include <iterator>

using namespace std;
template<typename _Tp = std::string, typename _CharT = char, 
    typename _Traits = char_traits<_CharT>, typename _Dist = ptrdiff_t> 
    class line_stream_iterator 
    : public iterator<input_iterator_tag, _Tp, _Dist, const _Tp*, const _Tp&>
{
    public:
        typedef _CharT                         char_type;
        typedef _Traits                        traits_type;
        typedef basic_istream<_CharT, _Traits> istream_type;

    private:
        istream_type* 	_M_stream;
        _Tp 		_M_value;
        bool 		_M_ok;

    public:      
        line_stream_iterator() : _M_stream(0), _M_ok(false) {}

        line_stream_iterator(istream_type& __s) : _M_stream(&__s) { _M_read(); }

        line_stream_iterator(const line_stream_iterator& __obj) 
            : _M_stream(__obj._M_stream), _M_value(__obj._M_value), 
        _M_ok(__obj._M_ok) 
        { }

        const _Tp& operator*() const { return _M_value; }

        const _Tp* operator->() const { return &(operator*()); }

        line_stream_iterator& operator++() 
            { _M_read(); return *this; }

        line_stream_iterator operator++(int)  
            {
                line_stream_iterator __tmp = *this;
                _M_read();
                return __tmp;
            }

        bool _M_equal(const line_stream_iterator& __x) const
            { return (_M_ok == __x._M_ok) && (!_M_ok || _M_stream == __x._M_stream);}

    private:      
        void _M_read() 
            {
                _M_ok = (_M_stream && *_M_stream) ? true : false;
                if (_M_ok) 
                {
                    // *_M_stream >> _M_value;
                    getline(*_M_stream, _M_value);
                    _M_ok = *_M_stream ? true : false;
                }
            }
};

template<typename _Tp, typename _CharT, typename _Traits, typename _Dist>
inline bool 
operator==(const line_stream_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
        const line_stream_iterator<_Tp, _CharT, _Traits, _Dist>& __y) 
{ return __x._M_equal(__y); }

template <class _Tp, class _CharT, class _Traits, class _Dist>
inline bool 
operator!=(const line_stream_iterator<_Tp, _CharT, _Traits, _Dist>& __x,
        const line_stream_iterator<_Tp, _CharT, _Traits, _Dist>& __y) 
{ return !__x._M_equal(__y); }


#endif


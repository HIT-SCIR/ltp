// Copyright (C) 2010-2012 Conrad Sanderson
// Copyright (C) 2010-2012 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Stanislav Funiak
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup span
//! @{


struct span_alt {};


template<typename Dummy = int>
class span_base
  {
  public:
  static const span_alt all;
  };


template<typename Dummy>
const span_alt span_base<Dummy>::all = span_alt();


class span : public span_base<>
  {
  public:

  uword a;
  uword b;
  bool  whole;
  
  inline
  span()
    : whole(true)
    {
    }
  
  
  inline
  span(const span_alt&)
    : whole(true)
    {
    }
  
  // TODO:
  // if the "explicit" keyword is removed or commented out,
  // the compiler will be able to automatically convert integers to an instance of the span class.
  // this is useful for Cube::operator()(span&, span&, span&),
  // but it might have unintended consequences or interactions elsewhere.
  // as such, removal of "explicit" needs thorough testing.
  inline
  explicit
  span(const uword in_a)
    : a(in_a)
    , b(in_a)
    , whole(false)
    {
    }
  
  
  inline
  span(const uword in_a, const uword in_b)
    : a(in_a)
    , b(in_b)
    , whole(false)
    {
    }

  };



//! @}

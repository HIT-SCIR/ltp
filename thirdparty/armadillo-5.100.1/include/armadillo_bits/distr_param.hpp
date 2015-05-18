// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



//! \addtogroup distr_param
//! @{



class distr_param
  {
  public:
  
  uword state;
  
  union
    {
    int    a_int;
    double a_double;
    };
  
  union
    {
    int    b_int;
    double b_double;
    };
  
  
  inline distr_param()
    : state(0)
    {
    }
  
  
  inline explicit distr_param(const int a, const int b)
    : state(1)
    , a_int(a)
    , b_int(b)
    {
    }
  
  
  inline explicit distr_param(const double a, const double b)
    : state(2)
    , a_double(a)
    , b_double(b)
    {
    }
  };



//! @}

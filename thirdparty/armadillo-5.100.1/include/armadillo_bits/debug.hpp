// Copyright (C) 2008-2015 Conrad Sanderson
// Copyright (C) 2008-2015 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Stanislav Funiak
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup debug
//! @{



template<typename T>
inline
std::ostream&
arma_stream_err1(std::ostream* user_stream)
  {
  static std::ostream* stream_err1 = &(ARMA_DEFAULT_OSTREAM);
  
  if(user_stream != NULL)
    {
    stream_err1 = user_stream;
    }
  
  return *stream_err1;
  }



template<typename T>
inline
std::ostream&
arma_stream_err2(std::ostream* user_stream)
  {
  static std::ostream* stream_err2 = &(ARMA_DEFAULT_OSTREAM);
  
  if(user_stream != NULL)
    {
    stream_err2 = user_stream;
    }
  
  return *stream_err2;
  }



inline
void
set_stream_err1(std::ostream& user_stream)
  {
  arma_stream_err1<char>(&user_stream);
  }



inline
void
set_stream_err2(std::ostream& user_stream)
  {
  arma_stream_err2<char>(&user_stream);
  }



inline
std::ostream&
get_stream_err1()
  {
  return arma_stream_err1<char>(NULL);
  }



inline
std::ostream&
get_stream_err2()
  {
  return arma_stream_err2<char>(NULL);
  }



//
// arma_stop

//! print a message to get_stream_err1() and/or throw a logic_error exception
template<typename T1>
arma_cold
arma_noinline
static
void
arma_stop(const T1& x)
  {
  #if defined(ARMA_PRINT_ERRORS)
    {
    std::ostream& out = get_stream_err1();
    
    out << '\n';
    out << "error: " << x << '\n';
    out << '\n';
    out.flush();
    }
  #else
    {
    arma_ignore(x);
    }
  #endif
  
  throw std::logic_error( std::string(x) );
  }



template<typename T1>
arma_cold
arma_noinline
static
void
arma_stop_bad_alloc(const T1& x)
  {
  #if defined(ARMA_PRINT_ERRORS)
    {
    std::ostream& out = get_stream_err2();
    
    out << '\n';
    out << "error: " << x << '\n';
    out << '\n';
    out.flush();
    }
  #else
    {
    arma_ignore(x);
    }
  #endif
  
  throw std::bad_alloc();
  }



//
// arma_bad

//! print a message to get_stream_err2() and/or throw a run-time error exception
template<typename T1>
arma_cold
arma_noinline
static
void
arma_bad(const T1& x, const bool hurl = true)
  {
  #if defined(ARMA_PRINT_ERRORS)
    {
    std::ostream& out = get_stream_err2();
    
    out << '\n';
    out << "error: " << x << '\n';
    out << '\n';
    out.flush();
    }
  #else
    {
    arma_ignore(x);
    }
  #endif
  
  if(hurl == true)
    {
    throw std::runtime_error( std::string(x) );
    }
  }



//
// arma_print


arma_cold
inline
void
arma_print()
  {
  get_stream_err1() << std::endl;
  }


template<typename T1>
arma_cold
arma_noinline
static
void
arma_print(const T1& x)
  {
  get_stream_err1() << x << std::endl;
  }



template<typename T1, typename T2>
arma_cold
arma_noinline
static
void
arma_print(const T1& x, const T2& y)
  {
  get_stream_err1() << x << y << std::endl;
  }



template<typename T1, typename T2, typename T3>
arma_cold
arma_noinline
static
void
arma_print(const T1& x, const T2& y, const T3& z)
  {
  get_stream_err1() << x << y << z << std::endl;
  }






//
// arma_sigprint

//! print a message the the log stream with a preceding @ character.
//! by default the log stream is cout.
//! used for printing the signature of a function
//! (see the arma_extra_debug_sigprint macro) 
inline
void
arma_sigprint(const char* x)
  {
  get_stream_err1() << "@ " << x;
  }



//
// arma_bktprint


inline
void
arma_bktprint()
  {
  get_stream_err1() << std::endl;
  }


template<typename T1>
inline
void
arma_bktprint(const T1& x)
  {
  get_stream_err1() << " [" << x << ']' << std::endl;
  }



template<typename T1, typename T2>
inline
void
arma_bktprint(const T1& x, const T2& y)
  {
  get_stream_err1() << " [" << x << y << ']' << std::endl;
  }






//
// arma_thisprint

inline
void
arma_thisprint(const void* this_ptr)
  {
  get_stream_err1() << " [this = " << this_ptr << ']' << std::endl;
  }



//
// arma_warn


//! print a message to the warn stream
template<typename T1>
arma_cold
arma_noinline
static
void
arma_warn(const bool state, const T1& x)
  {
  if(state==true)
    {
    get_stream_err2() << x << std::endl;
    }
  }


template<typename T1, typename T2>
arma_cold
arma_noinline
static
void
arma_warn(const bool state, const T1& x, const T2& y)
  {
  if(state==true)
    {
    get_stream_err2() << x << y << std::endl;
    }
  }


template<typename T1, typename T2, typename T3>
arma_cold
arma_noinline
static
void
arma_warn(const bool state, const T1& x, const T2& y, const T3& z)
  {
  if(state==true)
    {
    get_stream_err2() << x << y << z << std::endl;
    }
  }



//
// arma_check

//! if state is true, abort program
template<typename T1>
arma_hot
inline
void
arma_check(const bool state, const T1& x)
  {
  if(state==true)
    {
    arma_stop(arma_boost::str_wrapper(x));
    }
  }


template<typename T1, typename T2>
arma_hot
inline
void
arma_check(const bool state, const T1& x, const T2& y)
  {
  if(state==true)
    {
    arma_stop( std::string(x) + std::string(y) );
    }
  }


template<typename T1>
arma_hot
inline
void
arma_check_bad_alloc(const bool state, const T1& x)
  {
  if(state==true)
    {
    arma_stop_bad_alloc(x);
    }
  }



//
// arma_set_error


arma_hot
arma_inline
void
arma_set_error(bool& err_state, char*& err_msg, const bool expression, const char* message)
  {
  if(expression == true)
    {
    err_state = true;
    err_msg   = const_cast<char*>(message);
    }
  }




//
// functions for generating strings indicating size errors

arma_cold
arma_noinline
static
std::string
arma_incompat_size_string(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  std::stringstream tmp;
  
  tmp << x << ": incompatible matrix dimensions: " << A_n_rows << 'x' << A_n_cols << " and " << B_n_rows << 'x' << B_n_cols;
  
  return tmp.str();
  }



arma_cold
arma_noinline
static
std::string
arma_incompat_size_string(const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices, const char* x)
  {
  std::stringstream tmp;
  
  tmp << x << ": incompatible cube dimensions: " << A_n_rows << 'x' << A_n_cols << 'x' << A_n_slices << " and " << B_n_rows << 'x' << B_n_cols << 'x' << B_n_slices;
  
  return tmp.str();
  }



template<typename eT>
arma_cold
arma_noinline
static
std::string
arma_incompat_size_string(const subview_cube<eT>& Q, const Mat<eT>& A, const char* x)
  {
  std::stringstream tmp;
  
  tmp << x
      << ": interpreting matrix as cube with dimensions: "
      << A.n_rows << 'x' << A.n_cols << 'x' << 1
      << " or "
      << A.n_rows << 'x' << 1        << 'x' << A.n_cols
      << " or "
      << 1        << 'x' << A.n_rows << 'x' << A.n_cols
      << " is incompatible with cube dimensions: "
      << Q.n_rows << 'x' << Q.n_cols << 'x' << Q.n_slices;
      
  return tmp.str();
  }



//
// functions for checking whether two matrices have the same dimensions



arma_inline
arma_hot
void
arma_assert_same_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



//! stop if given matrices have different sizes
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Mat<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



//! stop if given proxies have different sizes
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Proxy<eT1>& A, const Proxy<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.get_n_rows();
  const uword A_n_cols = A.get_n_cols();
  
  const uword B_n_rows = B.get_n_rows();
  const uword B_n_cols = B.get_n_cols();
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview<eT1>& A, const subview<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Mat<eT1>& A, const subview<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Mat<eT1>& A, const Proxy<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.get_n_rows();
  const uword B_n_cols = B.get_n_cols();
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Proxy<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.get_n_rows();
  const uword A_n_cols = A.get_n_cols();
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Proxy<eT1>& A, const subview<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.get_n_rows();
  const uword A_n_cols = A.get_n_cols();
  
  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview<eT1>& A, const Proxy<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  
  const uword B_n_rows = B.get_n_rows();
  const uword B_n_cols = B.get_n_cols();
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



//
// functions for checking whether two cubes have the same dimensions



arma_hot
inline
void
arma_assert_same_size(const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices, const char* x)
  {
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) || (A_n_slices != B_n_slices) )
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, A_n_slices, B_n_rows, B_n_cols, B_n_slices, x) );
    }
  }



//! stop if given cubes have different sizes
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Cube<eT1>& A, const Cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != B.n_slices) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Cube<eT1>& A, const subview_cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != B.n_slices) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview_cube<eT1>& A, const Cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != B.n_slices) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview_cube<eT1>& A, const subview_cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != B.n_slices))
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



//! stop if given cube proxies have different sizes
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const ProxyCube<eT1>& A, const ProxyCube<eT2>& B, const char* x)
  {
  const uword A_n_rows   = A.get_n_rows();
  const uword A_n_cols   = A.get_n_cols();
  const uword A_n_slices = A.get_n_slices();
  
  const uword B_n_rows   = B.get_n_rows();
  const uword B_n_cols   = B.get_n_cols();
  const uword B_n_slices = B.get_n_slices();
  
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) || (A_n_slices != B_n_slices))
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, A_n_slices, B_n_rows, B_n_cols, B_n_slices, x) );
    }
  }



//
// functions for checking whether a cube or subcube can be interpreted as a matrix (i.e. single slice)



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Cube<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != 1) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, 1, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Mat<eT1>& A, const Cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (1 != B.n_slices) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, 1, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const subview_cube<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (A.n_slices != 1) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, A.n_slices, B.n_rows, B.n_cols, 1, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_same_size(const Mat<eT1>& A, const subview_cube<eT2>& B, const char* x)
  {
  if( (A.n_rows != B.n_rows) || (A.n_cols != B.n_cols) || (1 != B.n_slices) )
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, 1, B.n_rows, B.n_cols, B.n_slices, x) );
    }
  }



template<typename eT, typename T1>
inline
void
arma_assert_cube_as_mat(const Mat<eT>& M, const T1& Q, const char* x, const bool check_compat_size)
  {
  const uword Q_n_rows   = Q.n_rows;
  const uword Q_n_cols   = Q.n_cols;
  const uword Q_n_slices = Q.n_slices;
  
  const uword M_vec_state = M.vec_state;
  
  if(M_vec_state == 0)
    {
    if( ( (Q_n_rows == 1) || (Q_n_cols == 1) || (Q_n_slices == 1) ) == false )
      {
      std::stringstream tmp;
        
      tmp << x
          << ": can't interpret cube with dimensions "
          << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices 
          << " as a matrix; one of the dimensions must be 1";
      
      arma_stop( tmp.str() );
      }
    }
  else
    {
    if(Q_n_slices == 1)
      {
      if( (M_vec_state == 1) && (Q_n_cols != 1) )
        {
        std::stringstream tmp;
        
        tmp << x
            << ": can't interpret cube with dimensions "
            << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
            << " as a column vector";
        
        arma_stop( tmp.str() );
        }
      
      if( (M_vec_state == 2) && (Q_n_rows != 1) )
        {
        std::stringstream tmp;
        
        tmp << x
            << ": can't interpret cube with dimensions "
            << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
            << " as a row vector";
        
        arma_stop( tmp.str() );
        }
      }
    else
      {
      if( (Q_n_cols != 1) && (Q_n_rows != 1) )
        {
        std::stringstream tmp;
        
        tmp << x
            << ": can't interpret cube with dimensions "
            << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
            << " as a vector";
        
        arma_stop( tmp.str() );
        }
      }
    }
  
  
  if(check_compat_size == true)
    {
    const uword M_n_rows = M.n_rows;
    const uword M_n_cols = M.n_cols;
    
    if(M_vec_state == 0)
      {
      if(
          (
          ( (Q_n_rows == M_n_rows) && (Q_n_cols   == M_n_cols) )
          ||
          ( (Q_n_rows == M_n_rows) && (Q_n_slices == M_n_cols) )
          ||
          ( (Q_n_cols == M_n_rows) && (Q_n_slices == M_n_cols) )
          )
          == false
        )
        {
        std::stringstream tmp;
        
        tmp << x
            << ": can't interpret cube with dimensions "
            << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
            << " as a matrix with dimensions "
            << M_n_rows << 'x' << M_n_cols;
        
        arma_stop( tmp.str() );
        }
      }
    else
      {
      if(Q_n_slices == 1)
        {
        if( (M_vec_state == 1) && (Q_n_rows != M_n_rows) )
          {
          std::stringstream tmp;
          
          tmp << x
              << ": can't interpret cube with dimensions "
              << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
              << " as a column vector with dimensions "
              << M_n_rows << 'x' << M_n_cols;
          
          arma_stop( tmp.str() );
          }
        
        if( (M_vec_state == 2) && (Q_n_cols != M_n_cols) )
          {
          std::stringstream tmp;
          
          tmp << x
              << ": can't interpret cube with dimensions "
              << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
              << " as a row vector with dimensions "
              << M_n_rows << 'x' << M_n_cols;
          
          arma_stop( tmp.str() );
          }
        }
      else
        {
        if( ( (M_n_cols == Q_n_slices) || (M_n_rows == Q_n_slices) ) == false )
          {
          std::stringstream tmp;
          
          tmp << x
              << ": can't interpret cube with dimensions "
              << Q_n_rows << 'x' << Q_n_cols << 'x' << Q_n_slices
              << " as a vector with dimensions "
              << M_n_rows << 'x' << M_n_cols;
          
          arma_stop( tmp.str() );
          }
        }
      }
    }
  }



//
// functions for checking whether two matrices have dimensions that are compatible with the matrix multiply operation



arma_hot
inline
void
arma_assert_mul_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  if(A_n_cols != B_n_rows)
    {
    arma_stop( arma_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



//! stop if given matrices are incompatible for multiplication
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_mul_size(const Mat<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_cols = A.n_cols;
  const uword B_n_rows = B.n_rows;
  
  if(A_n_cols != B_n_rows)
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A_n_cols, B_n_rows, B.n_cols, x) );
    }
  }



//! stop if given matrices are incompatible for multiplication
template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_mul_size(const Mat<eT1>& A, const Mat<eT2>& B, const bool do_trans_A, const bool do_trans_B, const char* x)
  {
  const uword final_A_n_cols = (do_trans_A == false) ? A.n_cols : A.n_rows;
  const uword final_B_n_rows = (do_trans_B == false) ? B.n_rows : B.n_cols;
    
  if(final_A_n_cols != final_B_n_rows)
    {
    const uword final_A_n_rows = (do_trans_A == false) ? A.n_rows : A.n_cols;
    const uword final_B_n_cols = (do_trans_B == false) ? B.n_cols : B.n_rows;
    
    arma_stop( arma_incompat_size_string(final_A_n_rows, final_A_n_cols, final_B_n_rows, final_B_n_cols, x) );
    }
  }



template<const bool do_trans_A, const bool do_trans_B>
arma_hot
inline
void
arma_assert_trans_mul_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  const uword final_A_n_cols = (do_trans_A == false) ? A_n_cols : A_n_rows;
  const uword final_B_n_rows = (do_trans_B == false) ? B_n_rows : B_n_cols;
    
  if(final_A_n_cols != final_B_n_rows)
    {
    const uword final_A_n_rows = (do_trans_A == false) ? A_n_rows : A_n_cols;
    const uword final_B_n_cols = (do_trans_B == false) ? B_n_cols : B_n_rows;
    
    arma_stop( arma_incompat_size_string(final_A_n_rows, final_A_n_cols, final_B_n_rows, final_B_n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_mul_size(const Mat<eT1>& A, const subview<eT2>& B, const char* x)
  {
  if(A.n_cols != B.n_rows)
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, B.n_rows, B.n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_mul_size(const subview<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  if(A.n_cols != B.n_rows)
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, B.n_rows, B.n_cols, x) );
    }
  }



template<typename eT1, typename eT2>
arma_hot
inline
void
arma_assert_mul_size(const subview<eT1>& A, const subview<eT2>& B, const char* x)
  {
  if(A.n_cols != B.n_rows)
    {
    arma_stop( arma_incompat_size_string(A.n_rows, A.n_cols, B.n_rows, B.n_cols, x) );
    }
  }



template<typename T1>
arma_hot
inline
void
arma_assert_blas_size(const T1& A)
  {
  if(sizeof(uword) >= sizeof(blas_int))
    {
    bool overflow;
    
    overflow = (A.n_rows > ARMA_MAX_BLAS_INT);
    overflow = (A.n_cols > ARMA_MAX_BLAS_INT) || overflow;
    
    if(overflow)
      {
      arma_bad("integer overflow: matrix dimensions are too large for integer type used by BLAS and LAPACK");
      }
    }
  }



template<typename T1, typename T2>
arma_hot
inline
void
arma_assert_blas_size(const T1& A, const T2& B)
  {
  if(sizeof(uword) >= sizeof(blas_int))
    {
    bool overflow;
    
    overflow = (A.n_rows > ARMA_MAX_BLAS_INT);
    overflow = (A.n_cols > ARMA_MAX_BLAS_INT) || overflow;
    overflow = (B.n_rows > ARMA_MAX_BLAS_INT) || overflow;
    overflow = (B.n_cols > ARMA_MAX_BLAS_INT) || overflow;
    
    if(overflow)
      {
      arma_bad("integer overflow: matrix dimensions are too large for integer type used by BLAS and LAPACK");
      }
    }
  }



template<typename T1>
arma_hot
inline
void
arma_assert_atlas_size(const T1& A)
  {
  if(sizeof(uword) >= sizeof(int))
    {
    bool overflow;
    
    overflow = (A.n_rows > INT_MAX);
    overflow = (A.n_cols > INT_MAX) || overflow;
    
    if(overflow)
      {
      arma_bad("integer overflow: matrix dimensions are too large for integer type used by ATLAS");
      }
    }
  }



template<typename T1, typename T2>
arma_hot
inline
void
arma_assert_atlas_size(const T1& A, const T2& B)
  {
  if(sizeof(uword) >= sizeof(int))
    {
    bool overflow;
    
    overflow = (A.n_rows > INT_MAX);
    overflow = (A.n_cols > INT_MAX) || overflow;
    overflow = (B.n_rows > INT_MAX) || overflow;
    overflow = (B.n_cols > INT_MAX) || overflow;
    
    if(overflow)
      {
      arma_bad("integer overflow: matrix dimensions are too large for integer type used by ATLAS");
      }
    }
  }



//
// macros


// #define ARMA_STRING1(x) #x
// #define ARMA_STRING2(x) ARMA_STRING1(x)
// #define ARMA_FILELINE  __FILE__ ": " ARMA_STRING2(__LINE__)


#if defined(ARMA_NO_DEBUG)
  
  #undef ARMA_EXTRA_DEBUG
  
  #define arma_debug_print                   true ? (void)0 : arma_print
  #define arma_debug_warn                    true ? (void)0 : arma_warn
  #define arma_debug_check                   true ? (void)0 : arma_check
  #define arma_debug_set_error               true ? (void)0 : arma_set_error
  #define arma_debug_assert_same_size        true ? (void)0 : arma_assert_same_size
  #define arma_debug_assert_mul_size         true ? (void)0 : arma_assert_mul_size
  #define arma_debug_assert_trans_mul_size   true ? (void)0 : arma_assert_trans_mul_size
  #define arma_debug_assert_cube_as_mat      true ? (void)0 : arma_assert_cube_as_mat
  #define arma_debug_assert_blas_size        true ? (void)0 : arma_assert_blas_size
  #define arma_debug_assert_atlas_size       true ? (void)0 : arma_assert_atlas_size
  
#else
  
  #define arma_debug_print                 arma_print
  #define arma_debug_warn                  arma_warn
  #define arma_debug_check                 arma_check
  #define arma_debug_set_error             arma_set_error
  #define arma_debug_assert_same_size      arma_assert_same_size
  #define arma_debug_assert_mul_size       arma_assert_mul_size
  #define arma_debug_assert_trans_mul_size arma_assert_trans_mul_size
  #define arma_debug_assert_cube_as_mat    arma_assert_cube_as_mat
  #define arma_debug_assert_blas_size      arma_assert_blas_size
  #define arma_debug_assert_atlas_size     arma_assert_atlas_size
  
#endif



#if defined(ARMA_EXTRA_DEBUG)
  
  #define arma_extra_debug_sigprint       arma_sigprint(ARMA_FNSIG); arma_bktprint
  #define arma_extra_debug_sigprint_this  arma_sigprint(ARMA_FNSIG); arma_thisprint
  #define arma_extra_debug_print          arma_print
  #define arma_extra_debug_warn           arma_warn
  #define arma_extra_debug_check          arma_check

#else
  
  #define arma_extra_debug_sigprint        true ? (void)0 : arma_bktprint
  #define arma_extra_debug_sigprint_this   true ? (void)0 : arma_thisprint
  #define arma_extra_debug_print           true ? (void)0 : arma_print
  #define arma_extra_debug_warn            true ? (void)0 : arma_warn
  #define arma_extra_debug_check           true ? (void)0 : arma_check
 
#endif




#if defined(ARMA_EXTRA_DEBUG)

  namespace junk
    {
    class arma_first_extra_debug_message
      {
      public:
      
      inline
      arma_first_extra_debug_message()
        {
        union
          {
          unsigned short a;
          unsigned char  b[sizeof(unsigned short)];
          } endian_test;
          
        endian_test.a = 1;
        
        const bool  little_endian = (endian_test.b[0] == 1);
        const char* nickname      = ARMA_VERSION_NAME;
        
        std::ostream& out = get_stream_err1();
        
        out << "@ ---" << '\n';
        out << "@ Armadillo "
            << arma_version::major << '.' << arma_version::minor << '.' << arma_version::patch
            << " (" << nickname << ")\n";
        
        out << "@ arma_config::use_wrapper  = " << arma_config::use_wrapper  << '\n';
        out << "@ arma_config::use_cxx11    = " << arma_config::use_cxx11    << '\n';
        out << "@ arma_config::openmp       = " << arma_config::openmp       << '\n';
        out << "@ arma_config::lapack       = " << arma_config::lapack       << '\n';
        out << "@ arma_config::blas         = " << arma_config::blas         << '\n';
        out << "@ arma_config::arpack       = " << arma_config::arpack       << '\n';
        out << "@ arma_config::superlu      = " << arma_config::superlu      << '\n';
        out << "@ arma_config::atlas        = " << arma_config::atlas        << '\n';
        out << "@ arma_config::hdf5         = " << arma_config::hdf5         << '\n';
        out << "@ arma_config::good_comp    = " << arma_config::good_comp    << '\n';
        out << "@ arma_config::extra_code   = " << arma_config::extra_code   << '\n';
        out << "@ arma_config::mat_prealloc = " << arma_config::mat_prealloc << '\n';
        out << "@ sizeof(void*)    = " << sizeof(void*)    << '\n';
        out << "@ sizeof(int)      = " << sizeof(int)      << '\n';
        out << "@ sizeof(long)     = " << sizeof(long)     << '\n';
        out << "@ sizeof(uword)    = " << sizeof(uword)    << '\n';
        out << "@ sizeof(blas_int) = " << sizeof(blas_int) << '\n';
        out << "@ little_endian    = " << little_endian    << '\n';
        out << "@ ---" << std::endl;
        }
      
      };
    
    static arma_first_extra_debug_message arma_first_extra_debug_message_run;
    }

#endif



//! @}

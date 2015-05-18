// Copyright (C) 2013-2015 Conrad Sanderson
// Copyright (C) 2013-2015 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup arma_rng
//! @{


#if defined(ARMA_RNG_ALT)
  #undef ARMA_USE_EXTERN_CXX11_RNG
#endif


#if !defined(ARMA_USE_CXX11)
  #undef ARMA_USE_EXTERN_CXX11_RNG
#endif


#if defined(ARMA_USE_EXTERN_CXX11_RNG)
  extern thread_local arma_rng_cxx11 arma_rng_cxx11_instance;
  // namespace { thread_local arma_rng_cxx11 arma_rng_cxx11_instance; }
#endif


class arma_rng
  {
  public:
  
  #if   defined(ARMA_RNG_ALT)
    typedef arma_rng_alt::seed_type   seed_type;
  #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
    typedef arma_rng_cxx11::seed_type seed_type;
  #else
    typedef arma_rng_cxx98::seed_type seed_type;
  #endif
  
  #if   defined(ARMA_RNG_ALT)
    static const int rng_method = 2;
  #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
    static const int rng_method = 1;
  #else
    static const int rng_method = 0;
  #endif
  
  inline static void set_seed(const seed_type val);
  inline static void set_seed_random();
  
  template<typename eT> struct randi;
  template<typename eT> struct randu;
  template<typename eT> struct randn;
  };



inline
void
arma_rng::set_seed(const arma_rng::seed_type val)
  {
  #if   defined(ARMA_RNG_ALT)
    {
    arma_rng_alt::set_seed(val);
    }
  #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
    {
    arma_rng_cxx11_instance.set_seed(val);
    }
  #else
    {
    arma_rng_cxx98::set_seed(val);
    }
  #endif
  }



inline
void
arma_rng::set_seed_random()
  {
  seed_type seed1 = seed_type(0);
  seed_type seed2 = seed_type(0);
  seed_type seed3 = seed_type(0);
  seed_type seed4 = seed_type(0);
  seed_type seed5 = seed_type(0);
  
  bool rd_ok = false;
  
  #if defined(ARMA_USE_CXX11)
    {
    try
      {
      std::random_device rd;
      
      seed1 = static_cast<seed_type>( rd() );
      
      rd_ok = true;
      }
    catch(...) {}
    }
  #endif
  
  if(rd_ok == false)
    {
    try
      {
      unsigned char buffer = 0;
      
      std::ifstream f("/dev/urandom", std::ifstream::binary);
      
      f.read((char*)(&buffer), 1);
      
      seed2 = static_cast<seed_type>(buffer);
      }
    catch(...) {}
    
    // get better-than-nothing seeds in case reading /dev/urandom failed 
    
    #if defined(ARMA_HAVE_GETTIMEOFDAY)
      {
      struct timeval posix_time;
      
      gettimeofday(&posix_time, 0);
      
      seed3 = static_cast<seed_type>(posix_time.tv_usec);
      }
    #endif
    
    seed4 = static_cast<seed_type>( std::time(NULL) & 0xFFFF );
    
    union
      {
      uword*        a;
      unsigned char b[sizeof(uword*)];
      } address;
    
    uword junk = 0;
    
    address.a = &junk;
    
    seed5 = seed_type(address.b[0]) + seed_type(address.b[sizeof(uword*)-1]);
    }
  
  arma_rng::set_seed( seed1 + seed2 + seed3 + seed4 + seed5 );
  }



template<typename eT>
struct arma_rng::randi
  {
  arma_inline
  operator eT ()
    {
    #if   defined(ARMA_RNG_ALT)
      {
      return eT( arma_rng_alt::randi_val() );
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      return eT( arma_rng_cxx11_instance.randi_val() );
      }
    #else
      {
      return eT( arma_rng_cxx98::randi_val() );
      }
    #endif
    }
  
  
  inline
  static
  int
  max_val()
    {
    #if   defined(ARMA_RNG_ALT)
      {
      return arma_rng_alt::randi_max_val();
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      return arma_rng_cxx11::randi_max_val();
      }
    #else
      {
      return arma_rng_cxx98::randi_max_val();
      }
    #endif
    }
  
  
  inline
  static
  void
  fill(eT* mem, const uword N, const int a, const int b)
    {
    #if   defined(ARMA_RNG_ALT)
      {
      return arma_rng_alt::randi_fill(mem, N, a, b);
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      return arma_rng_cxx11_instance.randi_fill(mem, N, a, b);
      }
    #else
      {
      return arma_rng_cxx98::randi_fill(mem, N, a, b);
      }
    #endif
    }
  };



template<typename eT>
struct arma_rng::randu
  {
  arma_inline
  operator eT ()
    {
    #if   defined(ARMA_RNG_ALT)
      {
      return eT( arma_rng_alt::randu_val() );
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      return eT( arma_rng_cxx11_instance.randu_val() );
      }
    #else
      {
      return eT( arma_rng_cxx98::randu_val() );
      }
    #endif
    }
  
  
  inline
  static
  void
  fill(eT* mem, const uword N)
    {
    uword i,j;
    
    for(i=0, j=1; j < N; i+=2, j+=2)
      {
      const eT tmp_i = eT( arma_rng::randu<eT>() );
      const eT tmp_j = eT( arma_rng::randu<eT>() );
      
      mem[i] = tmp_i;
      mem[j] = tmp_j;
      }
    
    if(i < N)
      {
      mem[i] = eT( arma_rng::randu<eT>() );
      }
    }
  };



template<typename T>
struct arma_rng::randu< std::complex<T> >
  {
  arma_inline
  operator std::complex<T> ()
    {
    const T a = T( arma_rng::randu<T>() );
    const T b = T( arma_rng::randu<T>() );
    
    return std::complex<T>(a, b);
    }
  
  
  inline
  static
  void
  fill(std::complex<T>* mem, const uword N)
    {
    for(uword i=0; i < N; ++i)
      {
      const T a = T( arma_rng::randu<T>() );
      const T b = T( arma_rng::randu<T>() );
      
      mem[i] = std::complex<T>(a, b);
      }
    }
  };



template<typename eT>
struct arma_rng::randn
  {
  inline
  operator eT () const
    {
    #if   defined(ARMA_RNG_ALT)
      {
      return eT( arma_rng_alt::randn_val() );
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      return eT( arma_rng_cxx11_instance.randn_val() );
      }
    #else
      {
      return eT( arma_rng_cxx98::randn_val() );
      }
    #endif
    }
  
  
  arma_inline
  static
  void
  dual_val(eT& out1, eT& out2)
    {
    #if   defined(ARMA_RNG_ALT)
      {
      arma_rng_alt::randn_dual_val(out1, out2);
      }
    #elif defined(ARMA_USE_EXTERN_CXX11_RNG)
      {
      arma_rng_cxx11_instance.randn_dual_val(out1, out2);
      }
    #else
      {
      arma_rng_cxx98::randn_dual_val(out1, out2);
      }
    #endif
    }
  
  
  inline
  static
  void
  fill(eT* mem, const uword N)
    {
    uword i, j;
    
    for(i=0, j=1; j < N; i+=2, j+=2)
      {
      arma_rng::randn<eT>::dual_val( mem[i], mem[j] );
      }
    
    if(i < N)
      {
      mem[i] = eT( arma_rng::randn<eT>() );
      }
    }
  
  };



template<typename T>
struct arma_rng::randn< std::complex<T> >
  {
  inline
  operator std::complex<T> () const
    {
    T a, b;
    
    arma_rng::randn<T>::dual_val(a, b);
    
    return std::complex<T>(a, b);
    }
  
  
  inline
  static
  void
  fill(std::complex<T>* mem, const uword N)
    {
    for(uword i=0; i < N; ++i)
      {
      mem[i] = std::complex<T>( arma_rng::randn< std::complex<T> >() );
      }
    }
  
  };



//! @}

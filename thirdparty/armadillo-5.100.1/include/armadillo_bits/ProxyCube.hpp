// Copyright (C) 2010-2013 Conrad Sanderson
// Copyright (C) 2010-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup ProxyCube
//! @{



template<typename T1>
class ProxyCube
  {
  public:
  inline ProxyCube(const T1&)
    {
    arma_type_check(( is_arma_cube_type<T1>::value == false ));
    }
  };



// ea_type is the "element accessor" type,
// which can provide access to elements via operator[]

template<typename eT>
class ProxyCube< Cube<eT> >
  {
  public:
  
  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Cube<eT>                                 stored_type;
  typedef const eT*                                ea_type;
  typedef const Cube<eT>&                          aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const Cube<eT>& Q;
  
  inline explicit ProxyCube(const Cube<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q.memptr(); }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q;          }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>& X) const { return (void_ptr(&Q) == void_ptr(&X)); }
  
  arma_inline bool is_aligned() const { return memory::is_aligned(Q.memptr()); }
  };



template<typename eT, typename gen_type>
class ProxyCube< GenCube<eT, gen_type > >
  {
  public:
  
  typedef          eT                              elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef GenCube<eT, gen_type>                    stored_type;
  typedef const GenCube<eT, gen_type>&             ea_type;
  typedef const GenCube<eT, gen_type>&             aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const GenCube<eT, gen_type>& Q;
  
  inline explicit ProxyCube(const GenCube<eT, gen_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;                     }
  arma_inline uword get_n_cols()       const { return Q.n_cols;                     }
  arma_inline uword get_n_elem_slice() const { return Q.n_rows*Q.n_cols;            }
  arma_inline uword get_n_slices()     const { return Q.n_slices;                   }
  arma_inline uword get_n_elem()       const { return Q.n_rows*Q.n_cols*Q.n_slices; }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q[i];                  }
  
  arma_inline         ea_type         get_ea() const { return Q; }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q; }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>&) const { return false; }
  
  arma_inline bool is_aligned() const { return GenCube<eT, gen_type>::is_simple; }
  };



template<typename T1, typename op_type>
class ProxyCube< OpCube<T1, op_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Cube<elem_type>                          stored_type;
  typedef const elem_type*                         ea_type;
  typedef const Cube<elem_type>&                   aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const Cube<elem_type> Q;
  
  inline explicit ProxyCube(const OpCube<T1, op_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q.memptr(); }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q;          }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>&) const { return false; }
  
  arma_inline bool is_aligned() const { return memory::is_aligned(Q.memptr()); }
  };



template<typename T1, typename T2, typename glue_type>
class ProxyCube< GlueCube<T1, T2, glue_type> >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Cube<elem_type>                          stored_type;
  typedef const elem_type*                         ea_type;
  typedef const Cube<elem_type>&                   aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const Cube<elem_type> Q;
  
  inline explicit ProxyCube(const GlueCube<T1, T2, glue_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }

  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q.memptr(); }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q;          }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>&) const { return false; }
  
  arma_inline bool is_aligned() const { return memory::is_aligned(Q.memptr()); }
  };



template<typename eT>
class ProxyCube< subview_cube<eT> >
  {
  public:
  
  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_cube<eT>                         stored_type;
  typedef const subview_cube<eT>&                  ea_type;
  typedef const subview_cube<eT>&                  aligned_ea_type;
  
  static const bool prefer_at_accessor = true;
  static const bool has_subview        = true;
  
  arma_aligned const subview_cube<eT>& Q;
  
  inline explicit ProxyCube(const subview_cube<eT>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q; }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q; }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>& X) const { return (void_ptr(&(Q.m)) == void_ptr(&X)); }
  
  arma_inline bool is_aligned() const { return false; }
  };



template<typename T1, typename eop_type>
class ProxyCube< eOpCube<T1, eop_type > >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eOpCube<T1, eop_type>                    stored_type;
  typedef const eOpCube<T1, eop_type>&             ea_type;
  typedef const eOpCube<T1, eop_type>&             aligned_ea_type;
  
  static const bool prefer_at_accessor = eOpCube<T1, eop_type>::prefer_at_accessor;
  static const bool has_subview        = eOpCube<T1, eop_type>::has_subview;
  
  arma_aligned const eOpCube<T1, eop_type>& Q;
  
  inline explicit ProxyCube(const eOpCube<T1, eop_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.get_n_rows();       }
  arma_inline uword get_n_cols()       const { return Q.get_n_cols();       }
  arma_inline uword get_n_elem_slice() const { return Q.get_n_elem_slice(); }
  arma_inline uword get_n_slices()     const { return Q.get_n_slices();     }
  arma_inline uword get_n_elem()       const { return Q.get_n_elem();       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q; }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q; }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>& X) const { return Q.P.is_alias(X); }
  
  arma_inline bool is_aligned() const { return Q.P.is_aligned(); }
  };



template<typename T1, typename T2, typename eglue_type>
class ProxyCube< eGlueCube<T1, T2, eglue_type > >
  {
  public:
  
  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eGlueCube<T1, T2, eglue_type>            stored_type;
  typedef const eGlueCube<T1, T2, eglue_type>&     ea_type;
  typedef const eGlueCube<T1, T2, eglue_type>&     aligned_ea_type;
  
  static const bool prefer_at_accessor = eGlueCube<T1, T2, eglue_type>::prefer_at_accessor;
  static const bool has_subview        = eGlueCube<T1, T2, eglue_type>::has_subview;
  
  arma_aligned const eGlueCube<T1, T2, eglue_type>& Q;
  
  inline explicit ProxyCube(const eGlueCube<T1, T2, eglue_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.get_n_rows();       }
  arma_inline uword get_n_cols()       const { return Q.get_n_cols();       }
  arma_inline uword get_n_elem_slice() const { return Q.get_n_elem_slice(); }
  arma_inline uword get_n_slices()     const { return Q.get_n_slices();     }
  arma_inline uword get_n_elem()       const { return Q.get_n_elem();       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q; }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q; }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>& X) const { return (Q.P1.is_alias(X) || Q.P2.is_alias(X)); }
  
  arma_inline bool is_aligned() const { return Q.P1.is_aligned() && Q.P2.is_aligned(); }
  };



template<typename out_eT, typename T1, typename op_type>
class ProxyCube< mtOpCube<out_eT, T1, op_type> >
  {
  public:
  
  typedef          out_eT                       elem_type;
  typedef typename get_pod_type<out_eT>::result pod_type;
  typedef          Cube<out_eT>                 stored_type;
  typedef          const elem_type*             ea_type;
  typedef          const Cube<out_eT>&          aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const Cube<out_eT> Q;
  
  inline explicit ProxyCube(const mtOpCube<out_eT, T1, op_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q.memptr(); }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q;          }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>&) const { return false; }
  
  arma_inline bool is_aligned() const { return memory::is_aligned(Q.memptr()); }
  };



template<typename out_eT, typename T1, typename T2, typename glue_type>
class ProxyCube< mtGlueCube<out_eT, T1, T2, glue_type > >
  {
  public:
  
  typedef          out_eT                       elem_type;
  typedef typename get_pod_type<out_eT>::result pod_type;
  typedef          Cube<out_eT>                 stored_type;
  typedef          const elem_type*             ea_type;
  typedef          const Cube<out_eT>&          aligned_ea_type;
  
  static const bool prefer_at_accessor = false;
  static const bool has_subview        = false;
  
  arma_aligned const Cube<out_eT> Q;
  
  inline explicit ProxyCube(const mtGlueCube<out_eT, T1, T2, glue_type>& A)
    : Q(A)
    {
    arma_extra_debug_sigprint();
    }
  
  arma_inline uword get_n_rows()       const { return Q.n_rows;       }
  arma_inline uword get_n_cols()       const { return Q.n_cols;       }
  arma_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  arma_inline uword get_n_slices()     const { return Q.n_slices;     }
  arma_inline uword get_n_elem()       const { return Q.n_elem;       }
  
  arma_inline elem_type operator[] (const uword i)                                       const { return Q[i];                  }
  arma_inline elem_type at         (const uword row, const uword col, const uword slice) const { return Q.at(row, col, slice); }
  arma_inline elem_type at_alt     (const uword i)                                       const { return Q.at_alt(i);           }
  
  arma_inline         ea_type         get_ea() const { return Q.memptr(); }
  arma_inline aligned_ea_type get_aligned_ea() const { return Q;          }
  
  template<typename eT2>
  arma_inline bool is_alias(const Cube<eT2>&) const { return false; }
  
  arma_inline bool is_aligned() const { return memory::is_aligned(Q.memptr()); }
  };



//! @}

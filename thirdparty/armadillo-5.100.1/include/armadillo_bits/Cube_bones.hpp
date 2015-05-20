// Copyright (C) 2008-2013 Conrad Sanderson
// Copyright (C) 2008-2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup Cube
//! @{



struct Cube_prealloc
  {
  static const uword mat_ptrs_size = 4;
  static const uword mem_n_elem    = 64;
  };



//! Dense cube class

template<typename eT>
class Cube : public BaseCube< eT, Cube<eT> >
  {
  public:
  
  typedef eT                                elem_type; //!< the type of elements stored in the cube
  typedef typename get_pod_type<eT>::result pod_type;  //!< if eT is non-complex, pod_type is same as eT. otherwise, pod_type is the underlying type used by std::complex
  
  const uword  n_rows;       //!< number of rows in each slice (read-only)
  const uword  n_cols;       //!< number of columns in each slice (read-only)
  const uword  n_elem_slice; //!< number of elements in each slice (read-only)
  const uword  n_slices;     //!< number of slices in the cube (read-only)
  const uword  n_elem;       //!< number of elements in the cube (read-only)
  const uword  mem_state;
  
  // mem_state = 0: normal cube that can be resized; 
  // mem_state = 1: use auxiliary memory until change in the number of elements is requested;  
  // mem_state = 2: use auxiliary memory and don't allow the number of elements to be changed; 
  // mem_state = 3: fixed size (e.g. via template based size specification).
  
  
  arma_aligned const Mat<eT>** const mat_ptrs; //!< pointer to an array containing pointers to Mat instances (one for each slice)
  arma_aligned const eT*       const mem;      //!< pointer to the memory used by the cube (memory is read-only)
  
  protected:
  arma_align_mem Mat<eT>* mat_ptrs_local[ Cube_prealloc::mat_ptrs_size ];
  arma_align_mem eT            mem_local[ Cube_prealloc::mem_n_elem    ];
  
  
  public:
  
  inline ~Cube();
  inline  Cube();
  
  inline Cube(const uword in_rows, const uword in_cols, const uword in_slices);
  
  template<typename fill_type>
  inline Cube(const uword in_rows, const uword in_cols, const uword in_slices, const fill::fill_class<fill_type>& f);
  
  #if defined(ARMA_USE_CXX11)
  inline                  Cube(Cube&& m);
  inline const Cube& operator=(Cube&& m);
  #endif
  
  inline Cube(      eT* aux_mem, const uword aux_n_rows, const uword aux_n_cols, const uword aux_n_slices, const bool copy_aux_mem = true, const bool strict = true);
  inline Cube(const eT* aux_mem, const uword aux_n_rows, const uword aux_n_cols, const uword aux_n_slices);
  
  arma_inline const Cube&  operator=(const eT val);
  arma_inline const Cube& operator+=(const eT val);
  arma_inline const Cube& operator-=(const eT val);
  arma_inline const Cube& operator*=(const eT val);
  arma_inline const Cube& operator/=(const eT val);
  
  inline                   Cube(const Cube& m);
  inline const Cube&  operator=(const Cube& m);
  inline const Cube& operator+=(const Cube& m);
  inline const Cube& operator-=(const Cube& m);
  inline const Cube& operator%=(const Cube& m);
  inline const Cube& operator/=(const Cube& m);
  
  template<typename T1, typename T2>
  inline explicit Cube(const BaseCube<pod_type,T1>& A, const BaseCube<pod_type,T2>& B);
  
  inline                   Cube(const subview_cube<eT>& X);
  inline const Cube&  operator=(const subview_cube<eT>& X);
  inline const Cube& operator+=(const subview_cube<eT>& X);
  inline const Cube& operator-=(const subview_cube<eT>& X);
  inline const Cube& operator%=(const subview_cube<eT>& X);
  inline const Cube& operator/=(const subview_cube<eT>& X);
  
  arma_inline       Mat<eT>& slice(const uword in_slice);
  arma_inline const Mat<eT>& slice(const uword in_slice) const;
  
  arma_inline       subview_cube<eT> slices(const uword in_slice1, const uword in_slice2);
  arma_inline const subview_cube<eT> slices(const uword in_slice1, const uword in_slice2) const;
  
  arma_inline       subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2);
  arma_inline const subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2) const;
  
  inline            subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s);
  inline      const subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const;
  
  inline            subview_cube<eT> subcube(const span& row_span, const span& col_span, const span& slice_span);
  inline      const subview_cube<eT> subcube(const span& row_span, const span& col_span, const span& slice_span) const;
  
  inline            subview_cube<eT> operator()(const span& row_span, const span& col_span, const span& slice_span);
  inline      const subview_cube<eT> operator()(const span& row_span, const span& col_span, const span& slice_span) const;
  
  inline            subview_cube<eT> operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s);
  inline      const subview_cube<eT> operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const;
  
  arma_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1);
  arma_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1) const;
  
  arma_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2);
  arma_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const;
  
  arma_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1, const SizeMat& s);
  arma_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1, const SizeMat& s) const;
  
  inline            subview_cube<eT> tube(const span& row_span, const span& col_span);
  inline      const subview_cube<eT> tube(const span& row_span, const span& col_span) const;
  
  template<typename T1> arma_inline       subview_elem1<eT,T1> elem(const Base<uword,T1>& a);
  template<typename T1> arma_inline const subview_elem1<eT,T1> elem(const Base<uword,T1>& a) const;
  
  template<typename T1> arma_inline       subview_elem1<eT,T1> operator()(const Base<uword,T1>& a);
  template<typename T1> arma_inline const subview_elem1<eT,T1> operator()(const Base<uword,T1>& a) const;
  
  
  inline void shed_slice(const uword slice_num);
  
  inline void shed_slices(const uword in_slice1, const uword in_slice2);
  
  inline void insert_slices(const uword slice_num, const uword N, const bool set_to_zero = true);
  
  template<typename T1>
  inline void insert_slices(const uword row_num, const BaseCube<eT,T1>& X);
  
  
  template<typename gen_type> inline                   Cube(const GenCube<eT, gen_type>& X);
  template<typename gen_type> inline const Cube&  operator=(const GenCube<eT, gen_type>& X);
  template<typename gen_type> inline const Cube& operator+=(const GenCube<eT, gen_type>& X);
  template<typename gen_type> inline const Cube& operator-=(const GenCube<eT, gen_type>& X);
  template<typename gen_type> inline const Cube& operator%=(const GenCube<eT, gen_type>& X);
  template<typename gen_type> inline const Cube& operator/=(const GenCube<eT, gen_type>& X);
  
  template<typename T1, typename op_type> inline                   Cube(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube&  operator=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator+=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator-=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator%=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator/=(const OpCube<T1, op_type>& X);
  
  template<typename T1, typename eop_type> inline                   Cube(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline const Cube&  operator=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline const Cube& operator+=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline const Cube& operator-=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline const Cube& operator%=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline const Cube& operator/=(const eOpCube<T1, eop_type>& X);
  
  template<typename T1, typename op_type> inline                   Cube(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube&  operator=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator+=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator-=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator%=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline const Cube& operator/=(const mtOpCube<eT, T1, op_type>& X);
  
  template<typename T1, typename T2, typename glue_type> inline                   Cube(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube&  operator=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator+=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator-=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator%=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator/=(const GlueCube<T1, T2, glue_type>& X);
  
  template<typename T1, typename T2, typename eglue_type> inline                   Cube(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline const Cube&  operator=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline const Cube& operator+=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline const Cube& operator-=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline const Cube& operator%=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline const Cube& operator/=(const eGlueCube<T1, T2, eglue_type>& X);
  
  template<typename T1, typename T2, typename glue_type> inline                   Cube(const mtGlueCube<eT, T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube&  operator=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator+=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator-=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator%=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline const Cube& operator/=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  
  
  arma_inline arma_warn_unused const eT& at_alt     (const uword i) const;
  
  arma_inline arma_warn_unused       eT& operator[] (const uword i);
  arma_inline arma_warn_unused const eT& operator[] (const uword i) const;
  
  arma_inline arma_warn_unused       eT& at(const uword i);
  arma_inline arma_warn_unused const eT& at(const uword i) const;
  
  arma_inline arma_warn_unused       eT& operator() (const uword i);
  arma_inline arma_warn_unused const eT& operator() (const uword i) const;
  
  arma_inline arma_warn_unused       eT& at         (const uword in_row, const uword in_col, const uword in_slice);
  arma_inline arma_warn_unused const eT& at         (const uword in_row, const uword in_col, const uword in_slice) const;
  
  arma_inline arma_warn_unused       eT& operator() (const uword in_row, const uword in_col, const uword in_slice);
  arma_inline arma_warn_unused const eT& operator() (const uword in_row, const uword in_col, const uword in_slice) const;
  
  arma_inline const Cube& operator++();
  arma_inline void        operator++(int);
  
  arma_inline const Cube& operator--();
  arma_inline void        operator--(int);
  
  arma_inline arma_warn_unused bool is_finite() const;
  arma_inline arma_warn_unused bool is_empty()  const;
  
  arma_inline arma_warn_unused bool in_range(const uword i) const;
  arma_inline arma_warn_unused bool in_range(const span& x) const;
  
  arma_inline arma_warn_unused bool in_range(const uword   in_row, const uword   in_col, const uword   in_slice) const;
       inline arma_warn_unused bool in_range(const span& row_span, const span& col_span, const span& slice_span) const;
  
       inline arma_warn_unused bool in_range(const uword   in_row, const uword   in_col, const uword   in_slice, const SizeCube& s) const;
  
  arma_inline arma_warn_unused       eT* memptr();
  arma_inline arma_warn_unused const eT* memptr() const;
  
  arma_inline arma_warn_unused       eT* slice_memptr(const uword slice);
  arma_inline arma_warn_unused const eT* slice_memptr(const uword slice) const;
  
  arma_inline arma_warn_unused       eT* slice_colptr(const uword in_slice, const uword in_col);
  arma_inline arma_warn_unused const eT* slice_colptr(const uword in_slice, const uword in_col) const;
  
  inline void impl_print(const std::string& extra_text) const;
  inline void impl_print(std::ostream& user_stream, const std::string& extra_text) const;
  
  inline void impl_raw_print(const std::string& extra_text) const;
  inline void impl_raw_print(std::ostream& user_stream, const std::string& extra_text) const;
  
  inline void  set_size(const uword in_rows, const uword in_cols, const uword in_slices);
  inline void   reshape(const uword in_rows, const uword in_cols, const uword in_slices, const uword dim = 0);
  inline void    resize(const uword in_rows, const uword in_cols, const uword in_slices);
  
  template<typename eT2> inline void copy_size(const Cube<eT2>& m);
  
  
  template<typename functor>
  inline const Cube& transform(functor F);
  
  template<typename functor>
  inline const Cube& imbue(functor F);
  
  
  inline const Cube& fill(const eT val);
  
  inline const Cube& zeros();
  inline const Cube& zeros(const uword in_rows, const uword in_cols, const uword in_slices);
  
  inline const Cube& ones();
  inline const Cube& ones(const uword in_rows, const uword in_cols, const uword in_slices);
  
  inline const Cube& randu();
  inline const Cube& randu(const uword in_rows, const uword in_cols, const uword in_slices);
  
  inline const Cube& randn();
  inline const Cube& randn(const uword in_rows, const uword in_cols, const uword in_slices);
  
  inline void reset();
  
  
  template<typename T1> inline void set_real(const BaseCube<pod_type,T1>& X);
  template<typename T1> inline void set_imag(const BaseCube<pod_type,T1>& X);
  
  
  inline arma_warn_unused eT min() const;
  inline arma_warn_unused eT max() const;
  
  inline eT min(uword& index_of_min_val) const;
  inline eT max(uword& index_of_max_val) const;
  
  inline eT min(uword& row_of_min_val, uword& col_of_min_val, uword& slice_of_min_val) const;
  inline eT max(uword& row_of_max_val, uword& col_of_max_val, uword& slice_of_max_val) const;
  
  
  inline bool save(const std::string   name, const file_type type = arma_binary, const bool print_status = true) const;
  inline bool save(      std::ostream& os,   const file_type type = arma_binary, const bool print_status = true) const;
  
  inline bool load(const std::string   name, const file_type type = auto_detect, const bool print_status = true);
  inline bool load(      std::istream& is,   const file_type type = auto_detect, const bool print_status = true);
  
  inline bool quiet_save(const std::string   name, const file_type type = arma_binary) const;
  inline bool quiet_save(      std::ostream& os,   const file_type type = arma_binary) const;
  
  inline bool quiet_load(const std::string   name, const file_type type = auto_detect);
  inline bool quiet_load(      std::istream& is,   const file_type type = auto_detect);
  
  
  // iterators
  
  typedef       eT*       iterator;
  typedef const eT* const_iterator;
  
  typedef       eT*       slice_iterator;
  typedef const eT* const_slice_iterator;
  
  inline       iterator  begin();
  inline const_iterator  begin() const;
  inline const_iterator cbegin() const;
  
  inline       iterator  end();
  inline const_iterator  end() const;
  inline const_iterator cend() const;
  
  inline       slice_iterator begin_slice(const uword slice_num);
  inline const_slice_iterator begin_slice(const uword slice_num) const;
  
  inline       slice_iterator end_slice(const uword slice_num);
  inline const_slice_iterator end_slice(const uword slice_num)   const;
  
  inline void  clear();
  inline bool  empty() const;
  inline uword size()  const;
  
  inline void swap(Cube& B);
  
  inline void steal_mem(Cube& X);  //!< don't use this unless you're writing code internal to Armadillo
  
  template<uword fixed_n_rows, uword fixed_n_cols, uword fixed_n_slices> class fixed;
  
  
  protected:
  
  inline void init_cold();
  inline void init_warm(const uword in_rows, const uword in_cols, const uword in_slices);
  
  template<typename T1, typename T2>
  inline void init(const BaseCube<pod_type,T1>& A, const BaseCube<pod_type,T2>& B);
  
  inline void delete_mat();
  inline void create_mat();
  
  friend class glue_join;
  friend class op_reshape;
  friend class op_resize;
  
  
  public:
  
  #ifdef ARMA_EXTRA_CUBE_PROTO
    #include ARMA_INCFILE_WRAP(ARMA_EXTRA_CUBE_PROTO)
  #endif
  };



template<typename eT>
template<uword fixed_n_rows, uword fixed_n_cols, uword fixed_n_slices>
class Cube<eT>::fixed : public Cube<eT>
  {
  private:
  
  static const uword fixed_n_elem       = fixed_n_rows * fixed_n_cols * fixed_n_slices;
  static const uword fixed_n_elem_slice = fixed_n_rows * fixed_n_cols;
  
  static const bool use_extra = (fixed_n_elem > Cube_prealloc::mem_n_elem);
  
  arma_aligned   Mat<eT>* mat_ptrs_local_extra[ (fixed_n_slices > Cube_prealloc::mat_ptrs_size) ? fixed_n_slices : 1 ];
  arma_align_mem eT       mem_local_extra     [ use_extra                                       ? fixed_n_elem   : 1 ];
  
  arma_inline void mem_setup();
  
  
  public:
  
  inline fixed();
  inline fixed(const fixed<fixed_n_rows, fixed_n_cols, fixed_n_slices>& X);
  
  template<typename fill_type>       inline fixed(const fill::fill_class<fill_type>& f);
  template<typename T1>              inline fixed(const BaseCube<eT,T1>& A);
  template<typename T1, typename T2> inline fixed(const BaseCube<pod_type,T1>& A, const BaseCube<pod_type,T2>& B);
  
  using Cube<eT>::operator=;
  using Cube<eT>::operator();
  
  inline const Cube& operator=(const fixed<fixed_n_rows, fixed_n_cols, fixed_n_slices>& X);
  
  
  arma_inline arma_warn_unused       eT& operator[] (const uword i);
  arma_inline arma_warn_unused const eT& operator[] (const uword i) const;
  
  arma_inline arma_warn_unused       eT& at         (const uword i);
  arma_inline arma_warn_unused const eT& at         (const uword i) const;
  
  arma_inline arma_warn_unused       eT& operator() (const uword i);
  arma_inline arma_warn_unused const eT& operator() (const uword i) const;
  
  arma_inline arma_warn_unused       eT& at         (const uword in_row, const uword in_col, const uword in_slice);
  arma_inline arma_warn_unused const eT& at         (const uword in_row, const uword in_col, const uword in_slice) const;
  
  arma_inline arma_warn_unused       eT& operator() (const uword in_row, const uword in_col, const uword in_slice);
  arma_inline arma_warn_unused const eT& operator() (const uword in_row, const uword in_col, const uword in_slice) const;
  };



class Cube_aux
  {
  public:
  
  template<typename eT> arma_inline static void prefix_pp(Cube<eT>& x);
  template<typename T>  arma_inline static void prefix_pp(Cube< std::complex<T> >& x);
  
  template<typename eT> arma_inline static void postfix_pp(Cube<eT>& x);
  template<typename T>  arma_inline static void postfix_pp(Cube< std::complex<T> >& x);
  
  template<typename eT> arma_inline static void prefix_mm(Cube<eT>& x);
  template<typename T>  arma_inline static void prefix_mm(Cube< std::complex<T> >& x);
  
  template<typename eT> arma_inline static void postfix_mm(Cube<eT>& x);
  template<typename T>  arma_inline static void postfix_mm(Cube< std::complex<T> >& x);
  
  template<typename eT, typename T1> inline static void set_real(Cube<eT>&                out, const BaseCube<eT,T1>& X);
  template<typename eT, typename T1> inline static void set_imag(Cube<eT>&                out, const BaseCube<eT,T1>& X);
  
  template<typename T,  typename T1> inline static void set_real(Cube< std::complex<T> >& out, const BaseCube< T,T1>& X);
  template<typename T,  typename T1> inline static void set_imag(Cube< std::complex<T> >& out, const BaseCube< T,T1>& X);
  };



//! @}

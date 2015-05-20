// Copyright (C) 2012 Conrad Sanderson
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup subview_each
//! @{



template<typename parent, unsigned int mode>
class subview_each_common
  {
  public:
  
  typedef typename parent::elem_type eT;
  
  
  protected:
  
  parent& p;
  
  arma_inline subview_each_common(parent& in_p);
  
  arma_inline const Mat<typename parent::elem_type>& get_mat_ref_helper(const Mat    <typename parent::elem_type>& X) const;
  arma_inline const Mat<typename parent::elem_type>& get_mat_ref_helper(const subview<typename parent::elem_type>& X) const;
  
  arma_inline const Mat<typename parent::elem_type>& get_mat_ref() const;
  
  inline void check_size(const Mat<typename parent::elem_type>& A) const;
  
  arma_cold inline const std::string incompat_size_string(const Mat<typename parent::elem_type>& A) const;
  
  
  private:
  
  subview_each_common();
  };




template<typename parent, unsigned int mode>
class subview_each1 : public subview_each_common<parent, mode>
  {
  protected:
  
  arma_inline subview_each1(parent& in_p);
  
  
  public:
  
  typedef typename parent::elem_type eT;
  
  inline ~subview_each1();
  
  // deliberately returning void
  template<typename T1> inline void operator=  (const Base<eT,T1>& x);
  template<typename T1> inline void operator+= (const Base<eT,T1>& x);
  template<typename T1> inline void operator-= (const Base<eT,T1>& x);
  template<typename T1> inline void operator%= (const Base<eT,T1>& x);
  template<typename T1> inline void operator/= (const Base<eT,T1>& x);
  
  
  private:
  
  friend class Mat<eT>;
  friend class subview<eT>;
  
  subview_each1();
  };



template<typename parent, unsigned int mode, typename TB>
class subview_each2 : public subview_each_common<parent, mode>
  {
  protected:
  
  const Base<uword, TB>& base_indices;
  
  inline subview_each2(parent& in_p, const Base<uword, TB>& in_indices);
  
  inline void check_indices(const Mat<uword>& indices) const;
  
  
  public:
  
  typedef typename parent::elem_type eT;
  
  inline ~subview_each2();
  
  // deliberately returning void
  template<typename T1> inline void operator=  (const Base<eT,T1>& x);
  template<typename T1> inline void operator+= (const Base<eT,T1>& x);
  template<typename T1> inline void operator-= (const Base<eT,T1>& x);
  template<typename T1> inline void operator%= (const Base<eT,T1>& x);
  template<typename T1> inline void operator/= (const Base<eT,T1>& x);
  
  // TODO: add handling of scalars
  
  
  private:
  
  friend class Mat<eT>;
  friend class subview<eT>;
  
  subview_each2();
  };



//! @}

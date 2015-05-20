// Copyright (C) 2013 Conrad Sanderson
// Copyright (C) 2013 Alexandre Drouin 
// Copyright (C) 2013 NICTA (www.nicta.com.au)
// 
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//! \addtogroup fn_inplace_strans
//! @{



template<typename eT>
inline
void
inplace_strans
  (
        Mat<eT>& X,
  const char*    method = "std"
  )
  {
  arma_extra_debug_sigprint();
  
  const char sig = (method != NULL) ? method[0] : char(0);
  
  arma_debug_check( ((sig != 's') && (sig != 'l')), "inplace_strans(): unknown method specified" );
  
  const bool low_memory = (sig == 'l');
  
  if( (low_memory == false) || (X.n_rows == X.n_cols) )
    {
    op_strans::apply_mat_inplace(X);
    }
  else
    {
    // in-place algorithm inspired by:
    // Fred G. Gustavson, Tadeusz Swirszcz.
    // In-Place Transposition of Rectangular Matrices.
    // Applied Parallel Computing. State of the Art in Scientific Computing.
    // Lecture Notes in Computer Science. Volume 4699, pp. 560-569, 2007.
    
    
    // X.set_size() will check whether we can change the dimensions of X;
    // X.set_size() will also reuse existing memory, as the number of elements hasn't changed
    
    X.set_size(X.n_cols, X.n_rows);
    
    const uword m = X.n_cols;
    const uword n = X.n_rows;
    
    std::vector<bool> visited(X.n_elem);  // TODO: replace std::vector<bool> with a better implementation
    
    for(uword col = 0; col < m; ++col)
    for(uword row = 0; row < n; ++row)
      {
      const uword pos = col*n + row;
      
      if(visited[pos] == false)
        {
        uword curr_pos = pos;
          
        eT val = X.at(row, col);
        
        while(visited[curr_pos] == false) 
          {
          visited[curr_pos] = true;
            
          const uword j = curr_pos / m;
          const uword i = curr_pos - m * j;
          
          const eT tmp = X.at(j, i);
          X.at(j, i) = val;
          val = tmp;
            
          curr_pos = i*n + j;
          }
        }
      }
    }
  }



//! @}

#ifndef __MAT_H__
#define __MAT_H__

#include <iostream>
#include <vector>

namespace ltp {
namespace math {

/**
 * A class for vector
 */
template <typename T>
class Vec {
private:
  size_t nn;
  T * v;
public:
  Vec() : nn(0), v(0) {}

  ~Vec() {
    dealloc();
  }

  // zero-based array
  explicit Vec(const int n) : nn(0), v(0) {
    resize(n);
  }

  // initialize to constant value
  Vec(const T &a, const int n) : nn(n), v(new T[n]) {
    for (size_t i = 0; i < n; ++ i) {
      v[i] = a;
    }
  }

  // initialize to array
  Vec(const T *a, const int n) : nn(n), v(new T[n]) {
    for (size_t i = 0; i < n; ++ i) {
      v[i] = *a;
      a ++;
    }
  }

  // copy constructor
  Vec(const Vec<T> &rhs): nn(rhs.nn), v(new T[nn]) {
    for (size_t i = 0; i < nn; ++ i) {
      v[i] = rhs[i];
    }
  }

  Vec& resize(const int n) {
    if (nn != n) {
      if (v != 0) {
        delete [] (v);
      }
      nn = n;
      v = new T[n];
    }
    return *this;
  }

  Vec& operator=(const Vec& rhs) {
    if (this != &rhs) {
      if (nn != rhs.nn) {
        if (v != 0) {
          delete [] (v);
        }
        nn = rhs.nn;
        v = new T[nn];
      }
      for (size_t i = 0; i < nn; ++ i) v[i] = rhs[i];
    }
    return *this;
  }

  Vec& operator=(const T& a) {
    for (size_t i = 0; i < nn; ++ i) {
      v[i] = a;
    }
    return *this;
  }

  Vec& operator=(const std::vector<T>& a) {
    if (nn != a.size()) {
      if (v != 0) {
        delete [] (v);
      }
      nn = a.size();
      v = new T[nn];
    }

    for (size_t i = 0; i < nn; ++ i) v[i] = a[i];
    return *this;
  }

  inline T& operator [](const int i) {
    return v[i];
  }

  inline const T& operator [](const int i) const {
    return v[i];
  }

  inline int size() const {
    return nn;
  }

  inline void dealloc() {
    if (v != 0) {
      delete [] (v);
      v = 0;
      nn = 0;
    }
  }

  inline T* c_buf() {
    return v;
  }
};  //  end for class Vec

/**
 * A class for matrix(2d)
 */
template <typename T>
class Mat {
private:
  size_t nn;
  size_t mm;
  size_t tot_sz;
  T ** v;
public:
  ~Mat() {
    dealloc();
  }

  void dealloc() {
    if (v != 0) {
      delete [] (v[0]);
      delete [] (v);
      v = 0;
      nn = 0;
      mm = 0;
      tot_sz = 0;
    }
  }

  T* c_buf() {
    if (v) {
      return v[0];
    } else {
      return 0;
    }
  }

  Mat(): nn(0), mm(0), tot_sz(0), v(0) {}

  Mat& resize(const size_t& n, const size_t& m) {
    if (nn != n || mm != m) {
      dealloc();
      nn = n;
      mm = m;
      tot_sz = n * m;

      v = new T*[n];
      v[0] = new T[tot_sz];

      for (size_t i = 1; i < n; ++ i) {
        v[i] = v[i - 1] + m;
      }
    }
    return *this;
  }

  Mat(const size_t& n, const size_t& m): nn(0), mm(0), tot_sz(0), v(0) {
    resize(n, m);
  }

  Mat(const T& a, const size_t& n, const size_t& m): nn(0), mm(0), tot_sz(0), v(0) {
    resize(n, m);
    for (size_t i = 0; i < n; ++ i) {
      for (size_t j = 0; j < m; ++ j) {
        v[i][j] =a;
      }
    }
  }

  Mat(const T* a, const size_t n, const size_t m): nn(0), mm(0), tot_sz(0), v(0) {
    resize(n, m);
    for (size_t i = 0; i < n; ++ i) {
      for (size_t j = 0; j < m; ++ j) {
        v[i][j] = *a;
        ++ a;
      }
    }
  }

  Mat(const Mat& rhs) {
    resize(rhs.nn, rhs.mm);
    for (size_t i = 0; i < nn; ++ i) {
      for (size_t j = 0; j < mm; ++ j) {
        v[i][j] = rhs[i][j];
      }
    }
  }

  Mat& operator= (const Mat& rhs) {
    if (this != &rhs) {
      resize(rhs.nn, rhs.mm);

      for (size_t i = 0; i < nn; ++ i) {
        for (size_t j = 0; j < mm; ++ j) {
          v[i][j] = rhs[i][j];
        }
      }
    }
    return *this;
  }

  Mat& operator= (const T & a) {
    for (size_t i = 0; i < nn; ++ i) {
      for (size_t j = 0; j < mm; ++ j) {
        v[i][j] = a;
      }
    }
    return *this;
  }

  inline T* operator[] (const int i) {
    return v[i];
  }

  inline const T* operator[] (const int i) const {
    return v[i];
  }

  inline size_t nrows() const {
    return nn;
  }

  inline size_t ncols() const {
    return mm;
  }

  inline size_t total_size() const {
    return tot_sz;
  }
};  //  end for class Mat

/*
 *
 *
 */
template <typename T>
class Mat3 {
private:
  size_t nn;
  size_t mm;
  size_t kk;
  size_t tot_sz;

  T ***v;
public:
  Mat3() : nn(0), mm(0), kk(0), tot_sz(0), v(0) {}

  ~Mat3() {
    dealloc();
  }

  void dealloc() {
    if (v != 0) {
      delete [] (v[0][0]);
      delete [] (v[0]);
      delete [] (v);

      v = 0;
      nn = 0;
      mm = 0;
      kk = 0;
      tot_sz = 0;
    }
  }

  T* c_buf() {
    if (v) {
      return v[0][0];
    } else {
      return NULL;
    }
  }

  Mat3(const size_t& n, const size_t& m, const size_t& k)
    : nn(0), mm(0), kk(0), tot_sz(0), v(0) {
    resize(n, m, k);
  }

  Mat3& resize(const size_t& n, const size_t& m, const size_t& k) {
    if (nn != n || mm != m || kk != k) {
      dealloc();

      nn = n;
      mm = m;
      kk = k;
      tot_sz = n * m * k;

      v = new T**[n];
      v[0] = new T*[n * m];
      v[0][0] = new T[tot_sz];

      for (size_t j = 1; j < m; ++ j) {
        v[0][j] = v[0][j - 1] + k;
      }

      for (size_t i = 1; i < n; ++ i) {
        v[i] = v[i - 1] + m;
        v[i][0] = v[i - 1][0] + m * k;

        for (size_t j = 1; j < m; ++ j) {
          v[i][j] = v[i][j - 1] + k;
        }
      }
    }
    return *this;
  }

  Mat3& operator= (const T &a) {
    for (int i = 0; i < nn; ++ i) {
      for (int j = 0; j < mm; ++ j) {
        for (int k = 0; k < kk; ++ k) {
          v[i][j][k] = a;
        }
      }
    }

    return *this;
  }

  inline T** operator[] (const size_t& i) {
    return v[i];
  }

  inline const T * const * operator[] (const size_t& i) const {
    return v[i];
  }

  inline int dim1() const {
    return nn;
  }

  inline int dim2() const {
    return mm;
  }

  inline int dim3() const {
    return kk;
  }

  inline int total_size() const {
    return tot_sz;
  }
};  //  end for class Mat3

template <typename T>
class Mat4 {
private:
  size_t nn;
  size_t mm;
  size_t kk;
  size_t ll;
  size_t tot_sz;
  T ****v;

public:
  Mat4(): nn(0), mm(0), kk(0), ll(0), tot_sz(0), v(0) {}

  ~Mat4() {
    dealloc();
  }

  Mat4& resize(const size_t& n, const size_t& m, const size_t& k, const size_t& l) {
    if (n != nn || m != mm || k != kk || l != ll) {
      dealloc();

      nn = n;
      mm = m;
      kk = k;
      ll = l;
      tot_sz = n * m * k * l;

      v = new T***[n];
      v[0] = new T**[n * m];
      v[0][0] = new T*[n * m * k];
      v[0][0][0] = new T[n * m * k * l];

      for (size_t z = 1; z < k; ++ z) {
        v[0][0][z] = v[0][0][z - 1] + l;
      }

      for (size_t j = 1; j < m; ++ j) {
        v[0][j] = v[0][j - 1] + k;
        v[0][j][0] = v[0][j - 1][0] + k * l;

        for (size_t z = 1; z < k; ++ z) {
          v[0][j][z] = v[0][j][z - 1] + l;
        }
      }

      for (size_t i = 1; i < n; ++ i) {
        v[i] = v[i - 1] + m;
        v[i][0] = v[i - 1][0] + m * k;
        v[i][0][0] = v[i - 1][0][0] + m * k * l;

        for (size_t z = 1; z < k; ++ z) {
          v[i][0][z] = v[i][0][z - 1] + l;
        }

        for (size_t j = 1; j < m; ++ j) {
          v[i][j] = v[i][j - 1] + k;
          v[i][j][0] = v[i][j - 1][0] + k * l;

          for (size_t z = 1; z < k; ++ z) {
            v[i][j][z] = v[i][j][z - 1] + l;
          }
        }
      }
    }
    return (*this);
  }

  explicit Mat4(const size_t& n, const size_t& m, const size_t& k, const size_t& l) {
    resize(n, m, k, l);
  }

  Mat4& operator=(const T& a) {
    for (size_t i = 0; i < nn; ++ i) {
      for (size_t j = 0; j < mm; ++ j) {
        for (size_t k = 0; k < kk; ++ k) {
          for (size_t l = 0; l < ll; ++ l) {
            v[i][j][k][l] = a;
          }
        }
      }
    }
    return *this;
  }

  inline T*** operator[](const size_t& i) {
    return v[i];
  }

  inline const T* const * const * operator[](const size_t& i) const {
    return v[i];
  }

  inline size_t dim1() const {
    return nn;
  }

  inline size_t dim2() const {
    return mm;
  }

  inline size_t dim3() const {
    return kk;
  }

  inline size_t dim4() const {
    return ll;
  }

  inline size_t total_size() const {
    return tot_sz;
  }

  inline void dealloc() {
    if (v) {
      delete [] (v[0][0][0]);
      delete [] (v[0][0]);
      delete [] (v[0]);
      delete [] (v);
      v = 0;
      nn = 0;
      mm = 0;
      kk = 0;
      ll = 0;
      tot_sz = 0;
    }
  }

  inline T* c_buf() {
    if (v) {
      return v[0][0][0];
    } else {
      return 0;
    }
  }

private:
  Mat4(const Mat4& rhs) {}
  Mat4& operator =(const Mat4 & rhs) {}
};

}   //  end for namespace math
}   //  end for namespace ltp

#endif  // end for __MAT_H__

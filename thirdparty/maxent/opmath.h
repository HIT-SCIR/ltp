
/*
 * namespace `vectype` defines the structure of mathematic vector
 *  and necessary vector operations
 *
 * namespace `minifunc` defines several mini functions that is
 *  commonly used in optimization
 *
 */


#ifndef _MATH_VECTOR_H_
#define _MATH_VECTOR_H_

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

namespace math {

namespace vectype {

class Vec
{
    private:
        std::vector<double> _v;

    public:
        Vec(const size_t n = 0, const double val = 0)
        {
            _v.resize(n, val);
        }

        Vec(const std::vector<double> & v) : _v(v) {}

        const std::vector<double> & stl_vec() const { return _v; }
        std::vector<double>       & stl_vec()       { return _v; }

        size_t size() const { return _v.size(); }

        double       & operator[](int i)       { return _v[i]; }
        const double & operator[](int i) const { return _v[i]; }

        Vec & operator+=(const Vec & b)
        {
            assert(b.size() == _v.size());
            for (size_t i = 0; i < _v.size(); i++)
            {
                _v[i] += b[i];
            }
            return *this;
        }

        Vec & operator*=(const double c)
        {
            for (size_t i = 0; i < _v.size(); i++)
            {
                _v[i] *= c;
            }
            return *this;
        }

        /*
         * project the vector to the orthant defined by y
         */
        void project(const Vec & y)
        {
            for (size_t i = 0; i < _v.size(); i++)
            {
                if (_v[i] * y[i] <=0) _v[i] = 0;
            }
        }
};

inline double dot_product(const Vec & a, const Vec & b)
{
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

inline std::ostream & operator<<(std::ostream & s, const Vec & a)
{
    s << "(";
    for (size_t i = 0; i < a.size(); i++)
    {
        if (i != 0) s << ", ";
        s << a[i];
    }
    s << ")";
    return s;
}

inline const Vec operator+(const Vec & a, const Vec & b)
{
    Vec v(a.size());
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++)
    {
        v[i] = a[i] + b[i];
    }
    return v;
}

inline const Vec operator-(const Vec & a, const Vec & b)
{
    Vec v(a.size());
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++)
    {
        v[i] = a[i] - b[i];
    }
    return v;
}

inline const Vec operator*(const Vec & a, const double c)
{
    Vec v(a.size());
    for (size_t i = 0; i < a.size(); i++)
    {
        v[i] = a[i] * c;
    }
    return v;
}

inline const Vec operator*(const double c, const Vec & a)
{
    return a * c;
}

} // end namespace vectype

namespace minifunc {

inline double l1norm(const std::vector<double>& v)
{
    double norm = 0;
    for (size_t i = 0; i < v.size(); ++i)
    {
        norm += fabs(v[i]);
    }

    return norm;
}

inline int sign(double x)
{
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

} // end namespace minifunc

} // end namespace math

#endif

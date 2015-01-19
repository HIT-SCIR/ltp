// Portable STL hashmap include file.
#ifndef __UTILS_UNORDERED_MAP_HPP__
#define __UTILS_UNORDERED_MAP_HPP__
// Portable header for std::unordered_map<K,V> template. Welcome to C++. Enjoy!
// - rlyeh / BOOST licensed

/*
 * Just in case somebody else defined `unordered_map` before us.
 */

#ifdef unordered_map
#undef unordered_map
#endif

/* Headers (in order)
 * - std >= C++11: GCC <4.7.X defines __cplusplus as 1, use __GXX_EXPERIMENTAL_CXX0X__ instead
 * - ICC
 * - G++ >= 4.3.X
 * - G++ >= 3.X.X
 * - MSVC++ >= 9.0
 * - OTHERS
 */

#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#include <unordered_map>
#elif defined(__INTEL_COMPILER)
#include <ext/hash_map>
#elif defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 3)
#include <tr1/unordered_map>
#elif defined(__GNUC__) && __GNUC__ >= 3
#include <ext/hash_map>
#elif defined(_MSC_VER) && ( ( _MSC_VER >= 1500 && _HAS_TR1 ) || ( _MSC_VER >= 1600 ) )
#include <unordered_map>
#else
#include <hash_map>
#endif

/* Namespace and type (in order)
 * - C++11, C++0X (std::unordered_map)
 * - STLPORT (std::hash_map)
 * - MSVC++ 2010 (std::unordered_map)
 * - MSVC++ 9.0 (std::tr1::unordered_map)
 * - MSVC++ 7.0 (stdext::hash_map)
 * - G++ 4.3.X (std::tr1::unordered_map)
 * - G++ 3.X.X, ICC (__gnu_cxx::hash_map)
 * - OTHERS (std::hash_map)
 */

#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
// ok

#elif defined(_STLPORT_VERSION)
#define unordered_map hash_map
namespace std { using std::hash_map; }

#elif defined(_MSC_VER) && _MSC_VER >= 1600
// ok

#elif defined(_MSC_VER) && _MSC_VER >= 1500 && _HAS_TR1
namespace std { using std::tr1::unordered_map; }

#elif defined(_MSC_VER) && _MSC_VER >= 1300
#define unordered_map hash_map
namespace std { using stdext::hash_map; }

#elif defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 3)
namespace std { using std::tr1::unordered_map; }

#elif (defined(__GNUC__) && __GNUC__ >= 3) || defined(__INTEL_COMPILER)
#include <string>
#define unordered_map hash_map
namespace std { using __gnu_cxx::hash_map; }

    namespace __gnu_cxx {
        template<> struct hash<unsigned long long> {
            size_t operator()(const unsigned long long &__x) const {
                return (size_t)__x;
            }
        };
        template<typename T> struct hash<T *> {
            size_t operator()(T * const &__x) const {
                return (size_t)__x;
            }
        };
        template<> struct hash<std::string> {
            size_t operator()(const std::string &__x) const {
                return hash<const char *>()(__x.c_str());
            }
        };
    };

#else
#define unordered_map hash_map
namespace std { using std::hash_map; }

#endif

/*
#if defined(_MSC_VER)
  #include <hash_map>
#else
  #include <tr1/unordered_map>
#endif
*/
#endif  //  end for __UTILS_UNORDERED_MAP_HPP__

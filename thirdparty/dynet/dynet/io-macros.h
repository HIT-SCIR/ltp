#ifndef DYNET_IO_MACROS__
#define DYNET_IO_MACROS__

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#if BOOST_VERSION >= 105600
#include <boost/serialization/unordered_map.hpp>
#endif

#ifndef __CUDACC__
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#define MAX_SERIALIZE_VERSION 1024

#define DYNET_SERIALIZE_IMPL(MyClass) \
  template void MyClass::serialize<boost::archive::text_oarchive>(boost::archive::text_oarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::text_iarchive>(boost::archive::text_iarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::binary_oarchive>(boost::archive::binary_oarchive &ar, const unsigned int); \
  template void MyClass::serialize<boost::archive::binary_iarchive>(boost::archive::binary_iarchive &ar, const unsigned int);

#define DYNET_SAVELOAD_IMPL(MyClass) \
  template void MyClass::save<boost::archive::text_oarchive>(boost::archive::text_oarchive &ar, const unsigned int) const; \
  template void MyClass::load<boost::archive::text_iarchive>(boost::archive::text_iarchive &ar, const unsigned int); \
  template void MyClass::save<boost::archive::binary_oarchive>(boost::archive::binary_oarchive &ar, const unsigned int) const; \
  template void MyClass::load<boost::archive::binary_iarchive>(boost::archive::binary_iarchive &ar, const unsigned int);

#ifdef _MSC_VER
// for BOOST_PP_REPEAT usage, wrap the parameters with PP_NARG
#define DYNET_PP_FOREACH_ARRAY( ... )  ( BOOST_PP_VARIADIC_SIZE(__VA_ARGS__) , ( __VA_ARGS__ ) )

// apply A to all following parameters
#define DYNET_PP_FOREACH( A, ... )  BOOST_PP_REPEAT(BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), A, DYNET_PP_FOREACH_ARRAY(__VA_ARGS__) )
#else
#define DYNET_PP_NARG_(x64, x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53, x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42, x41, x40, x39, x38, x37, x36, x35, x34, x33, x32, x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, n, ...) n
// currently only support max 64 number of parameters
#define DYNET_PP_NARG(...) DYNET_PP_NARG_(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

// for BOOST_PP_REPEAT usage, wrap the parameters with PP_NARG
#define DYNET_PP_FOREACH_ARRAY( ... )  ( DYNET_PP_NARG(__VA_ARGS__) , ( __VA_ARGS__ ) )

// apply A to all following parameters
#define DYNET_PP_FOREACH( A, ... )  BOOST_PP_REPEAT(DYNET_PP_NARG(__VA_ARGS__), A, DYNET_PP_FOREACH_ARRAY(__VA_ARGS__) )
#endif

#define DYNET_ARCHIVE(z, n, data) \
  ar & BOOST_PP_ARRAY_ELEM(n, data);

#define DYNET_UNFOLD(z, n, data) \
  BOOST_PP_ARRAY_ELEM(n, data);

#define DYNET_FUNCTOR(z, n, data) \
  data;

// declare INTERFACE 
#define DYNET_SERIALIZE_DECLARE() \
  friend class boost::serialization::access; \
  template <class Archive> \
  void serialize(Archive &ar, const unsigned int);

// split declare INTERFACE
#define DYNET_SERIALIZE_SPLIT_DECLARE() \
  friend class boost::serialization::access; \
  template <class Archive> \
  void save(Archive & ar, const unsigned int) const; \
  template <class Archive> \
  void load(Archive & ar, const unsigned int); \
  BOOST_SERIALIZATION_SPLIT_MEMBER()

// INTERFACE: empty serialization definition macro
#define DYNET_SERIALIZE_COMMIT_EMPTY(...) \
  friend class boost::serialization::access; \
  template <class Archive> \
  void serialize(Archive & ar, const unsigned int version) {}

// INTERFACE: commit serialization operation macro
#define DYNET_SERIALIZE_COMMIT(MyClass, ...) \
  template <class Archive> \
  void MyClass::serialize(Archive & ar, const unsigned int version) { \
    DYNET_PP_FOREACH(DYNET_UNFOLD, __VA_ARGS__) \
  }

// INTERFACE: commit serialization save operation macro
#define DYNET_SERIALIZE_SAVE_COMMIT(MyClass, ...) \
  template <class Archive> \
  void MyClass::save(Archive & ar, const unsigned int version) const { \
    DYNET_PP_FOREACH(DYNET_UNFOLD, __VA_ARGS__) \
  }

// INTERFACE: commit serialization load operation macro
#define DYNET_SERIALIZE_LOAD_COMMIT(MyClass, FUNC, ...) \
  template <class Archive> \
  void MyClass::load(Archive & ar, const unsigned int version) { \
    DYNET_PP_FOREACH(DYNET_UNFOLD, __VA_ARGS__) \
    FUNC; \
  }

// INTERFACE: specify serialize version macro
#define DYNET_VERSION_DEFINE(T, VERSION) BOOST_CLASS_VERSION(T, VERSION)

// INTERFACE: serialize definition macro
#define DYNET_SERIALIZE_DEFINE(...) \
  DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)

// INTERFACE: serialize definition macro for derived class
#define DYNET_SERIALIZE_DERIVED_DEFINE(T, ...) \
  ar & boost::serialization::base_object<T>(*this); \
  DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)

// INTERFACE: serialize definition macro for derived class which is equal to base class
#define DYNET_SERIALIZE_DERIVED_EQ_DEFINE(T) \
  ar & boost::serialization::base_object<T>(*this);

#ifdef _MSC_VER

#define DYNET_VERSION_SERIALIZE_DEFINE(l, r, ...) \
  if (l >= 0 && r >= 0 && l < r && version >= l && version < r) { \
    DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)                  \
  }

#define DYNET_VERSION_SERIALIZE_DERIVED_DEFINE(T, l, r, ...)  \
  if (l >= 0 && r >= 0 && l < r && version >= l && version < r) { \
    ar & boost::serialization::base_object<T>(*this);             \
    DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)                  \
  } })

#else
// INTERFACE: serialize definition with version macro, l <= version < r
#define DYNET_VERSION_SERIALIZE_DEFINE(l, r, ...) \
  !(l >= 0) ? (void)0 :                           \
  !(r >= 0) ? (void)0 :                           \
  !(l < r) ? (void)0 :                            \
  ({ if (version >= l && version < r) {           \
    DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)  \
  } })

// INTERFACE: serialize definition with version macro for derived class, l <= version < r
#define DYNET_VERSION_SERIALIZE_DERIVED_DEFINE(T, l, r, ...)  \
  !(l >= 0) ? (void)0 :                                       \
  !(r >= 0) ? (void)0 :                                       \
  !(l < r) ? (void)0 :                                        \
  ({ if (version >= l && version < r) {                       \
    ar & boost::serialization::base_object<T>(*this);         \
    DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)              \
  } })
#endif

// INTERFACE: serialize definition macro for non-intrusive impl
#define DYNET_NINTRUSIVE_SERIALIZE_DEFINE(param, ...)       \
namespace boost {                                           \
namespace serialization {                                   \
  template <class Archive>                                  \
  void serialize(Archive & ar, param, const unsigned int) { \
    DYNET_PP_FOREACH(DYNET_ARCHIVE, __VA_ARGS__)            \
  }                                                         \
} /* namespace serialization */ } /* namespace boost */

#endif

INCLUDE_DIRECTORIES( ./
    ${THIRDPARTY_DIR}/boost/include/ )

set (boost_regex_VERSION "1.61")

set (boost_regex_SRC
  src/c_regex_traits.cpp
  src/cpp_regex_traits.cpp
  src/cregex.cpp
  src/fileiter.cpp
  src/icu.cpp
  src/instances.cpp
  src/internals.hpp
  src/posix_api.cpp
  src/regex.cpp
  src/regex_debug.cpp
  src/regex_raw_buffer.cpp
  src/regex_traits_defaults.cpp
  src/static_mutex.cpp
  src/usinstances.cpp
  src/w32_regex_traits.cpp
  src/wc_regex_traits.cpp
  src/wide_posix_api.cpp
  src/winstances.cpp
  )

add_library(boost_regex_static_lib STATIC ${boost_regex_SRC})

set_target_properties (boost_regex_static_lib   PROPERTIES
    OUTPUT_NAME boost_regex)

add_library(boost_regex_shared_lib SHARED ${boost_regex_SRC})

set_target_properties (boost_regex_shared_lib   PROPERTIES
    VERSION     ${boost_regex_VERSION}
    OUTPUT_NAME boost_regex)

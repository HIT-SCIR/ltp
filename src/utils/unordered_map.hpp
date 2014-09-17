// Portable STL hashmap include file.
#ifndef __UTILS_UNORDERED_MAP_HPP__
#define __UTILS_UNORDERED_MAP_HPP__

#ifdef _WIN32
  #include <hash_map>
#else
  #include <tr1/unordered_map>
#endif

#endif  //  end for __UTILS_UNORDERED_MAP_HPP__

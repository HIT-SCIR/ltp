#ifndef __LTP_STRUTILS_CHARTYPES_HPP__
#define __LTP_STRUTILS_CHARTYPES_HPP__

#include "chartypes.tab"
#include <string.h>

#ifdef _WIN32
#include <hash_map>
#else
#include <tr1/unordered_map>
#endif

namespace ltp {
namespace strutils {
namespace chartypes {

enum{ 
    // level 1
    CHAR_LETTER = 1,
    CHAR_DIGIT = 2,
    CHAR_PUNC = 3,
    CHAR_OTHER = -1, 
    // level 2
    CHAR_LETTER_SBC = 11, 
    CHAR_LETTER_DBC = 12,
    CHAR_DIGIT_SBC = 21,
    CHAR_DIGIT_DBC = 22, 
    CHAR_PUNC_SBC = 31,
    CHAR_PUNC_DBC = 32,
    // level 3
    CHAR_LETTER_SBC_UPPERCASE = 111,
    CHAR_LETTER_SBC_LOWERCASE = 112,
    CHAR_LETTER_DBC_UPPERCASE = 121,
    CHAR_LETTER_DBC_LOWERCASE = 122,
    CHAR_DIGIT_DBC_CL1 = 221,
    CHAR_DIGIT_DBC_CL2 = 222,
    CHAR_DIGIT_DBC_CL3 = 223,
    CHAR_PUNC_DBC_NORMAL = 321,
    CHAR_PUNC_DBC_CHINESE = 322,
    CHAR_PUNC_DBC_EXT = 323,
};


struct __chartype_char_array_equal_function {
    bool operator() (const char * s1, const char * s2) const {
        return (strcmp(s1, s2) == 0);
    }
};

struct __chartype_char_array_hash_function 
#ifdef _WIN32
    : public stdext::hash_compare<const char *>
#endif
{
    size_t operator() (const char * s) const {
        unsigned int hashTemp = 0;
        while (*s) {
            hashTemp = hashTemp * 101 + *s ++;
        }
        return (size_t(hashTemp));
    }

    bool operator() (const char * s1, const char * s2) const {
        return (strcmp(s1, s2) < 0);
    }
};

// chartype dictionary
// it's a singleton of key-value structure
template<typename T>
class __chartype_collections {
public:
    static __chartype_collections * get_collections() {
        if (0 == instance_) {
            instance_ = new __chartype_collections;
        }

        return instance_;
    }

    int chartype(const char * key) {
        internal_collection_t::const_iterator itx = collections.find(key);
        if (itx != collections.end()) {
            return itx->second;
        }
        return -1;
    }

protected:
    __chartype_collections() {
        const char * buff = 0;
        buff = __chartype_dbc_chinese_punc_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_chinese_punc_utf8_size__; ++ i) {
            collections[buff] = CHAR_PUNC;
            do {++ buff; } while (*(buff - 1));
        }

        buff = __chartype_dbc_digit_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_digit_utf8_size__; ++ i) {
            collections[buff] = CHAR_DIGIT;
            do {++ buff; } while (*(buff - 1));
        }

        buff = __chartype_dbc_punc_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_punc_utf8_size__; ++ i) {
            collections[buff] = CHAR_PUNC;
            do {++ buff; } while (*(buff - 1));
        }

        buff = __chartype_dbc_uppercase_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_uppercase_utf8_size__; ++ i) {
            collections[buff] = CHAR_LETTER;
            do {++ buff; } while (*(buff - 1));
        }

        buff = __chartype_dbc_punc_ext_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_punc_ext_utf8_size__; ++ i) {
            collections[buff] = CHAR_PUNC;
            do { ++ buff; } while (*(buff - 1));
        }

        buff = __chartype_dbc_lowercase_utf8_buff__;
        for (int i = 0; i < __chartype_dbc_lowercase_utf8_size__; ++ i) {
            collections[buff] = CHAR_LETTER;
            do { ++ buff; } while (*(buff - 1));
        }

        buff = __chartype_sbc_uppercase_utf8_buff__;
        for (int i = 0; i < __chartype_sbc_uppercase_utf8_size__; ++ i) {
            collections[buff] = CHAR_LETTER;
            do { ++ buff; } while (*(buff - 1));
        }

        buff = __chartype_sbc_digit_utf8_buff__;
        for (int i = 0; i < __chartype_sbc_digit_utf8_size__; ++ i) {
            collections[buff] = CHAR_DIGIT;
            do { ++ buff; } while (*(buff - 1));
        }

        buff = __chartype_sbc_punc_utf8_buff__;
        for (int i = 0; i < __chartype_sbc_punc_utf8_size__; ++ i) {
            collections[buff] = CHAR_PUNC;
            do { ++ buff; } while (*(buff - 1));
        }

        buff = __chartype_sbc_lowercase_utf8_buff__;
        for (int i = 0; i < __chartype_sbc_lowercase_utf8_size__; ++ i) {
            collections[buff] = CHAR_LETTER;
            do { ++ buff; } while (*(buff - 1));
        }
    }

    
private:
#ifdef _WIN32
    typedef stdext::hash_map<const char *, int, 
            __chartype_char_array_hash_function> internal_collection_t;
#else
    typedef std::tr1::unordered_map<const char *, int,
            __chartype_char_array_hash_function,
            __chartype_char_array_equal_function> internal_collection_t;
#endif // end for _WIN32

    static __chartype_collections * instance_;
    internal_collection_t  collections;
};

template<typename T> __chartype_collections<T> * __chartype_collections<T>::instance_ = 0;
//template<typename T> internal_collection_t __chartype_collections<T>::collections;

int chartype(const std::string & ch) {
    return __chartype_collections<void>::get_collections()->chartype(ch.c_str());
}

}       //  end for namespace chartypes
}       //  end for namespace strutils
}       //  end for namespace ltp

#endif  //  end for __LTP_STRUTILS_CHARTYPES_HPP__


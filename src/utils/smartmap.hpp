#ifndef __LTP_UTILS_SMARTMAP_HPP__
#define __LTP_UTILS_SMARTMAP_HPP__

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>

namespace ltp {
namespace utility {

struct __SmartMap_Default_HashFunction {
    size_t operator () (const char * s) const {
        unsigned int hash = 0;
        while (*s) {
            hash = hash * 101 + *s ++;
        }
        return size_t(hash);
    }
};

struct __SmartMap_Default_StringEqual {
    bool operator () (const char * s1, const char * s2) const {
        return (strcmp(s1, s2) == 0);
    }
};

struct __SmartMap_Hash_Node {
public:
    unsigned int    __key_off;
    unsigned int    __val_off;
    unsigned int    __freq;
    unsigned int    __hash_val;
    int             __next_off;

    __SmartMap_Hash_Node & operator = (const __SmartMap_Hash_Node & other) {
        __key_off  = other.__key_off;
        __val_off  = other.__val_off;
        __freq     = other.__freq;
        __hash_val = other.__hash_val;
        __next_off = other.__next_off;

        return (*this);
    }
};

template <class T = int>
struct __SmartMap_Const_Iterator {
    typedef __SmartMap_Hash_Node hash_node_t;

    __SmartMap_Const_Iterator(
            const hash_node_t * _ptr,
            const char *        _key_buffer,
            const T *           _val_buffer) :
        ptr(_ptr),
        key_buffer(_key_buffer),
        val_buffer(_val_buffer) {}

    __SmartMap_Const_Iterator() : 
        ptr(0), 
        key_buffer(0), 
        val_buffer(0) {}

    const char * key() { return key_buffer + ptr->__key_off; }
    const T * value() { return val_buffer + ptr->__val_off; }
    int frequency() { return ptr->__freq; }
    bool operator ==(const __SmartMap_Const_Iterator & other) const { return ptr == other.ptr; }
    bool operator !=(const __SmartMap_Const_Iterator & other) const { return ptr != other.ptr; }
    void operator ++() { ++ ptr; }

    const hash_node_t * ptr;
    const char *        key_buffer;
    const T *           val_buffer;
};

template <class T = int, 
         class HashFunction = __SmartMap_Default_HashFunction, 
         class StringEqual  = __SmartMap_Default_StringEqual>
class SmartMap {
public:
    typedef __SmartMap_Hash_Node            hash_node_t;
    typedef __SmartMap_Const_Iterator<T>    const_iterator;

public:
    explicit SmartMap() :
        _num_entries(0),
        _cap_entries(INIT_CAP_ENTRIES),
        _num_buckets(0),
        _cap_buckets_idx(0),
        _len_key_buffer(0),
        _hash_buckets(0),
        _hash_buffer(0),
        _key_buffer(0),
        _val_buffer(0),
        _hash_buckets_volumn(0),
        _cap_key_buffer(INIT_CAP_KEY_BUFFER) {

        _cap_buckets    = PRIMES[_cap_buckets_idx];
        _max_buckets    = int(0.7 * _cap_buckets);
        // allocate memory
        _hash_buckets   = new int[ _cap_buckets ];
        _hash_buffer    = new hash_node_t[_cap_entries];
        _key_buffer     = new char[_cap_key_buffer];
        _val_buffer     = new T[_cap_entries];

        _hash_buckets_volumn = new int[ _cap_buckets ];

        // set the hash_table to be empty
        for (unsigned i = 0; i < _cap_buckets; ++ i) {
            _hash_buckets[i] = -1;
            _hash_buckets_volumn[i] = 0;
        }
    }

    ~SmartMap() {
        clear();
    }

    /**
     * Set the key, value pair to SmartMap, Return true on
     * successfully set, otherwise false
     *
     *  @param[in]  key     the key
     *  @param[in]  val     the value
     *  @return     bool    true on successfully set, otherwise false
     */
    bool set(const char * key, const T & val) {
        bool ret = false;

        unsigned hv  = HashFunction()(key);
        unsigned idx = (hv % _cap_buckets);

        if (-1 == _hash_buckets[idx]) {
            // position in hash table is empty, key must be a new element
            // check if more than 70% element hash table is dirty, if so
            // realloc a new hash table
            _append(key, val, hv, idx);

            _hash_buckets[idx] = (_latest_hash_node - _hash_buffer);
            ++ _num_buckets;

            ret = true;

        } else {
            int p = _find(key, hv, idx, true);
            if (-1 == p) {
                // not find this hash node
                _append(key, val, hv, idx);

                for (p = _hash_buckets[idx]; 
                        _hash_buffer[p].__next_off >= 0; 
                        (p = _hash_buffer[p].__next_off));

                _hash_buffer[p].__next_off = (_latest_hash_node - _hash_buffer);
                ret = true;
            } else {
                // find this hash node
                // maintain sorted by frequency

                int q = -1;
                for (q = _hash_buckets[idx];
                        (_hash_buffer[q].__freq >= _hash_buffer[p].__freq && (q != p));
                        q = _hash_buffer[q].__next_off) ;

                if (_hash_buffer[q].__freq < _hash_buffer[p].__freq) {
                    std::swap((_hash_buffer[q].__freq),     (_hash_buffer[p].__freq));
                    std::swap((_hash_buffer[q].__key_off),  (_hash_buffer[p].__key_off));
                    std::swap((_hash_buffer[q].__val_off),  (_hash_buffer[p].__val_off));
                    std::swap((_hash_buffer[q].__hash_val), (_hash_buffer[p].__hash_val));
                }

                ret = false;
            }
        }

        if (_hash_buckets_volumn[idx] > 5 || _cap_buckets < _num_entries ) {
            // allocate a new bucket and delete the old one
            ++ _cap_buckets_idx;
            _cap_buckets = PRIMES[_cap_buckets_idx];
            _max_buckets = int(_cap_buckets * 0.7);

            // allocate a new bucket
            int * new_hash_buckets_volumn = new int[_cap_buckets];
            int * new_hash_buckets = new int[_cap_buckets];
            for (unsigned i = 0; i < _cap_buckets; ++ i) {
                new_hash_buckets[i] = -1;
                new_hash_buckets_volumn[i] = 0;
            }

            for (unsigned i = 0; i < _num_entries; ++ i) {
                unsigned int hash_val = _hash_buffer[i].__hash_val;
                unsigned int bucket_id = (hash_val % _cap_buckets);
                int freq = _hash_buffer[i].__freq;

                ++ (new_hash_buckets_volumn[bucket_id]);

                if (-1 == new_hash_buckets[bucket_id]) {
                    new_hash_buckets[bucket_id] = i;
                    _hash_buffer[i].__next_off = -1;
                } else {
                    int p = new_hash_buckets[bucket_id];
                    int q = -1;

                    while (p >= 0) {
                        if (_hash_buffer[p].__freq < freq) {
                            break;
                        }

                        q = p;
                        p = _hash_buffer[p].__next_off;
                    }

                    if (-1 == q) {
                        new_hash_buckets[bucket_id] = i;
                        _hash_buffer[i].__next_off = p;
                    } else {
                        _hash_buffer[q].__next_off = i;
                        _hash_buffer[i].__next_off = p;
                    }
                }
            }

            delete [](_hash_buckets_volumn);
            delete [](_hash_buckets);

            _hash_buckets = new_hash_buckets;
            _hash_buckets_volumn = new_hash_buckets_volumn;
        }

        // debug(std::cout);
        return ret;
    }

    /**
     * Get value of the key. Store the value to val and return
     * true when the key exist. Otherwise false
     *
     *  @param[in]  key     the key
     *  @param[out] val     the value
     *  @return     bool    true on key exist, otherwise false
     */
    bool get(const char * key, T & val) {
        unsigned hv = HashFunction()(key);
        unsigned idx = (hv % _cap_buckets);
        int p = _find(key, hv, idx, false);

        if (-1 == p) {
            return false;
        }
        val = _val_buffer[_hash_buffer[p].__val_off];
        return true;
    }

    /**
     * Get value of the key. Return the value's pointer when 
     * the key exist. Otherwise NULL
     *
     *  @param[in]  key     the key
     *  @param[out] val     the value
     *  @return     T *     pointer to the value when key exist,
     *                      otherwise NULL
     */
    T * get(const char * key) {
        unsigned hv = HashFunction()(key);
        unsigned idx = (hv % _cap_buckets);
        int p = _find(key, hv, idx, false);

        if (-1 == p) {
            return NULL;
        }

        return (_val_buffer + (_hash_buffer[p].__val_off));
    }

    /*
     * Get the frequency of the key. Return the key's frequency
     * It is a special usage of smartmap as a key, frequency
     * counter. If the key is not contained, return -1.
     *
     *  @param[in]  key     the key
     *  @return     int     the frequency
     */
    int frequency(const char * key) {
        unsigned hv = HashFunction()(key);
        unsigned idx = (hv % _cap_buckets);
        int p = _find(key, hv, idx, false);

        if (-1 == p) {
            return -1;
        }

        return _hash_buffer[p].__freq;
    }

    /**
     * Return whether the key exist.
     *
     *  @param[in]  key         the key
     *  @param[in]  add_freq    add the key's frequency when
     *                          this handle is set true
     *  @return     bool        true on this key exist, otherwise
     *                          false.
     */
    bool contains(const char * key, bool add_freq = false) {
        unsigned int hv = HashFunction()(key);
        unsigned int idx = (hv % _cap_buckets);

        return (-1 != _find(key, hv, idx, add_freq));
    }

    /*
     * clear the hash table and buffer
     */
    void clear() {
        if (_hash_buckets) {
            delete [](_hash_buckets);
            _hash_buckets = 0;
        }

        if (_hash_buffer) {
            delete [](_hash_buffer);
            _hash_buffer = 0;
        }

        if (_key_buffer) {
            delete [](_key_buffer);
            _key_buffer = 0;
        }

        if (_val_buffer) {
            delete [](_val_buffer);
            _val_buffer = 0;
        }

        if (_hash_buckets_volumn) {
            delete [](_hash_buckets_volumn);
            _hash_buckets_volumn = 0;
        }
    }

    /**
     * Get number of entries
     *
     *  @return int     the number of entries.
     */
    inline size_t size() const {
        return _num_entries;
    }

    const_iterator begin() {
        return const_iterator(_hash_buffer, _key_buffer, _val_buffer);
    }

    const_iterator end() {
        return const_iterator(_hash_buffer + _num_entries, _key_buffer, _val_buffer);
    }


    /**
     * Dump out SmartMap
     *
     *  @param[in/out]  out     the output file stream
     */
    void dump(std::ostream & out) {
        // write header information
        int header_size = 0;
        char header[4] = {'S', 'M', 'A', 'P'};

        unsigned int offset = (unsigned int)out.tellp();

        out.write(header, 4);
        header_size += 4;

        out.write(reinterpret_cast<const char *>(&_num_entries),    sizeof(unsigned int));
        out.write(reinterpret_cast<const char *>(&_len_key_buffer), sizeof(unsigned int));
        out.write(reinterpret_cast<const char *>(&_cap_buckets),    sizeof(unsigned int));
        out.write(reinterpret_cast<const char *>(_hash_buckets),    sizeof(int) * _cap_buckets);
        out.write(reinterpret_cast<const char *>(_hash_buffer),     sizeof(hash_node_t) * _num_entries);
        out.write(reinterpret_cast<const char *>(_key_buffer),      sizeof(char) * _len_key_buffer);
        out.write(reinterpret_cast<const char *>(_val_buffer),      sizeof(T) * _num_entries);
    }

    /**
     * Load SmartMap dump from disk
     *
     *  @param[in]  in      the input file stream
     *  @return     bool    return true when successful loaded.
     */
    bool load(std::istream & in) {
        clear();

        char chunk[4];

        in.read(chunk, 4);
        if (0 != strncmp(chunk, "SMAP", 4)) {
            std::cout << chunk << std::endl;
            return false;
        }

        in.read(reinterpret_cast<char *>(&_num_entries),    sizeof(unsigned int));
        in.read(reinterpret_cast<char *>(&_len_key_buffer), sizeof(unsigned int));
        in.read(reinterpret_cast<char *>(&_cap_buckets),    sizeof(unsigned int));

        _hash_buckets   = new int[_cap_buckets];
        _hash_buffer    = new hash_node_t[_num_entries];
        _key_buffer     = new char[_len_key_buffer];
        _val_buffer     = new T[_num_entries];


        in.read(reinterpret_cast<char *>(_hash_buckets),    sizeof(int) * _cap_buckets);
        in.read(reinterpret_cast<char *>(_hash_buffer),     sizeof(hash_node_t) * _num_entries);
        in.read(reinterpret_cast<char *>(_key_buffer),      sizeof(char) * _len_key_buffer);
        in.read(reinterpret_cast<char *>(_val_buffer),      sizeof(T) * _num_entries);

        return true;
    }

    void debug(std::ostream & out) {
        out << "===== SMARTMAP DEBUG =====" << std::endl;
        out << "number of buckets: " << _num_buckets << std::endl;
        out << "capacity of buckets: " << _cap_buckets << std::endl;
        out << "number of entries: " << _num_entries << std::endl;
        out << "capacity of entries: " << _cap_entries << std::endl;
        out << "hash bucket address: " << _hash_buckets << std::endl;
        for (int i = 0; i < _cap_buckets; ++ i) {
            out << "[" << i << "]";
            int p = _hash_buckets[i];
            while (p >= 0) {
                out << " -> " << p << "(" << _hash_buffer + p << ")";
                p = _hash_buffer[p].__next_off;
            }
            out << std::endl;
        }
        out << std::endl;
        out << "hash node buffer address: " << _hash_buffer << std::endl;
        for (int i = 0; i < _num_entries; ++ i) {
            out << _hash_buffer + i 
                << " "
                << "(\"" << _key_buffer + _hash_buffer[i].__key_off << "\""
                << ", " << _hash_buffer[i].__hash_val
                << ", " << _val_buffer[_hash_buffer[i].__val_off]
                << ", " << _hash_buffer[i].__freq
                << ", " << _hash_buffer[i].__next_off
                << ")" << std::endl;
        }
        out << "==========================" << std::endl << std::endl;
    }

protected:

    static const unsigned int INIT_CAP_BUCKETS    = 256;
    static const unsigned int INIT_CAP_ENTRIES    = 256;
    static const unsigned int INIT_CAP_KEY_BUFFER = 1024;
    static const unsigned int PRIMES[100]; 

protected:
    int *           _hash_buckets;
    int *           _hash_buckets_volumn;
    hash_node_t *   _hash_buffer;
    char *          _key_buffer;    /*< the buffer of key */
    T *             _val_buffer;    /*< the buffer of value */


protected:
    /*< buckets related counter */
    unsigned int    _num_buckets;
    unsigned int    _cap_buckets;
    unsigned int    _max_buckets;
    unsigned int    _cap_buckets_idx;

    /*< entries related counter */
    unsigned int    _num_entries;
    unsigned int    _cap_entries;

    /*< buffer related counter */
    unsigned int    _len_key_buffer;
    unsigned int    _cap_key_buffer;

    unsigned int    _father;

    char *          _latest_key;
    T *             _latest_val;
    hash_node_t *   _latest_hash_node;

protected:
    /**
     * internal function for appending a (key, value, frequence) 
     * tuple into the pool
     *
     *  @param[in]  key     the key
     *  @param[in]  val     the value
     */
    void _append(const char * key, const T & val, const int hv, const int idx) {
        int len = strlen(key) + 1;

        // if key buffer is not enough
        if ( _cap_key_buffer <= (_len_key_buffer + len) ) {
            // duplicate the key buffer capicity
            _cap_key_buffer = (_len_key_buffer + len) << 1;

            // allocate new memory buffer
            char * new_key_buffer = new char[ _cap_key_buffer ];

            // copy the old buffer to the new buffer
            memcpy(new_key_buffer, _key_buffer, _len_key_buffer);

            delete [](_key_buffer);
            // 
            _key_buffer = new_key_buffer;
        }

        // update the latest key position
        _latest_key = _key_buffer + _len_key_buffer;

        // copy the key to the latest key position
        memcpy( _latest_key, key, len );

        // increase the buffer length
        _len_key_buffer += len;

        // if the hash buffer and value buffer is not enough
        if ( _cap_entries <= (_num_entries + 1) ) {
            // duplicate the capacity of the entries;
            _cap_entries = (_num_entries + 1) << 1;

            T * new_val_buffer = new T[_cap_entries];
            //memcpy(new_val_buffer, _val_buffer, sizeof(T) * _num_entries);
            std::copy(_val_buffer, _val_buffer + _num_entries, new_val_buffer);
            delete [](_val_buffer);
            _val_buffer = new_val_buffer;

            hash_node_t * new_hash_buffer = new hash_node_t[_cap_entries];
            // memcpy(new_hash_buffer, _hash_buffer, sizeof(hash_node_t) * _num_entries);
            /*for (int i = 0; i < _num_entries; ++ i) {
                new_hash_buffer[i] = _hash_buffer[i];
            }*/
            std::copy(_hash_buffer, _hash_buffer + _num_entries, new_hash_buffer);

            delete [](_hash_buffer);
            _hash_buffer = new_hash_buffer;
        }

        _latest_hash_node = _hash_buffer + _num_entries;
        _latest_val = _val_buffer + _num_entries;
        (*_latest_val) = val;

        _latest_hash_node->__key_off    = _latest_key - _key_buffer;
        _latest_hash_node->__val_off    = _num_entries;
        _latest_hash_node->__hash_val   = hv;
        _latest_hash_node->__freq       = 1;
        _latest_hash_node->__next_off   = -1;

        ++ _num_entries;
        ++ (_hash_buckets_volumn[idx]);
    }

    /**
     * Internal function for find a key in hash table. Return the
     * position of the key in hash buffer. If key not found, 
     * return -1.
     *
     *  @param[in]  key         the key
     *  @param[in]  hv          the hash value
     *  @param[in]  idx         the hash table index
     *  @param[in]  add_freq    if this is set true, hash frequency
     *                          is added when the hash entry is found
     *  @return     int         the position of the hash node in hsh buffer
     */
    int _find(const char * key, unsigned hv, unsigned idx, bool add_freq) {
        int p = _hash_buckets[idx];

        while (p >= 0) {

            if (_hash_buffer[p].__hash_val != hv) {
                p = _hash_buffer[p].__next_off;
            } else {
                if ( StringEqual()((_key_buffer + _hash_buffer[p].__key_off), key) ) {
                    if (add_freq) {
                        ++ (_hash_buffer[p].__freq);
                    }
                    return p;
                } else {
                    _father = p;
                    p = _hash_buffer[p].__next_off;
                }
            }
        }

        return -1;
    }
};

template <class T, class HashFunction, class StringEqual>
const unsigned int SmartMap<T, HashFunction, StringEqual>::PRIMES[100] = {
    53,         97,         193,        389,        769,
    1543,       3079,       6151,       12289,      24593,
    49157,      98317,      196613,     393241,     786433,
    1572869,    3145739,    6291469,    12582917,   25165843,
    50331653,   100663319,  201326611,  402653189,  805306457,
    1610612741,
};

class IndexableSmartMap : public SmartMap<int> {
public:
    IndexableSmartMap() : entries(0), cap_entries(0) {}

    ~IndexableSmartMap() {
        if (entries) {
            delete [](entries);
        }
    }

private:
    int cap_entries;
    int * entries;

public:
    /**
     * push a key to the labelcollections
     *
     *  @param[in]  key     the key
     *  @return     int     the index of the key
     */
    // offsets of the key in hashmap key buffer is stored in entries.
    // when a new key is insert into the collection, check if entries
    // buffer is big enough. if not, duplicate the memory.
    int push(const char * key) {
        if (!SmartMap<int>::contains(key)) {
            int idx = SmartMap<int>::size();
            set(key, idx);

            if (cap_entries < SmartMap<int>::_num_entries) {
                cap_entries = ( SmartMap<int>::_num_entries << 1);
                int * new_entries = new int[cap_entries];
                if ( entries ) {
                    memcpy(new_entries, entries, sizeof(int) * (SmartMap<int>::_num_entries - 1));
                    delete [](entries);
                }
                entries = new_entries;
            }

            // SmartMap<int>::debug(cout);
            entries[_num_entries-1] = SmartMap<int>::_latest_hash_node->__key_off;
            return idx;
        } else {
            return (*SmartMap<int>::get(key));
        }

        return -1;
    }

    int push(const std::string & key) {
        return push(key.c_str());
    }

    /**
     * get the key whose index is i
     *
     *  @param[in]  i               the index
     *  @return     const char *    pointer to the key
     */
    const char * at(int i) {
        if (i >= 0 && i < _num_entries) {
            return SmartMap<int>::_key_buffer + entries[i];
        } else {
            return 0;
        }
    }

    /**
     * get the index of the key. if the key doesn't exist, return -1
     *
     *  @param[in]  key             the key
     *  @return     int             index of the key if exist, otherwise -1
     */
    int index( const char * key ) {
        int val = -1;
        if (SmartMap<int>::get(key, val)) {
            return val;
        }

        return -1;
    }

    int index( const std::string & key) {
        return index(key.c_str());
    }

    /**
     * dump the collection to output stream
     *
     *  @param[out] out     the output stream
     */
    void dump(std::ostream & out) {
        SmartMap<int>::dump(out);
        out.write(reinterpret_cast<const char *>(entries), sizeof(int) * _num_entries);
    }

    /**
     * load the collections from input stream.
     *
     *  @param[in]  in      the input stream.
     *  @return     bool    true on success, otherwise false
     */
    bool load(std::istream & in) {
        bool ret = SmartMap<int>::load(in);
        if (!ret) {
            return ret;
        }

        if (entries) {
            delete [](entries);
        }

        entries = new int[SmartMap<int>::_num_entries];
        if (!entries) {
            return false;
        }

        in.read(reinterpret_cast<char *>(entries), sizeof(int) * _num_entries);
        return true;
    }
};

}       //  end for namespace strutils
}       //  end for namespace ltp

#endif  //  end for __SMARTMAP_HPP__

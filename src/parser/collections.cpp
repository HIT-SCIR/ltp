#include "parser/collections.h"

namespace ltp {
namespace parser {

DictionaryCollections::DictionaryCollections(int num_dicts) :
  idx(0) {
  dicts.resize( num_dicts );

  for (int i = 0; i < num_dicts; ++ i) {
    dicts[i] = new Dictionary( this );
  }
}

DictionaryCollections::~DictionaryCollections() {
  for (int i = 0; i < dicts.size(); ++ i) {
    delete dicts[i];
  }
}

Dictionary *
DictionaryCollections::getDictionary(int i) {
  if (i < dicts.size()) {
    return dicts[i];
  }

  return NULL;
}

int
DictionaryCollections::retrieve(int tid, const char * key, bool create) {
  return dicts[tid]->retrieve(key, create);
}

size_t
DictionaryCollections::dim() const {
  return idx;
}

int
DictionaryCollections::size() {
  return dicts.size();
}

void
DictionaryCollections::dump(ostream & out) {
  char chunk[32];
  unsigned int sz = dicts.size();
  strncpy(chunk, "collections", 16);

  out.write(chunk, 16);
  out.write(reinterpret_cast<const char *>(&idx), sizeof(int));
  out.write(reinterpret_cast<const char *>(&sz), sizeof(unsigned int));
  for (int i = 0; i < dicts.size(); ++ i) {
    // strncpy(chunk, dicts[i]->dict_name.c_str(), 32);
    // out.write(chunk, 32);

    dicts[i]->database.dump(out);
  }
}

bool
DictionaryCollections::load(istream & in) {
  char chunk[32];
  unsigned int sz;

  in.read(chunk, 16);
  if (strcmp(chunk, "collections")) {
    return false;
  }

  in.read(reinterpret_cast<char *>(&idx), sizeof(int));
  in.read(reinterpret_cast<char *>(&sz), sizeof(unsigned int));

  if (sz != dicts.size()) {
    return false;
  }

  for (unsigned i = 0; i < sz; ++ i) {
    // in.read(chunk, 32);

    // Dictionary * dict = new Dictionary(this);
    if (!dicts[i]->database.load(in)) {
      return false;
    }

    // dicts[i].push_back(dict);
  }

  return true;
}

}   //  end for namespace parser
}   //  end for namespace ltp

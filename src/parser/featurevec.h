#ifndef __LTP_PARSER_FEATURE_VECTOR_H__
#define __LTP_PARSER_FEATURE_VECTOR_H__

#include <iostream>
#include <fstream>

namespace ltp {
namespace parser {

struct FeatureVector {
  FeatureVector() : n(0), idx(0), val(0), loff(0) {}
  ~FeatureVector() {
    // clear();
  }

  int n;
  int * idx; 
  double * val;
  int loff;

  /*
   * clear the FeatureVector
   */
  void clear() {
    if (idx) {
      delete [](idx);
      idx = 0;
    }

    if (val) {
      delete [](val);
      val = 0;
    }
  }
};

struct FeatureVectorDB : public FeatureVector {
  FeatureVectorDB() : offset(-1) {}
  ~FeatureVectorDB() {
  }

  long long offset;

  /*
   * Write the feature vector to file, return offset of the featurevec
   * This method is discarded because of the low performance of file
   * operation.
   *
   *  @param[in]  ofs   the output filestream
   *  @return   int   offset of the feature.
   */
  long long write(std::ostream & ofs) {
    // 
    if (n <= 0 || idx == 0) {
      offset = -1;
      return -1;
    }

    char ch = (val == 0 ? 0 : 1);
    offset = ofs.tellp();

    ofs.write(&ch, 1);
    ofs.write(reinterpret_cast<const char *>(&n), sizeof(int));
    ofs.write(reinterpret_cast<const char *>(idx), sizeof(int) * n);
    if (val) {
      ofs.write(reinterpret_cast<const char *>(val), sizeof(double) * n);
    }

    return offset;
  }

  /*
   * Read the feature vector from filestream, This method is discarded
   * because of the low performance of file operation.
   *
   *  @param[in]  ifs   the input filestream
   */
  int read(std::istream & ifs) {
    if (offset < 0) {
      return -1;
    }

    ifs.seekg(offset);

    char ch = 0;
    ifs.read(&ch, 1);
    ifs.read(reinterpret_cast<char *>(&n), sizeof(int));
    idx = new int[n];
    ifs.read(reinterpret_cast<char *>(idx), sizeof(int) * n);
    if (ch) {
      val = new double[n];
      ifs.read(reinterpret_cast<char *>(val), sizeof(double) * n);
    }

    return 0;
  }

  /*
   * free memory of feature vector.
   */
  void nice() {
    if (idx) {
      delete [](idx);
      idx = 0;
    }

    if (val) {
      delete [](val);
      val = 0;
    }
  }
};

}     //  end for namespace parser
}     //  end for namespace ltp

#endif  //  end for __LTP_PARSER_FEATURE_VECTOR_H__

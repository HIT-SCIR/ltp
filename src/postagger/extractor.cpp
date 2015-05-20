#include "postagger/extractor.h"
#include "postagger/settings.h"
#include "utils/codecs.hpp"
#include "utils/strutils.hpp"
#include "utils/chartypes.hpp"

namespace ltp {
namespace postagger {

using strutils::codecs::decode;
using strutils::to_str;
using utility::StringVec;
using utility::Template;

std::vector<Template *> Extractor::templates;

Extractor& Extractor::extractor() {
  static Extractor instance_;
  return instance_;
}

int Extractor::num_templates() {
  extractor();
  return templates.size();
}

Extractor::Extractor() {
  templates.push_back(new Template("1={c-2}"));
  templates.push_back(new Template("2={c-1}"));
  templates.push_back(new Template("3={c-0}"));
  templates.push_back(new Template("4={c+1}"));
  templates.push_back(new Template("5={c+2}"));
  templates.push_back(new Template("6={c-1}-{c-0}"));
  templates.push_back(new Template("7={c-0}-{c+1}"));
  templates.push_back(new Template("8={c-1}-{c+1}"));
  templates.push_back(new Template("12={prefix}"));
  templates.push_back(new Template("13={suffix}"));
}

Extractor::~Extractor() {
  for (int i = 0; i < templates.size(); ++ i) {
    delete templates[i];
  }
}

int Extractor::extract1o(const Instance& inst, int idx,
    std::vector<StringVec>& cache) {
  int len = inst.size();
  std::vector<std::string> chars;
  decode(inst.forms[idx], chars);

  Template::Data data;

  data.set( "c-2",  (idx-2 < 0 ? BOS : inst.forms[idx-2]) ); 
  data.set( "c-1",  (idx-1 < 0 ? BOS : inst.forms[idx-1]) );
  data.set( "c-0",  inst.forms[idx] );
  data.set( "c+1",  (idx+1 >= len ? EOS : inst.forms[idx+1]) );
  data.set( "c+2",  (idx+2 >= len ? EOS : inst.forms[idx+2]) );
  int length = inst.forms[idx].size(); length = (length < 5 ? length : 5);
  data.set( "len",  to_str(length));

  std::string feat;
  feat.reserve(1024);
  int N = templates.size();

  // 1-9 basic feature
  for (int i = 0; i < N - 2; ++ i) {
    templates[i]->render(data, feat);
    cache[i].push_back(feat);
  }

  // 12-13 prefix and suffix feature.
  for (int i = N - 2; i < N; ++ i) {
    std::string prefix = "";
    std::string suffix = "";
    int num_chars = chars.size();
    for (int j = 0; j < num_chars && j < 3; ++ j) {
      prefix = prefix + chars[j];
      suffix = chars[num_chars-j-1] + suffix;
      data.set( "prefix", prefix);
      data.set( "suffix", suffix);
      templates[i]->render(data, feat);
      cache[i].push_back(feat);
    }
  }

  return 0;
}

}     //  end for namespace postagger
}     //  end for namespace ltp

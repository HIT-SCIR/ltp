#include "ner/extractor.h"
#include "ner/settings.h"
#include "utils/strutils.hpp"
#include "utils/chartypes.hpp"

namespace ltp {
namespace ner {

using utility::Template;
using utility::StringVec;

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
  templates.push_back(new Template("6={c-2}-{c-1}"));
  templates.push_back(new Template("7={c-1}-{c-0}"));
  templates.push_back(new Template("8={c-0}-{c+1}"));
  templates.push_back(new Template("9={c+1}-{c+2}"));
  templates.push_back(new Template("10={p-2}"));
  templates.push_back(new Template("11={p-1}"));
  templates.push_back(new Template("12={p-0}"));
  templates.push_back(new Template("13={p+1}"));
  templates.push_back(new Template("14={p+2}"));
  templates.push_back(new Template("15={p-2}-{p-1}"));
  templates.push_back(new Template("16={p-1}-{p-0}"));
  templates.push_back(new Template("17={p-0}-{p+1}"));
  templates.push_back(new Template("18={p+1}-{p+2}"));
}

Extractor::~Extractor() {
  for (size_t i = 0; i < templates.size(); ++ i) {
    delete templates[i];
  }
}

int Extractor::extract1o(const Instance& inst, int idx,
    std::vector<StringVec>& cache) {
  size_t len = inst.size();
  Template::Data data;

#define EQU(x, y) (inst.forms[(x)] == inst.forms[(y)])
#define TYPE(x) (strutils::to_str(inst->chartypes[(x)]&0x07))

  data.set( "c-2", (idx-2 < 0 ? BOS : inst.forms[idx-2]) );
  data.set( "c-1", (idx-1 < 0 ? BOS : inst.forms[idx-1]) );
  data.set( "c-0", inst.forms[idx] );
  data.set( "c+1", (idx+1 >= len ? EOS : inst.forms[idx+1]) );
  data.set( "c+2", (idx+2 >= len ? EOS : inst.forms[idx+2]) );
  data.set( "p-2", (idx-2 < 0 ? BOP : inst.postags[idx-2]) );
  data.set( "p-1", (idx-1 < 0 ? BOP : inst.postags[idx-1]) );
  data.set( "p-0", inst.postags[idx] );
  data.set( "p+1", (idx+1 >= len ? EOP : inst.postags[idx+1]) );
  data.set( "p+2", (idx+2 >= len ? EOP : inst.postags[idx+2]) );

#undef TYPE
#undef EQU

  std::string feat;
  feat.reserve(1024);
  for (size_t i = 0; i < templates.size(); ++ i) {
    templates[i]->render(data, feat);
    cache[i].push_back(feat);
  }
  return 0;
}

}       //  end for namespace ner
}       //  end for namespace ltp

#include "postagger/extractor.h"
#include "postagger/settings.h"
#include "utils/codecs.hpp"
#include "utils/strutils.hpp"
#include "utils/chartypes.hpp"

namespace ltp {
namespace postagger {

Extractor * Extractor::instance_ = 0;
std::vector<Template *> Extractor::templates;

Extractor * Extractor::extractor() {
  if (0 == instance_) {
    instance_ = new Extractor;
  }

  return instance_;
}

int Extractor::num_templates() {
  if (0 == instance_) {
    instance_ = new Extractor;
  }

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
  //templates.push_back(new Template("9={ct-1}"));
  //templates.push_back(new Template("10={ct-0}"));
  //templates.push_back(new Template("11={ct+1}"));
  //templates.push_back(new Template("7={c-1}-{c-0}-{c+1}"));
  //templates.push_back(new Template("9={len}"));
  //templates.push_back(new Template("9={ch-0,0}-{ch-0,n}"));
  // templates.push_back(new Template("10={ch-1,n}-{ch-0,0}"));
  //templates.push_back(new Template("11={ch-0,n}-{ch+1,0}"));
  templates.push_back(new Template("12={prefix}"));
  templates.push_back(new Template("13={suffix}"));
  //templates.push_back(new Template("14={pos}"));
  //templates.push_back(new Template("14={ct-1}"));
  //templates.push_back(new Template("15={ct-0}"));
  //templates.push_back(new Template("16={ct+1}"));
  //templates.push_back(new Template("17={dup-1}"));
  //templates.push_back(new Template("18={dup-0}"));
  //templates.push_back(new Template("19={dup2-2}"));
  //templates.push_back(new Template("20={dup2-1}"));
  //templates.push_back(new Template("21={dup2-0}"));
}

Extractor::~Extractor() {
  for (int i = 0; i < templates.size(); ++ i) {
    delete templates[i];
  }
}

int Extractor::extract1o(Instance * inst, int idx, std::vector< StringVec > & cache) {
  int len = inst->size();

  if (inst->chars.size() == 0) {
    inst->chars.resize(len);
    for (int i = 0; i < len; ++ i) {
      strutils::codecs::decode(inst->forms[i], inst->chars[i]);
    }
  }

  Template::Data data;

  //#define TYPE(x) (strutils::to_str(inst->wordtypes[(x)]))

  data.set( "c-2",  (idx-2 < 0 ? BOS : inst->forms[idx-2]) ); 
  data.set( "c-1",  (idx-1 < 0 ? BOS : inst->forms[idx-1]) );
  data.set( "c-0",  inst->forms[idx] );
  data.set( "c+1",  (idx+1 >= len ? EOS : inst->forms[idx+1]) );
  data.set( "c+2",  (idx+2 >= len ? EOS : inst->forms[idx+2]) );
  //data.set( "ct-1", (idx-1 < 0 ? BOT : TYPE(idx-1)) );
  //data.set( "ct-0", TYPE(idx) );
  //data.set( "ct+1", (idx+1 >= len ? EOT : TYPE(idx+1)) );

  int length = inst->forms[idx].size(); length = (length < 5 ? length : 5);
  data.set( "len",  strutils::to_str(length));

  data.set( "ch-1,n", (idx-1 < 0 ? BOC : inst->chars[idx-1][inst->chars[idx-1].size()-1]));
  data.set( "ch-0,0", inst->chars[idx][0] );
  data.set( "ch-0,n", inst->chars[idx][inst->chars[idx].size()-1]);
  data.set( "ch+1,0", (idx+1 >= len ? EOC : inst->chars[idx+1][0]));

  string feat;
  feat.reserve(1024);

  int N = templates.size();
  for (int i = 0; i < N - 2; ++ i) {
    templates[i]->render(data, feat);
    cache[i].push_back(feat);
  }

  for (int i = N - 2; i < N; ++ i) {
    string prefix = "";
    string suffix = "";
    int num_chars = inst->chars[idx].size();
    for (int j = 0; j < num_chars && j < 3; ++ j) {
      prefix = prefix + inst->chars[idx][j];
      suffix = inst->chars[idx][num_chars-j-1] + suffix;

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

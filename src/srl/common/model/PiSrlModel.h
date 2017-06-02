//
// Created by liu on 2017/5/12.
//

#ifndef BILSTM_SRL_PISRLMODEL_H
#define BILSTM_SRL_PISRLMODEL_H

#include <base/debug.h>
#include <model/SeqLabelModel.h>
#include "../structure/SrlPiSample.h"
#include "Const.h"

// model builders
#include <model/LookupModelBuilder.h>
#include <model/BiRNNModelBuilder.h>
#include <model/AffineTransformModelBuilder.h>

class PiSrlModel : public SeqLabelModel<SrlPiSample> {



public:
  enum Look { WORD = 0, POS, REL, POSITION, ARG, ALL };

  PiSrlModel(ModelConf &config) : SeqLabelModel<SrlPiSample>(config)
  { }

  void registerDict(vector<SrlPiSample>& samples) {
    dict.resize(ALL);
    dict[WORD].convert(ROOT_MARK);
    dict[POS].convert(ROOT_MARK);
    dict[REL].convert(ROOT_MARK);
    for (int j = 0; j < samples.size(); ++j) {
      for (int k = 0; k < samples[j].size(); ++k) {
        dict[WORD].convert(samples[j].getWord(k).getWord());
        dict[POS].convert(samples[j].getWord(k).getPos());
        dict[REL].convert(samples[j].getWord(k).getRel());
        dict[POSITION].convert(samples[j].getWord(k).getPosition());
        vector<string>& args = samples[j].getWord(k).getArgs();
        for (auto i = args.begin(); i != args.end(); i++) {
          dict[ARG].convert(*i);
        }
      }
    }

    freezeDict();
    dict[WORD].set_unk(UNK_WORD);
    dict[POS].set_unk(UNK_WORD);
    dict[REL].set_unk(UNK_WORD);
  }
};


#endif //BILSTM_SRL_PISRLMODEL_H

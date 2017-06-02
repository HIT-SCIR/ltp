//
// Created by liu on 2017/5/21.
//

#include "SrlSrlModel.h"

vector<Expression> SrlSrlModel::label(ComputationGraph &hg, SrlPiSample &samples) {
  ctx_sent_lstm.newGraph(hg); stx_sent_lstm.newGraph(hg); stx_rel_lstm.newGraph(hg);
  vector<Expression> ans;
  vector<int> predicates = samples.getPredicateList();
  for (int j = 0; j < predicates.size(); ++j) {
    vector<Expression> one_pred_label = labelOnePredicate(hg, samples, predicates[j]);
    ans.insert(ans.end(), one_pred_label.begin(), one_pred_label.end());
  }
  return ans;
}

vector<Expression> SrlSrlModel::labelOnePredicate(ComputationGraph &hg, SrlPiSample &samples, int predIndex)  {
  vector<Expression> sentList, relList;
  Expression sent_root = activate(sentTransform.forward(hg, {
          word_lookup.forward(hg, (unsigned) dict[WORD].convert(ROOT_MARK)),
          dynet::expr::input(hg, {config.emb_dim}, emb_lookup.getEmb(ROOT_MARK)),
          pos_lookup.forward(hg, (unsigned) dict[POS].convert(ROOT_MARK)),
  }));
  Expression rel_root = rel_lookup.forward(hg, (unsigned) dict[REL].convert(ROOT_MARK));
  for (int j = 0; j < samples.size(); ++j) {
    vector<Expression> sent;
    if (config.word_dim) {
      sent.push_back(word_lookup.forward(hg, (unsigned) dict[WORD].convert(samples.getWord(j).getWord())));
    }
    if (config.emb_dim) {
      sent.push_back(dynet::expr::input(hg, {config.emb_dim}, emb_lookup.getEmb(samples.getWord(j).getWord())));
    }
    if (config.pos_dim) {
      sent.push_back(pos_lookup.forward(hg, (unsigned) dict[POS].convert(samples.getWord(j).getPos())));
    }
    sentList.push_back(activate(sentTransform.forward(hg, sent)));

    if (config.rel_dim) {
      relList.push_back(rel_lookup.forward(hg, (unsigned) dict[REL].convert(samples.getWord(j).getRel())));
    }
  }
  ctx_sent_lstm.startNewSequence();
  vector<Expression> sent_lstm_out_list = ctx_sent_lstm.forward(hg, sentList);
  vector<Expression> ans;
  for (int j = 0; j < samples.size(); ++j) {
    vector<int> pred_path, arg_path;
    getStnPath(samples, predIndex, j, pred_path, arg_path);
    stx_sent_lstm.startNewSequence(); stx_rel_lstm.startNewSequence();
    Expression stx_sent_out = stx_sent_lstm.forwardBy2Order(hg, sentList, pred_path, arg_path, sent_root);
    Expression stx_rel_out = stx_rel_lstm.forwardBy2Order(hg, relList, pred_path, arg_path, rel_root);

    Expression position = position_lookup.forward(hg, dict[POSITION].convert(samples.getWord(j).getPosition()));
    Expression hidden = hiddenTransform.forward(hg, {
            position,
            sent_lstm_out_list[j],
            sent_lstm_out_list[predIndex],
            stx_sent_out,
            stx_rel_out
    });
    ans.push_back(softmax(resultTransform.forward(hg, {activate(hidden)})));
  }
  return ans;
}

void SrlSrlModel::getStnPath(SrlPiSample &samples, int predIndex, int argIndex, vector<int> &predPath,
                             vector<int> &argPath)  {
  vector<bool> is_on_pred_path(samples.size(), false);
  predPath.resize(0); argPath.resize(0);
  for (int p = predIndex; p != -1; p = samples.getWord(p).getParent()) {
    is_on_pred_path[p] = true;
    predPath.push_back(p);
  }
  predPath.push_back(-1);

  int nca;
  for (nca = argIndex; nca != -1; nca = samples.getWord(nca).getParent()) {
    if (is_on_pred_path[nca]) break;
    argPath.push_back(nca);
  }
  argPath.push_back(nca);

  if (nca != -1) {
    predPath.erase(find(predPath.begin(), predPath.end(), nca) + 1, predPath.end());
  }
  return;
}

void SrlSrlModel::init()  {
  // sent
  vector<unsigned int> sentDims;
  if (config.word_dim) {
    word_lookup = LookupModelBuilder(dict[WORD].size(), config.word_dim); word_lookup.init(model);
    sentDims.push_back(config.word_dim);
  }
  if (config.emb_dim) {
    sentDims.push_back(config.emb_dim);
  }
  if (config.pos_dim) {
    pos_lookup = LookupModelBuilder(dict[POS].size(), config.pos_dim); pos_lookup.init(model);
    sentDims.push_back(config.pos_dim);
  }

  sentTransform = AffineTransformModelBuilder(sentDims, config.lstm_input_dim); sentTransform.init(model);
  ctx_sent_lstm = BiLSTMModelBuilder(config.layers, config.lstm_input_dim, config.lstm_hidden_dim); ctx_sent_lstm.init(model);

  if (config.rel_dim) {
    rel_lookup = LookupModelBuilder(dict[REL].size(), config.rel_dim); rel_lookup.init(model);
  }
  stx_sent_lstm = BiLSTMModelBuilder(config.layers, config.lstm_input_dim, config.lstm_hidden_dim); stx_sent_lstm.init(model);
  stx_rel_lstm = BiLSTMModelBuilder(config.layers, config.rel_dim, config.lstm_hidden_dim); stx_rel_lstm.init(model);

  if (config.position_dim) {
    position_lookup = LookupModelBuilder(dict[POSITION].size(), config.position_dim); position_lookup.init(model);
  }

  hiddenTransform = AffineTransformModelBuilder({
                                                        config.position_dim,
                                                        config.lstm_hidden_dim, // pred_sent
                                                        config.lstm_hidden_dim, // arg_sent
                                                        config.lstm_hidden_dim, // stx_sent
                                                        config.lstm_hidden_dim // stx_rel
                                                }, config.hidden_dim); hiddenTransform.init(model);
  resultTransform = AffineTransformModelBuilder({config.hidden_dim}, dict[ARG].size()); resultTransform.init(model);
}

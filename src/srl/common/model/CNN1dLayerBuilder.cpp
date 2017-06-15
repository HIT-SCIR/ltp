//
// Created by liu on 2017/4/7.
//

#include "CNN1dLayerBuilder.h"

CNN1dLayerBuilder::CNN1dLayerBuilder(int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps, int out_length)
        : in_rows(in_rows), k_fold_rows(k_fold_rows), filter_width(filter_width),
          in_nfmaps(in_nfmaps), out_nfmaps(out_nfmaps), out_length(out_length) {

  if (k_fold_rows < 1 || ((in_rows / k_fold_rows) * k_fold_rows != in_rows)) {
    cerr << "Bad k_fold_rows=" << k_fold_rows << endl;
    abort();
  }
}

void CNN1dLayerBuilder::init(dynet::Model &model) {
  p_filts.resize(in_nfmaps); p_fbias.resize(in_nfmaps);
  for (int i = 0; i < in_nfmaps; ++i) {
    p_filts[i].resize((unsigned long) out_nfmaps);
    p_fbias[i].resize((unsigned long) out_nfmaps);
    for (int j = 0; j < out_nfmaps; ++j) {
      p_filts[i][j] = model.add_parameters({(unsigned)in_rows, (unsigned)filter_width}, 0.01);
      p_fbias[i][j] = model.add_parameters({(unsigned)in_rows}, 0.05);
    }
  }
}

vector<Expression>
CNN1dLayerBuilder::forward(dynet::ComputationGraph &cg, const vector<Expression> &inlayer) {
  const unsigned out_nfmaps = (const unsigned int) p_filts.front().size();
  const unsigned in_nfmaps = (const unsigned int) p_filts.size();
  if (in_nfmaps != inlayer.size()) {
    cerr << "Mismatched number of input features (" << inlayer.size() << "), expected " << in_nfmaps << endl;
    abort();
  }
  vector<Expression> r(out_nfmaps);

  vector<Expression> tmp(in_nfmaps);
  for (unsigned fj = 0; fj < out_nfmaps; ++fj) {
    for (unsigned fi = 0; fi < in_nfmaps; ++fi) {
      Expression t = conv2d(inlayer[fi], parameter(cg, p_filts[fi][fj]), {1, 1});
      t = colwise_add(t, parameter(cg, p_fbias[fi][fj]));
      tmp[fi] = t;
    }
    Expression s = sum(tmp);
    if (k_fold_rows > 1)
      s = fold_rows(s, (unsigned int) k_fold_rows);
    s = kmax_pooling(s, (unsigned int) out_length);
    r[fj] = rectify(s);
  }
  return r;
}

#include "DepDecoder.h"

void DepDecoder::getTypes(const MultiArray<double> &nt_probs, int len, MultiArray<int> &type) 
{
	vector<unsigned int> type_dim(2);
	unsigned int type_pos;
	type.setDemisionVal(type_dim, len, len);
	type.resize(type_dim);
	type.setDemisionVal(type_dim, 0, 0);
	type.getElement(type_dim, type_pos);
	int i = 0;
	for(; i < len; i++) {
		int j = 0;
		for(; j < len; j++) {
			if(i == j) {
				type.getElement(type_pos++) = -1; continue; 
			}
			int wh = -1;
			double best = DOUBLE_NEGATIVE_INFINITY;
			int t = 0;
			for(; t < pipe.m_vecTypes.size(); ++t) {
				double score = 0.0;
				vector<unsigned int> nt_dim1;
				vector<unsigned int> nt_dim2;
				unsigned int nt_pos1;
				unsigned int nt_pos2;
				if(i < j) {
					nt_probs.setDemisionVal(nt_dim1, i, t, 0, 1);	// <-
					nt_probs.setDemisionVal(nt_dim2, j, t, 0, 0);	// ->
					score = nt_probs.getElement(nt_dim1, nt_pos1) + nt_probs.getElement(nt_dim2, nt_pos2);
				}
				else {
					nt_probs.setDemisionVal(nt_dim1, i, t, 1, 1);
					nt_probs.setDemisionVal(nt_dim2, j, t, 1, 0);	
					score = nt_probs.getElement(nt_dim1, nt_pos1) + nt_probs.getElement(nt_dim2, nt_pos2);
				}

				if(score > best + EPS) { wh = t; best = score; }
			}
			if (wh < 0) {
				cerr << "DepDecoder::getTypes(): type index err: " << wh << endl;
			}
			type.getElement(type_pos++) = wh; // i->j
		}
	}
}

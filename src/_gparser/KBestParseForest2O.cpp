#include "KBestParseForest2O.h"

void KBestParseForest2O::viterbi( DepInstance *inst, 
							   MultiArray<FeatureVec> &fvs, 
							   MultiArray<double>  &probs, 	
							   MultiArray<FeatureVec> &fvs_trips,
							   MultiArray<double> &probs_trips,
							   MultiArray<FeatureVec> &fvs_sibs,
							   MultiArray<double> &probs_sibs,
							   MultiArray<FeatureVec> &nt_fvs,
							   MultiArray<double> &nt_probs,	
							   const MultiArray<int> &static_types, bool isLabeled)
{
	int s = 0;
	for(; s < inst->size(); s++) {
		add(s, -1, 0, 0.0, FeatureVec());	// -1:type; 0:dir; 
		add(s, -1, 1, 0.0, FeatureVec());
 	}

	vector<unsigned int> fvs_dim;
	unsigned int fvs_pos;
	vector<unsigned int> trip_dim;
	unsigned int trip_pos;
	vector<unsigned int> sib_dim;
	unsigned int sib_pos;	
	vector<unsigned int> type_dim;
	unsigned int type_pos;
	vector<unsigned int> nt_dim;
	unsigned int nt_pos;

	vector<unsigned int> chart_dim;
	unsigned int chart_pos;
	vector<unsigned int> chart_dim2;
	unsigned int chart_pos2;
//	cerr << "\n2-order viterbi start" << endl;
	int j = 1;
	for(; j < inst->size(); j++) 
	{
//		cerr << "j : " << j << endl;
		int s = 0;
		for(; s < inst->size() && s+j < inst->size(); s++) 
		{
			int t = s + j;
//			cerr << "(s, t) : " << s << " " << t << endl;
			fvs.setDemisionVal(fvs_dim, s, t, 0);
			FeatureVec &prodFV_st = fvs.getElement(fvs_dim, fvs_pos);
			double prodProb_st = probs.getElement(fvs_pos);
			++fvs_pos;
			FeatureVec &prodFV_ts = fvs.getElement(fvs_pos);
			double prodProb_ts = probs.getElement(fvs_pos);

			static_types.setDemisionVal(type_dim, s, t);
			int type1 = isLabeled ? static_types.getElement(type_dim, type_pos) : -1;
			static_types.setDemisionVal(type_dim, t, s);
			int type2 = isLabeled ? static_types.getElement(type_dim, type_pos) : -1;

			FeatureVec tmp_feature_vec;

			nt_fvs.setDemisionVal(nt_dim, s, type1, 0, 1);
			FeatureVec &nt_fv_s_01 = isLabeled ? nt_fvs.getElement(nt_dim, nt_pos) : tmp_feature_vec;
			double nt_prob_s_01 = isLabeled ? nt_probs.getElement(nt_pos) : 0.0;

			nt_fvs.setDemisionVal(nt_dim, s, type2, 1, 0);
			FeatureVec &nt_fv_s_10 = isLabeled ? nt_fvs.getElement(nt_dim, nt_pos) : tmp_feature_vec;
			double nt_prob_s_10 = isLabeled ? nt_probs.getElement(nt_pos) : 0.0;

			nt_fvs.setDemisionVal(nt_dim, t, type1, 0, 0);
			FeatureVec &nt_fv_t_00 = isLabeled ? nt_fvs.getElement(nt_dim, nt_pos) : tmp_feature_vec;
			double nt_prob_t_00 = isLabeled ? nt_probs.getElement(nt_pos) : 0.0;

			nt_fvs.setDemisionVal(nt_dim, t, type2, 1, 1);
			FeatureVec &nt_fv_t_11 = isLabeled ? nt_fvs.getElement(nt_dim, nt_pos) : tmp_feature_vec;
			double nt_prob_t_11 = isLabeled ? nt_probs.getElement(nt_pos) : 0.0;

			double prodProb = 0.0;
			if(true)
			{
				{	// case when r == s
					chart.setDemisionVal(chart_dim, s, s, 0, 0, 0);		// s->s comp
					chart.setDemisionVal(chart_dim2, s+1, t, 1, 0, 0);	// s+1<-t comp
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);

					FeatureVec prodFV_sst;
					fvs_trips.setDemisionVal(trip_dim, s, s, t);
					fvs_trips.getElement(trip_dim, trip_pos);
					fvs_sibs.setDemisionVal(sib_dim, s, t, 0);
					fvs_sibs.getElement(sib_dim, sib_pos);
					prodFV_sst.add( &fvs_trips.getElement(trip_pos) );
					prodFV_sst.add( &fvs_sibs.getElement(sib_pos) );
					double prodProb_sst = probs_trips.getElement(trip_pos) + probs_sibs.getElement(sib_pos);

					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					for(int k = 0; k < pairs.size(); k++)
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;
						// create sibling pair
						// create parent pair: s->t and s->(start,t)
						bc += prodProb_st + prodProb_sst;

						FeatureVec fv_fin = prodFV_sst;
						fv_fin.add(&prodFV_st);

						if(isLabeled) {
							bc += nt_prob_s_01 + nt_prob_t_00;
							fv_fin.add(&nt_fv_s_01).add(&nt_fv_t_00);
						}				
						add(s,s,t,type1,0,1,bc,fv_fin, &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					}
				}
				{	// case when r == t				
					chart.setDemisionVal(chart_dim, s, t-1, 0, 0, 0);
					chart.setDemisionVal(chart_dim2, t, t, 1, 0, 0);
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);

					FeatureVec prodFV_stt;
					fvs_trips.setDemisionVal(trip_dim, t, t, s);
					fvs_trips.getElement(trip_dim, trip_pos);
					fvs_sibs.setDemisionVal(sib_dim, t, s, 0);
					fvs_sibs.getElement(sib_dim, sib_pos);
					prodFV_stt.add( &fvs_trips.getElement(trip_pos) );
					prodFV_stt.add( &fvs_sibs.getElement(sib_pos) );
					double prodProb_stt = probs_trips.getElement(trip_pos) + probs_sibs.getElement(sib_pos);

					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					int k = 0;
					for(; k < pairs.size(); k++) 
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;
						// create sibling pair
						// create parent pair: t->s and t->(start,s)
						bc += prodProb_ts + prodProb_stt;

						FeatureVec fv_fin = prodFV_stt;
						fv_fin.add(&prodFV_ts);

						if(isLabeled) {
							bc += nt_prob_t_11 + nt_prob_s_10;
							fv_fin.add(&nt_fv_t_11).add(&nt_fv_s_10);
						}				
						add(s,t,t,type2,1,1,bc,fv_fin,&chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					}
				} // end case when r == t	
			} // end if(true)

			int r;
			for(r = s; r < t; r++) 	// First case - create sibling
			{
				chart.setDemisionVal(chart_dim, s, r, 0, 0, 0);		// -> complete
				chart.setDemisionVal(chart_dim2, r+1, t, 1, 0, 0);	// <- complete
				chart.getElement(chart_dim, chart_pos);
				chart.getElement(chart_dim2, chart_pos2);

				vector< pair<int, int> > pairs;
				getKBestPairs(chart_pos, chart_pos2, pairs);
				int k = 0;
				for(; k < pairs.size(); k++)
				{
					if(pairs[k].first == -1 || pairs[k].second == -1) break;
					int comp1 = pairs[k].first;
					int comp2 = pairs[k].second;
					double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;
					// create sibling pair
					// create parent pair: t->s and t->(start,s)
					add(s,r,t,-1,0,2,bc, FeatureVec(), &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					add(s,r,t,-1,1,2,bc, FeatureVec(), &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
				}
			} // end for(r = s; r < t; r++)

			for(r = s+1; r < t; r++) {
				if (s != 0) // special case: NO multi-ROOT in one sent.
				{ // s -> (r,t)
					chart.setDemisionVal(chart_dim, s, r, 0, 1, 0);		// s->r incomplete
					chart.setDemisionVal(chart_dim2, r, t, 0, 2, 0);	// r-t-sib
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);
					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					int k = 0;
					for(; k < pairs.size(); k++)
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;

						fvs_trips.setDemisionVal(trip_dim, s, r, t);
						fvs_trips.getElement(trip_dim, trip_pos);
						fvs_sibs.setDemisionVal(sib_dim, r, t, 1);
						fvs_sibs.getElement(sib_dim, sib_pos);
						bc += prodProb_st + probs_trips.getElement(trip_pos) + probs_sibs.getElement(sib_pos);

						FeatureVec fv_fin;
						fv_fin.add(&prodFV_st);
						fv_fin.add( &fvs_trips.getElement(trip_pos) );
						fv_fin.add( &fvs_sibs.getElement(sib_pos) );					
						if(isLabeled) {
							bc += nt_prob_s_01 + nt_prob_t_00;
							fv_fin.add(&nt_fv_s_01).add(&nt_fv_t_00);
						}

						add(s,r,t,type1,0,1,bc,fv_fin, &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					}	
				} // end // s -> (r,t)
				{ // t -> (r,s)
					chart.setDemisionVal(chart_dim, s, r, 1, 2, 0);		// s-r-sib 
					chart.setDemisionVal(chart_dim2, r, t, 1, 1, 0);	// r<-t incomplete
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);
					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					int k = 0;
					for(; k < pairs.size(); k++)
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						ParseForestItem &item1 = chart.getElement(chart_pos+comp1);
						ParseForestItem &item2 = chart.getElement(chart_pos2+comp2);
						double prob1 = item1.prob;
						double prob2 = item2.prob;
						double bc = prob1;
/*						if (s == 2 && r == 3 && t == 4) {
							cerr << "bc = prob1: " << bc << endl;
						}
*/						bc += prob2;
/*						if (s == 2 && r == 3 && t == 4) {
							cerr << "bc += prob2: " << bc << endl;
						}
*/
						fvs_trips.setDemisionVal(trip_dim, t, r, s);
						fvs_trips.getElement(trip_dim, trip_pos);
						fvs_sibs.setDemisionVal(sib_dim, r, s, 1);
						fvs_sibs.getElement(sib_dim, sib_pos);
						double trip_prob = probs_trips.getElement(trip_pos);
						double sib_prob = probs_sibs.getElement(sib_pos);
						bc += prodProb_ts;
/*						if (s == 2 && r == 3 && t == 4) {
							cerr << bc << endl;
						}
*/						bc += trip_prob;
/*						if (s == 2 && r == 3 && t == 4) {
							cerr << bc << endl;
						}
*/						bc += sib_prob;
/*						if (s == 2 && r == 3 && t == 4) {
							cerr << bc << endl;
						}
*/						FeatureVec fv_fin;
						fv_fin.add(&prodFV_ts);
						fv_fin.add( &fvs_trips.getElement(trip_pos) );
						fv_fin.add( &fvs_sibs.getElement(sib_pos) );					
						if(isLabeled) {
							bc += nt_prob_t_11+nt_prob_s_10;
							fv_fin.add(&nt_fv_t_11).add(&nt_fv_s_10);
						}

/*						if (s == 2 && r == 3 && t == 4) {
							cerr << "good" << endl;
							cerr << item1.prob << endl;
							cerr << item2.prob << endl;
							cerr << prodProb_ts << endl;	// 0.151
							cerr << trip_prob << endl;
							cerr << sib_prob << endl;
							cerr << nt_prob_t_11 << endl;
							cerr << nt_prob_s_10 << endl;
							cerr << bc << endl;			
						}			    
*/
						add(s,r,t,type2,1,1,bc,fv_fin, &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					}	
				} // end // t -> (r,s)
			} // end for(r = s+1; r < t; r++)

			// Finish off pieces incom + comp -> comp
			for(r = s; r <= t; r++) 
			{
				if(r != s) 
				{
					chart.setDemisionVal(chart_dim, s, r, 0, 1, 0);		// s->r incomplete 
					chart.setDemisionVal(chart_dim2, r, t, 0, 0, 0);	// r->t complete
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);
					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					int k = 0;
					for(; k < pairs.size(); k++)
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;
						add(s,r,t,-1,0,0,bc, FeatureVec(), &chart.getElement(chart_pos+comp1), &chart.getElement(chart_pos2+comp2));
					}
				} // end if(r != s)

				if(r != t) 
				{
					chart.setDemisionVal(chart_dim, s, r, 1, 0, 0);		// s<-r complete 
					chart.setDemisionVal(chart_dim2, r, t, 1, 1, 0);	// r<-t incomplete
					chart.getElement(chart_dim, chart_pos);
					chart.getElement(chart_dim2, chart_pos2);
					vector< pair<int, int> > pairs;
					getKBestPairs(chart_pos, chart_pos2, pairs);
					int k = 0;
					for(; k < pairs.size(); k++)
					{
						if(pairs[k].first == -1 || pairs[k].second == -1) break;
						int comp1 = pairs[k].first;
						int comp2 = pairs[k].second;
						double bc = chart.getElement(chart_pos+comp1).prob + chart.getElement(chart_pos2+comp2).prob;
						add(s,r,t,-1,1,0,bc, FeatureVec(), &chart.getElement(chart_pos+comp1),  &chart.getElement(chart_pos2+comp2));
					}
				} // end if(r != t)
			} // end for(r = s; r <= t; r++) {			
		} // end for(; s < inst->size() && s+j < inst->size(); s++)
	} // end for(; j < inst->size(); j++)

/*	for (j = 1; j < inst->size(); ++j) {
		if (j >=5) break;

		int s = 0;
		for (; s < inst->size() && s + j < inst->size(); ++s) {
			int t = s + j;
			int k = 0;
			for (; k < K; ++k) {
				chart.setDemisionVal(chart_dim, s, t, 0, 0, k);
				if (chart.getElement(chart_dim, chart_pos).m_isInit && chart.getElement(chart_dim, chart_pos).prob > DOUBLE_NEGATIVE_INFINITY + EPS) {
					cerr << "[" << s << " " << t << " 0 0 " << k <<"] " << chart.getElement(chart_dim, chart_pos).r 
						<< " " << chart.getElement(chart_pos).prob << endl;
				}
				chart.setDemisionVal(chart_dim, s, t, 0, 1, k);
				if (chart.getElement(chart_dim, chart_pos).m_isInit && chart.getElement(chart_dim, chart_pos).prob > DOUBLE_NEGATIVE_INFINITY + EPS) {
					cerr << "[" << s << " " << t << " 0 1 " << k <<"] " << chart.getElement(chart_dim, chart_pos).r
						<< " " << chart.getElement(chart_pos).prob << endl;
				}
				chart.setDemisionVal(chart_dim, s, t, 0, 2, k);
				if (chart.getElement(chart_dim, chart_pos).m_isInit && chart.getElement(chart_dim, chart_pos).prob > DOUBLE_NEGATIVE_INFINITY + EPS) {
					cerr << "[" << s << " " << t << " 0 2 " << k <<"] " << chart.getElement(chart_dim, chart_pos).r
						<< " " << chart.getElement(chart_pos).prob << endl;
				}
				chart.setDemisionVal(chart_dim, s, t, 1, 0, k);
				if (chart.getElement(chart_dim, chart_pos).m_isInit && chart.getElement(chart_dim, chart_pos).prob > DOUBLE_NEGATIVE_INFINITY + EPS) {
					cerr << "[" << s << " " << t << " 1 0 " << k <<"] " << chart.getElement(chart_dim, chart_pos).r
						<< " " << chart.getElement(chart_pos).prob << endl;
				}
				chart.setDemisionVal(chart_dim, s, t, 1, 1, k);
				if (chart.getElement(chart_dim, chart_pos).m_isInit && chart.getElement(chart_dim, chart_pos).prob > DOUBLE_NEGATIVE_INFINITY + EPS) {
					cerr << "[" << s << " " << t << " 1 1 " << k <<"] " << chart.getElement(chart_dim, chart_pos).r
						<< " " << chart.getElement(chart_pos).prob << endl;
				}
			}
		}
	} // end for (j = 1; j < inst->size(); ++j) {
*/
}


void KBestParseForest2O::getDepString(const ParseForestItem &pfi, string &strDep)
{
	if (!pfi.m_isInit) {
		cerr << "KBestParseForest::getDepString() : pfi is not initialized" << endl;
		strDep = "";
		return;
	}

	if(!pfi.left) {
		strDep = "";
		return;
	}

	string left;
	getDepString(*(pfi.left), left);
	string right;
	getDepString(*(pfi.right), right);
	string left_right = left + " " + right;
	KBestParseForest::trim(left_right);

	if(pfi.dir == 0 && pfi.comp == 1) {
		ostringstream out;
		out << left_right << " " << pfi.s << "|" << pfi.t << ":" << pfi.type;
		strDep = out.str();
		trim(strDep);
//		return ((getDepString(pfi.left)+" "+getDepString(pfi.right)).trim()+" "+pfi.s+"|"+pfi.t+":"+pfi.type).trim();
	}
	else if(pfi.dir == 1 && pfi.comp == 1) {
		ostringstream out;
		out << pfi.t << "|" << pfi.s << ":" << pfi.type << " " << left_right;
		strDep = out.str();
		trim(strDep);
//		return (pfi.t+"|"+pfi.s+":"+pfi.type+" "+(getDepString(pfi.left)+" "+getDepString(pfi.right)).trim()).trim();
	}
	else {
		strDep = left_right;
	}
}


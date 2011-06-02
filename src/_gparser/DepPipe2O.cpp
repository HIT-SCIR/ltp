#include "DepPipe2O.h"

DepPipe2O::DepPipe2O(const ParserOptions &_options) : DepPipe(_options)
{
}

DepPipe2O::~DepPipe2O(void)
{
}

int DepPipe2O::readInstance(FILE *featFile, int length,
				 MultiArray<FeatureVec> &fvs,
				 MultiArray<double> &probs,
				 MultiArray<FeatureVec> &fvs_trips,
				 MultiArray<double> &probs_trips,
				 MultiArray<FeatureVec> &fvs_sibs,
				 MultiArray<double> &probs_sibs,
				 MultiArray<FeatureVec> &nt_fvs,
				 MultiArray<double> &nt_probs,
				 FeatureVec &fv,
				 string &actParseTree,
				 const Parameter &params)
{
//	cerr << "\n+++++-----+++++" << endl;
//	cerr << "read instance" << endl;
//	cerr << endl;
	vector<unsigned int> fvs_dim;
	unsigned int fvs_pos;

	MyVector<int> vecKeys;

	// Get production crap.
	int w1;
	for(w1 = 0; w1 < length; w1++) {
		int w2 = w1 + 1;
		if (w2 >= length) continue;
		fvs.setDemisionVal(fvs_dim, w1, w2, 0);
		fvs.getElement(fvs_dim, fvs_pos);
		for(; w2 < length; w2++) {
			for(int ph = 0; ph < 2; ph++) {
				FeatureVec &prodFV = fvs.getElement(fvs_pos);
				::readObject(featFile, vecKeys);
				prodFV.setKeys(vecKeys);
				probs.getElement(fvs_pos) = params.getScore(prodFV);
				//cerr << "(" << vecKeys.size() << ",";
				//cerr << probs.getElement(fvs_pos) << ")\t";
				++fvs_pos;
			}
		}
		// cerr << endl;
	}
	int last;
	::readObject(featFile, last);
	if(last != -1) { cerr << "DepPipe2O::readInstance() Error reading file. -1" << endl; return -1; }

	// cerr << endl;
	if(options.m_isLabeled) {
		vector<unsigned int> nt_dim(4);
		unsigned int nt_pos;

		nt_fvs.setDemisionVal(nt_dim, 0, 0, 0, 0);
		nt_fvs.getElement(nt_dim, nt_pos);
		int w1;
		for(w1 = 0; w1 < length; w1++) {
			int t;
			for(t = 0; t < m_vecTypes.size(); t++) {
				const string &type = m_vecTypes[t];
				int ph;
				for(ph = 0; ph < 2; ph++) {						
					int ch;
					for(ch = 0; ch < 2; ch++) {
						FeatureVec &prodFV = nt_fvs.getElement(nt_pos);
						::readObject(featFile, vecKeys);
						prodFV.setKeys(vecKeys);
						// cerr << vecKeys.size() << " ";
						nt_probs.getElement(nt_pos) = params.getScore(prodFV);
						++nt_pos;
					}
				}
			}
			// cerr << endl;
		}
		::readObject(featFile, last);
		if(last != -2) { cerr << "DepPipe2O::readInstance() Error reading file.-2" << endl; return -1; }
	}

	int w2, w3, wh;
	for(w1 = 0; w1 < length; w1++) {
		for(w2 = w1; w2 < length; w2++) {
			for(w3 = w2+1; w3 < length; w3++) {
				fvs_trips.setDemisionVal(fvs_dim, w1, w2, w3);
				FeatureVec &prodFV = fvs_trips.getElement(fvs_dim, fvs_pos);
				::readObject(featFile, vecKeys);
				prodFV.setKeys(vecKeys);
				probs_trips.getElement(fvs_pos) = params.getScore(prodFV);
/*				if (w3 <= w2 + 3 && !vecKeys.empty()) {				
					cerr << vecKeys.size() << " ";
					cerr << probs_trips.getElement(fvs_pos) << endl;
				}
*/			}
		}
		for(w2 = w1; w2 >= 0; w2--) {
			for(w3 = w2-1; w3 >= 0; w3--) {
				fvs_trips.setDemisionVal(fvs_dim, w1, w2, w3);
				FeatureVec &prodFV = fvs_trips.getElement(fvs_dim, fvs_pos);
				::readObject(featFile, vecKeys);
				prodFV.setKeys(vecKeys);
				probs_trips.getElement(fvs_pos) = params.getScore(prodFV);
/*				if (w3 >= w2 - 3 && !vecKeys.empty()) {
					cerr << vecKeys.size() << " ";
					cerr << probs_trips.getElement(fvs_pos) << endl;
				}
*/			}
		}
	}
	::readObject(featFile, last);
	if(last != -13) { cerr << "DepPipe2O::readInstance() Error reading file.-13 vs. " << last << endl; return -1; }

	for(w1 = 0; w1 < length; w1++) {
		for(w2 = 0; w2 < length; w2++) {
			for(wh = 0; wh < 2; wh++) {
				if(w1 != w2) {
					fvs_sibs.setDemisionVal(fvs_dim, w1, w2, wh);
					FeatureVec &prodFV = fvs_sibs.getElement(fvs_dim, fvs_pos);
					::readObject(featFile, vecKeys);
					prodFV.setKeys(vecKeys);
					probs_sibs.getElement(fvs_pos) = params.getScore(prodFV);
/*					if ((w1 < w2 + 5 || w1 > w2 - 5) && !vecKeys.empty()) {
						cerr << vecKeys.size() << " ";
						cerr << probs_sibs.getElement(fvs_pos) << endl;
					}
*/				}
			}
		}
	}

	::readObject(featFile, last);
	if(last != -14) { cerr << "DepPipe2O::readInstance() Error reading file.-14 vs. " << last << endl; return -1; }

	::readObject(featFile, vecKeys);
	fv.setKeys(vecKeys);
	::readObject(featFile, last);
	if(last != -3) { cerr << "DepPipe2O::readInstance() Error reading file.-3" << endl; return -1; }

	MyVector<char> my_str;
	::readObject(featFile, my_str);
	actParseTree = my_str.begin();
	// cerr << actParseTree << endl;
	// cerr << actParseTree.size() << endl;
	::readObject(featFile, last);
	if(last != -4) { cerr << "DepPipe2O::readInstance() Error reading file.-4 vs. " << last << endl; return -1; }
	return 0;
}

void DepPipe2O::fillFeatureVectors(DepInstance *instance,
						MultiArray<FeatureVec> &fvs,
						MultiArray<double> &probs,
						MultiArray<FeatureVec> &fvs_trips,
						MultiArray<double> &probs_trips,
						MultiArray<FeatureVec> &fvs_sibs,
						MultiArray<double> &probs_sibs,
						MultiArray<FeatureVec> &nt_fvs,
						MultiArray<double> &nt_probs,
						const Parameter &params)
{
	DepPipe::fillFeatureVectors(instance, fvs, probs, fvs_trips, probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, params);

	int instanceLength = instance->size();
	vector<unsigned int> fvs_dim(3);

	fvs_trips.setDemisionVal(fvs_dim, instanceLength, instanceLength, instanceLength);
	fvs_trips.resize(fvs_dim);
	probs_trips.resize(fvs_dim);

	fvs_sibs.setDemisionVal(fvs_dim, instanceLength, instanceLength, 2);
	fvs_sibs.resize(fvs_dim);
	probs_sibs.resize(fvs_dim);

	unsigned int fvs_pos;
	int w1, w2, w3, wh;
	for(w1 = 0; w1 < instanceLength; w1++) {
		for(w2 = w1; w2 < instanceLength; w2++) {
			for(w3 = w2+1; w3 < instanceLength; w3++) {
				fvs_trips.setDemisionVal(fvs_dim, w1, w2, w3);
				fvs_trips.getElement(fvs_dim, fvs_pos);

				addTripFeature(instance,w1,w2,w3, fvs_trips.getElement(fvs_pos));
				probs_trips.getElement(fvs_pos) = params.getScore( fvs_trips.getElement(fvs_pos) );
			}
		}
		for(w2 = w1; w2 >= 0; w2--) {
			for(w3 = w2-1; w3 >= 0; w3--) {
				fvs_trips.setDemisionVal(fvs_dim, w1, w2, w3);
				fvs_trips.getElement(fvs_dim, fvs_pos);

				addTripFeature(instance,w1,w2,w3, fvs_trips.getElement(fvs_pos));
				probs_trips.getElement(fvs_pos) = params.getScore( fvs_trips.getElement(fvs_pos) );
			}
		}
	}

	for(w1 = 0; w1 < instanceLength; w1++) {
		for(w2 = 0; w2 < instanceLength; w2++) {
			for(wh = 0; wh < 2; wh++) {
				if(w1 != w2) {
					fvs_sibs.setDemisionVal(fvs_dim, w1, w2, wh);
					fvs_sibs.getElement(fvs_dim, fvs_pos);

					addSibFeature(instance,w1,w2, wh == 0, fvs_sibs.getElement(fvs_pos));
					probs_sibs.getElement(fvs_pos) = params.getScore( fvs_sibs.getElement(fvs_pos) );
				}
			}
		}
	}
}

void DepPipe2O::addExtendedFeature(DepInstance *pInstance, FeatureVec &fv)
{
	const int instanceLength = pInstance->size();
	const vector<int> &heads = pInstance->heads;

	// find all trip features
	for(int i = 0; i < instanceLength; i++) {
		if(heads[i] == -1 && i != 0) continue;
		// right children
		int prev = i;
		for(int j = i+1; j < instanceLength; j++) {
			if(heads[j] == i) {
				addTripFeature(pInstance, i, prev, j, fv);
				addSibFeature(pInstance, prev, j, prev==i, fv);
				prev = j;
			}
		}

		prev = i;
		for(int j = i-1; j >= 0; j--) {
			if(heads[j] == i) {
				addTripFeature(pInstance,i,prev,j,fv);
				addSibFeature(pInstance,prev,j,prev==i,fv);
				prev = j;
			}
		}
	}
}
void DepPipe2O::writeExtendedFeatures(DepInstance *pInstance, FILE *featFile)
{
	const int pInstanceLength = pInstance->size();
	int w1, w2, w3, wh;
	for(w1 = 0; w1 < pInstanceLength; w1++) {
		for(w2 = w1; w2 < pInstanceLength; w2++) {
			for(w3 = w2+1; w3 < pInstanceLength; w3++) {
				FeatureVec prodFV;
				addTripFeature(pInstance,w1,w2,w3,prodFV);
				vector<int> vecKeys;
				prodFV.getKeys(vecKeys);
				// cerr << vecKeys.size() << " ";
				::writeObject(featFile, vecKeys);
			}
		}
		for(w2 = w1; w2 >= 0; w2--) {
			for(w3 = w2-1; w3 >= 0; w3--) {
				FeatureVec prodFV;
				addTripFeature(pInstance,w1,w2,w3,prodFV);
				vector<int> vecKeys;
				prodFV.getKeys(vecKeys);
				// cerr << vecKeys.size() << " ";
				::writeObject(featFile, vecKeys);
			}
		}
	}

	::writeObject(featFile, (int)-13);

	for(w1 = 0; w1 < pInstanceLength; w1++) {
		for(w2 = 0; w2 < pInstanceLength; w2++) {
			for(wh = 0; wh < 2; wh++) {
				if(w1 != w2) {
					FeatureVec prodFV;
					addSibFeature(pInstance,w1,w2,wh == 0,prodFV);
					vector<int> vecKeys;
					prodFV.getKeys(vecKeys);
					// cerr << vecKeys.size() << " ";
					::writeObject(featFile, vecKeys);
				}
			}
		}
	}

	::writeObject(featFile, (int)-14);
}

void DepPipe2O::addSibFeature(DepInstance *pInstance, int ch1, int ch2, bool isST, FeatureVec &fv)
{
	const vector<string> &forms = pInstance->forms;
//	const vector<string> &lemmas = pInstance->lemmas;
	const vector<string> &cpostags = pInstance->cpostags;
//	const vector<string> &postags = pInstance->postags;

	// ch1 is always the closes to par
	string dir = ch1 > ch2 ? "RA" : "LA";

	string ch1_pos = isST ? "STPOS" : cpostags[ch1];
	string ch2_pos = cpostags[ch2];
	string ch1_word = isST ? "STWRD" : forms[ch1];
	string ch2_word = forms[ch2];

	add("CH_PAIR="+ch1_pos+"_"+ch2_pos+"_"+dir,1.0,fv);		// pos1 pos2 dir
	add("CH_WPAIR="+ch1_word+"_"+ch2_word+"_"+dir,1.0,fv);	// wrd1 wrd2 dir
	add("CH_WPAIRA="+ch1_word+"_"+ch2_pos+"_"+dir,1.0,fv);	// wrd1 pos2 dir
	add("CH_WPAIRB="+ch1_pos+"_"+ch2_word+"_"+dir,1.0,fv);	// pos1 wrd2 dir
	add("ACH_PAIR="+ch1_pos+"_"+ch2_pos,1.0,fv);			// pos1 pos2
	add("ACH_WPAIR="+ch1_word+"_"+ch2_word,1.0,fv);			// wrd1 wrd2
	add("ACH_WPAIRA="+ch1_word+"_"+ch2_pos,1.0,fv);			// wrd1 pos2
	add("ACH_WPAIRB="+ch1_pos+"_"+ch2_word,1.0,fv);			// pos1 wrd2

	int dist = max(ch1,ch2) - min(ch1,ch2);
	string distBool = "0";
	if(dist > 1)
		distBool = "1";
	if(dist > 2)
		distBool = "2";
	if(dist > 3)
		distBool = "3";
	if(dist > 4)
		distBool = "4";
	if(dist > 5)
		distBool = "5";
	if(dist > 10)
		distBool = "10";

	add("SIB_PAIR_DIST="+distBool+"_"+dir,1.0,fv);			// dist dir
	add("ASIB_PAIR_DIST="+distBool,1.0,fv);					// dist
	add("CH_PAIR_DIST="+ch1_pos+"_"+ch2_pos+"_"+distBool+"_"+dir,1.0,fv);	// pos1 pos2 dist dir
	add("ACH_PAIR_DIST="+ch1_pos+"_"+ch2_pos+"_"+distBool,1.0,fv);			// pos1 pos2 dist
}

void DepPipe2O::addTripFeature(DepInstance *pInstance, int par, int ch1, int ch2, FeatureVec &fv)
{
	const vector<string> &cpostags = pInstance->cpostags;

	// ch1 is always the closest to par
	string dir = par > ch2 ? "RA" : "LA";

	string par_pos = cpostags[par];
	string ch1_pos = ch1 == par ? "STPOS" : cpostags[ch1];
	string ch2_pos = cpostags[ch2];

	string pTrip = par_pos + "_" + ch1_pos + "_" + ch2_pos;	
	add("POS_TRIP=" + pTrip + "_" + dir, 1.0, fv);	// pos_f pos1 pos2 dir
	add("APOS_TRIP=" + pTrip, 1.0, fv);				// pos_f pos1 pos2
}


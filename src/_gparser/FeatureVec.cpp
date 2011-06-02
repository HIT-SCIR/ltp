#include "FeatureVec.h"

double FeatureVec::getScore(const vector<double> &parameters, bool negate) const 
{
	double score = 0.0;
	int i = 0;
	for (; i < subfv.size(); ++i) {
		bool neg = negate ? !subfv[i].negate : subfv[i].negate;
		score += subfv[i].pFv->getScore(parameters, neg);
	}

	vector<Feature>::const_iterator it = m_fv.begin();		
	int neg = negate ? -1 : 1;
	while (it != m_fv.end()) {
		if (it->index < 0 || it->index >= parameters.size()) {
			cerr << "index err: " << it->index << endl;
			continue;
		}
		score += neg * parameters[it->index] * it->value;
		++it;
	}
	return score;
}

void FeatureVec::addKeys2List(vector<int> &vecKeys) const 
{
	int i = 0;
	for (; i < subfv.size(); ++i) {
		subfv[i].pFv->addKeys2List(vecKeys);
	}
	vector<Feature>::const_iterator it = m_fv.begin();
	while (it != m_fv.end()) {
		vecKeys.push_back(it->index);
		++it;
	}
}

void FeatureVec::addKeys2Set(set<int> &setKeys) const 
{
	int i = 0;
	for (; i < subfv.size(); ++i) {
		subfv[i].pFv->addKeys2Set(setKeys);
	}
	vector<Feature>::const_iterator it = m_fv.begin();
	while (it != m_fv.end()) {
		setKeys.insert(it->index);
		++it;
	}
}

void FeatureVec::update(vector<double> &parameters, vector<double> &total, 
			double alpha_k, double upd, bool negate) const
{
	int i = 0;
	for (; i < subfv.size(); ++i) {
		bool neg = negate ? !subfv[i].negate : subfv[i].negate;
		subfv[i].pFv->update(parameters, total, alpha_k, upd, neg);
	}

	vector<Feature>::const_iterator it = m_fv.begin();		
	int neg = negate ? -1 : 1;
	int cnt = 0;
//	cerr << "\n[" << neg << "]" << endl;
	while (it != m_fv.end()) {
		if (it->index < 0 || it->index >= parameters.size()) {
			cerr << "index err: " << it->index << endl;
			continue;
		}
/*		if (++cnt % 15 == 0) {
			cerr << endl;
			//break;
		}
		cerr << "(" << it->index << " " << parameters[it->index] << " | " << alpha_k * it->value << ")\t";
*/		parameters[it->index] += neg * alpha_k * it->value;
		total[it->index] += neg * upd * alpha_k * it->value;
		++it;
	}
//	cerr << "\n***\n" << endl;;
}

double FeatureVec::dotProduct(const FeatureVec &fv1,const FeatureVec &fv2)
{
	map<int, double> map1;
	fv1.addFeaturesToMap(map1, false);
	map<int, double> map2;
	fv2.addFeaturesToMap(map2, false);

	double result = 0.0;
	map<int, double>::const_iterator it = map1.begin();
	while (it != map1.end()) {
		map<int, double>::const_iterator it2 = map2.find(it->first);
		if (it2 != map2.end()) {
			result += it->second * it2->second;
		}
		++it;
	}
	return result;
}

void FeatureVec::addFeaturesToMap(map<int, double> &mapFv, bool negate) const {
	int i = 0;
	for (; i < subfv.size(); ++i) {
		bool neg = negate ? !subfv[i].negate : subfv[i].negate;
		subfv[i].pFv->addFeaturesToMap(mapFv, neg);
	}

	vector<Feature>::const_iterator it = m_fv.begin();
	int neg = negate ? -1 : 1;
	while (it != m_fv.end()) {
		map<int, double>::iterator it_map = mapFv.find(it->index);
		if (it_map == mapFv.end()) {
			mapFv[it->index] = neg * it->value;
		} else {
			it_map->second += neg * it->value;
		}
		++it;
	}
}


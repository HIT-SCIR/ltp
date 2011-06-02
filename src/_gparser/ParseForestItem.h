#ifndef _PARSE_FOREST_ITEM_
#define _PARSE_FOREST_ITEM_

#pragma once

#include "FeatureVec.h"

class ParseForestItem
{
public:
	int s, r, t; 
	int dir; // direction ? 0 : right-arc; 1 : left-arc
	int comp; // is complete ? 0 : yes; 1 : no
	int length;
	int type; // label type
	double prob;
	FeatureVec fv;
	ParseForestItem *left, *right; // left and right sub-span
	bool m_isInit;

public:
	ParseForestItem(int _s, int _r, int _t, int _type,
		int _dir, int _comp, 
		double _prob, const FeatureVec &_fv,
		ParseForestItem *_left, ParseForestItem *_right) :
			s(_s), r(_r), t(_t), dir(_dir), comp(_comp), type(_type), length(6),
			prob(_prob), fv(_fv), left(_left), right(_right), m_isInit(true) {}
	ParseForestItem(int _s, int _type, int _dir, 
		double _prob, const FeatureVec &_fv):
			s(_s), r(-1), t(-1), dir(_dir), comp(-1), type(_type), length(2),
			prob(_prob), fv(_fv), left(0), right(0), m_isInit(true) {}
	ParseForestItem() : m_isInit(false) {}
	ParseForestItem(const ParseForestItem &other) {
		other.copyValuesTo(*this);
	}
	~ParseForestItem(void) {}

	void copyValuesTo(ParseForestItem &p) const 
	{
		p.s = s;
		p.r = r;
		p.t = t;
		p.dir = dir;
		p.comp = comp;
		p.prob = prob;
		p.fv = fv;
		p.length = length;
		p.left = left;
		p.right = right;
		p.type = type;
		p.m_isInit = m_isInit;
	}

	// way forest works, only have to check rule and indeces
	// for equality.
	bool equals(const ParseForestItem &p) {
		return (m_isInit && p.m_isInit && s == p.s && t == p.t && r == p.r
			&& dir == p.dir && comp == p.comp
			&& type == p.type);
	}

	bool isPre() { return length == 2; }
};

#endif

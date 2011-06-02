#ifndef _MULTI_ARRAY_
#define _MULTI_ARRAY_
#pragma once

#include <vector>
#include <exception>
#include <iostream>
using namespace std;

template <typename Ty>
class MultiArray {
private:
	vector<unsigned int> m_vecDimension;
	vector<unsigned int> m_vecDimensionElementNum;
	vector<Ty> m_data; // The last one is not used!
public:
	int resize(const vector<unsigned int> &vecDimension, const Ty &default_data) {
		if (vecDimension.empty()) {
			cerr << "dimension empty err" << endl;
			return -1;
		}

		m_vecDimensionElementNum.resize(vecDimension.size());
		unsigned int arr_size = 1;
		int i = vecDimension.size()-1;
		for (; i >= 0; --i) {
			if (vecDimension[i] == 0) {
				cerr << "dimension element num is 0" << endl;
				return -1;
			}
			m_vecDimensionElementNum[i] = arr_size;
			arr_size *= vecDimension[i];
		}
		m_vecDimension = vecDimension;
//		try {
			m_data.resize(0);
			m_data.resize(arr_size + 1, default_data);			
/*		} catch (const exception &e) {
			cerr << "MultiArray::vector::resize( " << arr_size + 1 << " ) exception" << endl;
			cerr << "element size: " << sizeof(Ty) << endl;
			cerr << e.what() << endl;
			return -1;
		}
*/
		return 0;
	}

	int resize(const vector<unsigned int> &vecDimension) {
		if (vecDimension.empty()) {
			cerr << "dimension empty err" << endl;
			return -1;
		}
		m_vecDimensionElementNum.resize(vecDimension.size());
		unsigned int arr_size = 1;
		int i = vecDimension.size()-1;
		for (; i >= 0; --i) {
			if (vecDimension[i] == 0) {
				cerr << "dimension element num is 0" << endl;
				return -1;
			}
			m_vecDimensionElementNum[i] = arr_size;
			arr_size *= vecDimension[i];
		}
		m_vecDimension = vecDimension;
//		try {
			m_data.resize(0);
			m_data.resize(arr_size + 1);			
/*		} catch (const exception &e) {
			cerr << "MultiArray::vector::resize( " << arr_size + 1 << " ) exception" << endl;
			cerr << "element size: " << sizeof(Ty) << endl;
			cerr << e.what() << endl;
			return -1;
		}
*/
		return 0;
	}


	unsigned int size() const {
		return m_data.size()-1;
	}

	unsigned int end_pos() const {
		return m_data.size();
	}

	unsigned int dimension() const {
		return m_vecDimension.size();
	}

	Ty &getElement(const vector<unsigned int> &vecDemension, unsigned int &pos) {
		pos = getPosition(vecDemension);
		return getElement(pos);
	}

	const Ty &getElement(const vector<unsigned int> &vecDemension, unsigned int &pos) const {
		pos = getPosition(vecDemension);
		return getElement(pos);
	}

	Ty &getElement(unsigned int pos) {
		if (pos >= end_pos()) {
			cerr << "position err: reach end: " << pos << endl;
			cerr << "total size: " << size() << endl;
			return m_data.back();
		}
		return m_data[pos];
	}

	const Ty &getElement(unsigned int pos) const {
		if (pos >= end_pos()) {
			cerr << "position err: reach end: " << pos << endl;
			cerr << "total size: " << size() << endl;
			return m_data.back();
		}
		return m_data[pos];
	}
	void setDemisionVal(	
		vector<unsigned int> &dim,
		unsigned int v0,
		unsigned int v1,
		unsigned int v2,
		unsigned int v3,
		unsigned int v4) const
	{
		dim.resize(5); dim[0] = v0; dim[1] = v1; dim[2] = v2; dim[3] = v3; dim[4] = v4;
	}
	void setDemisionVal(	
		vector<unsigned int> &dim,
		unsigned int v0,
		unsigned int v1, 
		unsigned int v2, 
		unsigned int v3) const
	{
		dim.resize(4); dim[0] = v0; dim[1] = v1; dim[2] = v2; dim[3] = v3;
	}
	void setDemisionVal(	
		vector<unsigned int> &dim,
		unsigned int v0,
		unsigned int v1, 
		unsigned int v2) const
	{
		dim.resize(3); dim[0] = v0; dim[1] = v1; dim[2] = v2;
	}
	void setDemisionVal(	
		vector<unsigned int> &dim,
		unsigned int v0,
		unsigned int v1) const
	{
		dim.resize(2); dim[0] = v0; dim[1] = v1;
	}
private:
	unsigned int getPosition(const vector<unsigned int> &vecDemension) const {
		if (vecDemension.size() != m_vecDimension.size()) {
			cerr << "dimension not equal, should be: " << m_vecDimension.size() << " not: " << vecDemension.size() << endl;
			return end_pos();
		}
		unsigned int pos = 0;
		int i = 0;
		for (; i < vecDemension.size(); ++i) {
			if (vecDemension[i] >= m_vecDimension[i]) {
				cerr << "dimension " << i << " value err: " << vecDemension[i] << endl;
				cerr << "dimension size is: " << m_vecDimension[i] << endl;
				return end_pos();
			}
			pos += vecDemension[i] * m_vecDimensionElementNum[i];
		}
		return pos;	
	}

};




#endif


#ifndef _MY_VECTOR_
#define _MY_VECTOR_
#pragma once

#include <exception>
#include <iostream>
using namespace std;

template <typename Ty>
class MyVector {
public:
	Ty *m_data; // The last one is not used!
public:
	int m_capacity;
	int m_size;
public:
	MyVector() : m_data(0), m_capacity(0), m_size(0) {
		
	}

	~MyVector() {
		if (m_data) {
			delete [] m_data;
		}
	}

	bool empty() {
		return m_size == 0;
	}
	int capacity() const {
		return m_capacity;
	}

	int size() const {
		return m_size;
	}

	void clear() {
		m_size = 0;
	}
	int resize(int _size) {
		if (_size < 0) {
			cerr << "MyVector::resize() err: new size is: " << _size << endl;
			return -1;
		}

		if (_size <= m_capacity) {
			m_size = _size;
		}
		else { // _size > m_capacity
			int new_capacity = 2 * _size;
			try {
				if (m_data) delete [] m_data;
				m_capacity = 0;
				m_size = 0;
				m_data = 0;
				m_data = new Ty[new_capacity + 1]; // The last one is not used!
			} catch (const exception &e) {
				cerr << "MyVector::resize( " << new_capacity + 1 << " ) exception" << endl;
				cerr << "element size: " << sizeof(Ty) << endl;
				cerr << e.what() << endl;
				return -1;
			}
			m_capacity = new_capacity;
			m_size = _size;
		}

		return 0;
	}

	Ty *begin() {
		return m_data;
	}

	const Ty *begin() const {
		return m_data;
	}

	Ty &operator[](int pos) {
		if (pos < 0 || pos >= size()) return m_data[size()];
		return m_data[pos];
	}

	const Ty &operator[](int pos) const {
		if (pos < 0 || pos >= size()) return m_data[size()];
		return m_data[pos];
	}
};




#endif


/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * modelfile.cpp  -  helper classes for loading and saving Maxent/RandomField
 * Model
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 28-May-2003
 * Last Change : 24-Apr-2004.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

// TODO: write a note on model file format

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include "modelfile.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/progress.hpp>
// #include "mmapfile.hpp"
// #include "display.hpp"
#ifdef HAVE_LIBZ
    #include <zlib.h>
#else
    // fallback to stdio
    //#warning zlib not found use stdio instead
    #include <cstdio>
    typedef FILE* gzFile;
    static inline gzFile gzopen(const char* fn, const char* mode) {
        return fopen(fn, mode);
    }
    static inline int gzclose(gzFile f) {
            return fclose(f);
    }
    static inline size_t gzread(gzFile f, void* buf, size_t len) {
            return fread(buf, len, 1u, f);
    }
    static inline size_t gzwrite(gzFile f, void* buf, size_t len) {
            return fwrite(buf,len, 1u, f);
    }
    static inline int gzseek(gzFile f, long offset, int pos) {
            return fseek(f, offset, pos);
    }
#endif

namespace maxent {
using namespace std;
using namespace boost;

// header can be: #txt or #bin
// TODO:write some CRC info
static char model_header[] = "#txt,randomfield";
static const int header_len = sizeof(model_header) - 1;

void check_modeltype(const string& model, bool& binary, bool& randomfield) {
    gzFile f = gzopen(model.c_str(), "rb");
    if (f == NULL)
        throw runtime_error("Unable to open model file to read");

    char buf[100];
    gzread(f, (void*)buf, header_len);
    buf[header_len] = '\0';
    string s = buf;

    if (s.find("txt") != s.npos) {
        binary = false;
    } else if (s.find("bin") != s.npos) {
        binary = true;
    } else {
        throw runtime_error("Unable to detect model file format");
    }

    if (s.find("randomfield") != s.npos) {
        randomfield = true;
    } else if (s.find("maxent") != s.npos) {
        randomfield = false;
    } else {
        throw runtime_error("Unable to detect model file format");
    }
    gzclose(f);
}

MaxentModelFile::MaxentModelFile(): m_n_theta(0) {}

shared_ptr<me::PredMapType> MaxentModelFile::pred_map() {
    if (!m_pred_map)
        throw runtime_error("No model loaded");
    return m_pred_map;
}

shared_ptr<me::OutcomeMapType> MaxentModelFile::outcome_map() {
    if (!m_outcome_map)
        throw runtime_error("No model loaded");
    return m_outcome_map;
}

void MaxentModelFile::params(shared_ptr<me::ParamsType>& params,
        size_t& n_theta, shared_array<double>& theta) {
    if (!m_params)
        throw runtime_error("No model loaded");
     params  = m_params;
     n_theta = m_n_theta;
     theta   = m_theta;
}

void MaxentModelFile::set_pred_map(shared_ptr<me::PredMapType> pred_map) {
    assert(pred_map);
    m_pred_map = pred_map;
}

void MaxentModelFile::set_outcome_map(
        shared_ptr<me::OutcomeMapType> outcome_map) {
    assert(outcome_map);
    m_outcome_map = outcome_map;
}

void MaxentModelFile::set_params(shared_ptr<me::ParamsType> params, 
        size_t n_theta, shared_array<double> theta){
    assert(params);
    m_params           = params;
    m_n_theta          = n_theta;
    m_theta            = theta;
}

void MaxentModelFile::load(const string& model) {
    bool binary;
    bool rf;
    check_modeltype(model, binary, rf);
    cout << "check model over! " << binary << endl;
    if (rf != false)
        throw runtime_error("Trying to load a previously saved RandomField modelin MaxentModelFile::load().");
    if (binary)
        load_model_bin(model);
    else
        load_model_txt(model);
}

void MaxentModelFile::load_model_txt(const string& model) {
    // cerr << "load model txt from " <<  model << endl;

    ifstream f(model.c_str());

    if (!f)
        throw runtime_error("fail to open model file");

    size_t count;
    string line;

    m_pred_map.reset(new me::PredMapType);
    m_outcome_map.reset(new me::OutcomeMapType);
    m_params.reset(new me::ParamsType);
    m_theta.reset(0);

    // skip header comments
    getline(f, line);
    while (line.empty() || line[0] == '#')
        getline(f, line);

    // read context predicates
    count = atoi(line.c_str());
    for (size_t i = 0; i < count; ++i) {
        getline(f, line);
        m_pred_map->add(line);
    }

    // read outcomes
    getline(f, line);
    count = atoi(line.c_str());
    for (size_t i = 0; i < count; ++i) {
        getline(f, line);
        m_outcome_map->add(line);
    }

    // read paramaters
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer_t;
    boost::char_separator<char> sep(" \t");
    tokenizer_t tokens(line, sep);
    count = m_pred_map->size();
    assert(count > 0);
    size_t fid = 0;
    std::vector<pair<me::outcome_id_type, size_t> > params;
    for (size_t i = 0; i < count; ++i) {
        params.clear();
        getline(f, line);
        me::outcome_id_type oid;
        tokens.assign(line);

        tokenizer_t::iterator it = tokens.begin();
        ++it; // skip count which is only used in binary format
        for (; it != tokens.end();) {
            oid = atoi(it->c_str()); ++it;
            params.push_back(make_pair(oid,fid++));
        }
        m_params->push_back(params);
    }

    // load theta
    getline(f, line);
    m_n_theta = atoi(line.c_str());
    cout << "fid = " << fid << " m_n_theta = " << m_n_theta << endl;
    assert(fid == m_n_theta);
    m_theta.reset(new double[m_n_theta]);

    size_t i = 0;
    while (getline(f, line)) {
        assert(!line.empty());
        // if (line[0] == '#') continue;
        m_theta[i++] = atof(line.c_str());
    }
    assert(i == m_n_theta);
}

// load model from a binary file using zlib file io
// TODO: io error detection
void MaxentModelFile::load_model_bin(const string& model) {
    gzFile f;
    f = gzopen(model.c_str(), "rb");
    if (f == NULL)
        throw runtime_error("Fail to open model file to read");

    // skip header
    gzseek(f, header_len, 0);

    m_pred_map.reset(new me::PredMapType);
    m_outcome_map.reset(new me::OutcomeMapType);
    m_params.reset(new me::ParamsType);
    m_theta.reset(0);

    // size_t count;
    unsigned count;
    // size_t len;
    unsigned len;
    char buf[4096]; // TODO: handle unsafe buffer
    // read context predicates
    gzread(f, (void*)&count, sizeof(count));
    for (size_t i = 0; i < count; ++i) {
        gzread(f, (void*)&len, sizeof(len));
        gzread(f, (void*)&buf, len);
        m_pred_map->add(string(buf, len));
    }

    // read outcomes
    gzread(f, (void*)&count, sizeof(count));
    for (size_t i = 0; i < count; ++i) {
        gzread(f, (void*)&len, sizeof(len));
        gzread(f, (void*)&buf, len);
        m_outcome_map->add(string(buf, len));
    }

    // read paramaters
    count = m_pred_map->size();
    assert(count > 0);
    // size_t fid = 0;
    unsigned fid = 0;
    // size_t oid;
    unsigned oid;
    for (size_t i = 0; i < count; ++i) {
        std::vector<pair<me::outcome_id_type, size_t> > params;

        gzread(f, (void*)&len, sizeof(len));
        for (size_t j = 0; j < len; ++j) {
            gzread(f, (void*)&oid, sizeof(oid));
            params.push_back(make_pair(oid,fid++));
        }
        m_params->push_back(params);
    }

    // load theta
    gzread(f, (void*)&m_n_theta, sizeof(m_n_theta));
    assert(fid == m_n_theta);
    m_theta.reset(new double[m_n_theta]);

    for (size_t i = 0; i < m_n_theta; ++i) {
        gzread(f, (void*)&m_theta[i], sizeof(double));
    }

    gzclose(f);
}


void MaxentModelFile::save(const string& model, bool binary) {
    if (!m_params || !m_pred_map || !m_outcome_map)
        throw runtime_error("can not save empty model");

    if (binary)
        save_model_bin(model);
    else
        save_model_txt(model);
}

void MaxentModelFile::save_model_txt(const string& model) {
    assert(m_params);

    ofstream f(model.c_str());
    f.precision(20);
    // f << scientific;
    if (!f)
        throw runtime_error("unable to open model file to write");

    // f2 << scientific;

    // todo: write a header section here
    f << "#txt,maxent" << endl;

    f << m_pred_map->size() << endl;
    for (size_t i = 0;i < m_pred_map->size(); ++i)
        f << (*m_pred_map)[i] << endl;

    f << m_outcome_map->size() << endl;
    for (size_t i = 0;i < m_outcome_map->size(); ++i)
        f << (*m_outcome_map)[i] << endl;

    for (size_t i = 0;i < m_params->size(); ++i) {
        const vector<pair<me::outcome_id_type, size_t> >& param = (*m_params)[i];
        f << param.size() << ' ';
        for (size_t j = 0; j < param.size(); ++j) {
            f << param[j].first << ' ';
            // f2 << '#' << (*m_outcome_map)[param[j].first] << '<--' <<  (*m_pred_map)[i] << endl;
        }
        f << endl;
    }

    // write theta
    f << m_n_theta << endl;
    for (size_t i = 0; i < m_n_theta; ++i)
        f << m_theta[i] << endl;
}

void MaxentModelFile::save_model_bin(const string& model) {
    assert(m_params);

    gzFile f;
	f = gzopen(model.c_str(),"wb");
	if (f == NULL)
        throw runtime_error("unable to open model file to write");

    // todo: write a header section here
    gzwrite(f, (void*)"#bin,maxent", header_len);

    size_t uint;
    uint = m_pred_map->size();
    gzwrite(f, (void*)&uint, sizeof(uint));
    for (size_t i = 0;i < m_pred_map->size(); ++i) {
        const string& s = (*m_pred_map)[i];
        uint = s.size();
        gzwrite(f,(void*)&uint, sizeof(uint));
        gzwrite(f, (void*)s.data(), s.size());
    }

    uint = m_outcome_map->size();
    gzwrite(f, (void*)&uint, sizeof(uint));
    for (size_t i = 0;i < m_outcome_map->size(); ++i) {
        const string& s = (*m_outcome_map)[i];
        uint = s.size();
        gzwrite(f, (void*)&uint, sizeof(uint));
        gzwrite(f, (void*)s.data(), s.size());
    }

    // write parameters
    for (size_t i = 0;i < m_params->size(); ++i) {
        const vector<pair<me::outcome_id_type, size_t> >& param = (*m_params)[i];
        uint = param.size();
        gzwrite(f, (void*)&uint, sizeof(uint));
        for (size_t j = 0; j < param.size(); ++j) {
            uint = param[j].first;
            gzwrite(f, (void*)&uint, sizeof(uint));
        }
    }

    // write theta
    gzwrite(f, (void*)&m_n_theta, sizeof(m_n_theta));
    for (size_t i = 0; i < m_n_theta; ++i)
        gzwrite(f, (void*)&m_theta[i], sizeof(double));
    gzclose(f);
}

//namespace rf {
RandomFieldModelFile::RandomFieldModelFile(): m_Z(0.0), m_n_theta(0) {}

shared_ptr<rf::featmap_type> RandomFieldModelFile::feat_map() {
    if (!m_feat_map)
        throw runtime_error("No model loaded");
    return m_feat_map;
}

void RandomFieldModelFile::params(double& Z, size_t& n_theta,
        shared_array<double>& theta) {
    if (!m_theta)
        throw runtime_error("No model loaded");
     Z       = m_Z;
     n_theta = m_n_theta;
     theta   = m_theta;
}

void RandomFieldModelFile::set_feat_map(
        shared_ptr<rf::featmap_type> feat_map) {
    assert(feat_map);
    m_feat_map = feat_map;
}

void RandomFieldModelFile::set_params(double Z, size_t n_theta,
        shared_array<double> theta){
    m_Z = Z;
    m_n_theta          = n_theta;
    m_theta            = theta;
}

void RandomFieldModelFile::load(const string& model) {
    bool binary;
    bool randomfield;
    check_modeltype(model, binary, randomfield);
    if (!randomfield)
        throw runtime_error("Trying to load a previously saved Maxent model from RandomFieldModelFile::load().");

    if (binary)
        load_model_bin(model);
    else
        load_model_txt(model);
}

void RandomFieldModelFile::load_model_txt(const string& model) {
    // cerr << "load model txt from " <<  model << endl;

    ifstream f(model.c_str());

    if (!f)
        throw runtime_error("fail to open model file");

    size_t count;
    string line;

    m_feat_map.reset(new rf::featmap_type);
    m_theta.reset(0);

    // skip header comments
    getline(f, line);
    while (line.empty() || line[0] == '#')
        getline(f, line);

    // read Z
    m_Z = atof(line.c_str());

    // read feature names
    getline(f, line);
    count = atoi(line.c_str());
    for (size_t i = 0; i < count; ++i) {
        getline(f, line);
        m_feat_map->add(line);
    }

    // load theta
    getline(f, line);
    m_n_theta = atoi(line.c_str());
    assert(m_feat_map->size() == m_n_theta);
    m_theta.reset(new double[m_n_theta]);

    size_t i = 0;
    while (getline(f, line)) {
        assert(!line.empty());
        // if (line[0] == '#') continue;
        m_theta[i++] = atof(line.c_str());
    }
    assert(i == m_n_theta);
}

// load model from a binary file using zlib file io
// TODO: io error detection
void RandomFieldModelFile::load_model_bin(const string& model) {
    gzFile f;
    f = gzopen(model.c_str(), "rb");
    if (f == NULL)
        throw runtime_error("Fail to open model file to read");

    // skip header
    gzseek(f, header_len, 0);

    m_feat_map.reset(new rf::featmap_type);
    m_theta.reset(0);

    size_t count;
    size_t len;
    char buf[4096]; // TODO: handle unsafe buffer
    // read global constant Z
    gzread(f, (void*)&m_Z, sizeof(m_Z));

    // read feature names
    gzread(f, (void*)&count, sizeof(count));
    for (size_t i = 0; i < count; ++i) {
        gzread(f, (void*)&len, sizeof(len));
        gzread(f, (void*)&buf, len);
        m_feat_map->add(string(buf, len));
    }

    // load theta
    gzread(f, (void*)&m_n_theta, sizeof(m_n_theta));
    assert(m_feat_map->size() == m_n_theta);
    m_theta.reset(new double[m_n_theta]);

    for (size_t i = 0; i < m_n_theta; ++i) {
        gzread(f, (void*)&m_theta[i], sizeof(double));
    }

    gzclose(f);
}


void RandomFieldModelFile::save(const string& model, bool binary) {
    if (!m_feat_map || !m_theta)
        throw runtime_error("can not save empty model");

    if (binary)
        save_model_bin(model);
    else
        save_model_txt(model);
}

void RandomFieldModelFile::save_model_txt(const string& model) {

    ofstream f(model.c_str());
    f.precision(20);
    // f << scientific;
    if (!f)
        throw runtime_error("unable to open model file to write");

    // f2 << scientific;

    // todo: write a header section here
    f << "#txt,randomfield" << endl;

    f << m_Z << endl;

    f << m_feat_map->size() << endl;
    for (size_t i = 0;i < m_feat_map->size(); ++i)
        f << (*m_feat_map)[i] << endl;

    // write theta
    f << m_n_theta << endl;
    for (size_t i = 0; i < m_n_theta; ++i)
        f << m_theta[i] << endl;
}

void RandomFieldModelFile::save_model_bin(const string& model) {

    gzFile f;
	f = gzopen(model.c_str(),"wb");
	if (f == NULL)
        throw runtime_error("unable to open model file to write");

    // todo: write a header section here
    gzwrite(f, (void*)"#bin,randomfield", header_len);

    gzwrite(f, (void*)&m_Z, sizeof(m_Z));

    size_t uint;
    uint = m_feat_map->size();
    gzwrite(f, (void*)&uint, sizeof(uint));
    for (size_t i = 0;i < m_feat_map->size(); ++i) {
        const string& s = (*m_feat_map)[i];
        uint = s.size();
        gzwrite(f,(void*)&uint, sizeof(uint));
        gzwrite(f, (void*)s.data(), s.size());
    }

    // write theta
    gzwrite(f, (void*)&m_n_theta, sizeof(m_n_theta));
    for (size_t i = 0; i < m_n_theta; ++i)
        gzwrite(f, (void*)&m_theta[i], sizeof(double));
    gzclose(f);
}

//} // namespace rf

// (disabled) load model from a binary file using mmap() call {{{
/*
void RandomFieldModelFile::load_model_bin(const string& model, const string& param) {
    string file = model + ".model";

    FMmap mmapfile(file.c_str());
    const char* ptr = mmapfile.data();
    ptr += header_len * sizeof(char);

    m_feat_map.reset(new FeatMapType);
    m_outcome_map.reset(new OutcomeMapType);
    m_params.reset(new ParamsType);
    m_theta.reset(0);

    // load theta {{{
    {
        if (!param.empty())
            file = param + ".param";
        else
            file = model + ".param";
        FMmap f(file.c_str());

        const char* ptr = f.data();

        m_n_theta = *(size_t*)ptr;
        ptr += sizeof(size_t);
        m_theta.reset(new double[m_n_theta]);

        const double* q = (const double*)ptr;
        for (size_t i = 0; i < m_n_theta; ++i) {
            m_theta[i] = *q++;
        }
    } // }}}

    size_t count;
    // read context predicates
    count = *(size_t*)ptr;
    ptr += sizeof(size_t);
    for (size_t i = 0; i < count; ++i) {
        size_t len = *(size_t*)ptr;
        m_feat_map->add(string(ptr + sizeof(size_t), len));
        ptr += len + sizeof(size_t);
    }

    // read outcomes
    count = *(size_t*)ptr;
    ptr += sizeof(size_t);
    for (size_t i = 0; i < count; ++i) {
        size_t len = *(size_t*)ptr;
        m_outcome_map->add(string(ptr + sizeof(size_t), len));
        ptr += len + sizeof(size_t);
    }

    // read paramaters
    count = m_feat_map->size();
    assert(count > 0);
    size_t fid = 0;
    for (size_t i = 0; i < count; ++i) {
        std::vector<pair<me::outcome_id_type, size_t> > params;

        size_t* q = (size_t*)(ptr);
        size_t len = *q++;
        for (size_t j = 0; j < len; ++j)
            params.push_back(make_pair(*q++,fid++));
        m_params->push_back(params);
        ptr = (const char*)q;
    }
    assert(fid == m_n_theta);
} }}} */

} // namespace maxent


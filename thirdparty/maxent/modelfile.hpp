/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * modelfile.hpp  -  helper classes for loading and saving Maxent/RandomField
 * Model
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 28-May-2003
 * Last Change : 12-Mar-2004.
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

#ifndef MODELFILE_H
#define MODELFILE_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "meevent.hpp"
#include "rfevent.hpp"

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

namespace maxent {
using boost::shared_ptr;
using boost::shared_array;
using std::string;

void check_modeltype(const string& file, bool& binary, bool& randomfield);

//namespace me {
class MaxentModelFile : boost::noncopyable {
    public:
        MaxentModelFile();
        shared_ptr<me::PredMapType> pred_map();
        shared_ptr<me::OutcomeMapType> outcome_map();
        void params(shared_ptr<me::ParamsType>& params, size_t& n_theta,
                shared_array<double>& theta);

        void set_pred_map(shared_ptr<me::PredMapType> pred_map);
        void set_outcome_map(shared_ptr<me::OutcomeMapType> outcome_map);
        void set_params(shared_ptr<me::ParamsType> params, size_t n_theta, 
                shared_array<double> theta);

        void load(const string& model);
        void save(const string& model, bool binary);

    private:
        void load_model_txt(const string& model);
        void save_model_txt(const string& model);
        void save_model_bin(const string& model);
        void load_model_bin(const string& model);

        // size_t m_n_theta; // number of feature weights
        unsigned m_n_theta; // number of feature weights
        shared_ptr<me::ParamsType>     m_params;    // params for builtin model
        shared_array<double>       m_theta;         // feature weight
        // shared_array<double>       m_sigma;         // Gaussian prior
        shared_ptr<me::PredMapType>    m_pred_map;
        shared_ptr<me::OutcomeMapType> m_outcome_map;
};
//} // namespace me

//namespace rf {
class RandomFieldModelFile : boost::noncopyable{
    public:
        RandomFieldModelFile();

        shared_ptr<rf::featmap_type> feat_map();

        void params(double& Z, size_t& n_theta, shared_array<double>& theta);

        void set_feat_map(shared_ptr<rf::featmap_type> feat_map);

        void set_params(double Z, size_t n_theta, shared_array<double> theta);

        void load(const string& model);

        void save(const string& model, bool binary);

    private:
        void load_model_txt(const string& model);
        void save_model_txt(const string& model);
        void save_model_bin(const string& model);
        void load_model_bin(const string& model);

        double m_Z; // global constant
        size_t m_n_theta; // number of feature weights
        shared_array<double>       m_theta;         // feature weight
        shared_ptr<rf::featmap_type>    m_feat_map;

};

//} // namespace rf
} // namespace maxent
#endif /* ifndef MODELFILE_H */


/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * trainer.hpp  -  abstract Trainer interface for conditional ME trainers
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 01-Jun-2003
 * Last Change : 24-Dec-2004.
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

#ifndef TRAINER_H
#define TRAINER_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <string>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include "meevent.hpp"

namespace maxent{
using namespace std;
using boost::shared_ptr;
using boost::shared_array;
using namespace me;

// utility function
void load_events_txt(const string& filename, MEEventSpace& events);
void save_events_txt(const string& filename, const MEEventSpace& events);

/**
 * Trainer class provides an abstract interface to various training
 * algorithms.
 *
 * Usually you need not use this class explicitly. \ref MaxentModel::train()
 * provides a wrapper for the underline Trainer instances.
 */
class Trainer /*: boost::noncopyable*/{
    public:
        virtual ~Trainer() {}
        virtual void train(size_t iter = 15, double tol = 1E-05) = 0;

        // void save_param(const string& model, bool binary) const;

        void load_training_data(const string& events, const string& model);

        void set_training_data(shared_ptr<MEEventSpace> es,
                shared_ptr<ParamsType> params,
                size_t n_theta,
                shared_array<double> theta,
                shared_array<double> sigma2,
                size_t n_outcomes,
                shared_ptr<MEEventSpace> heldout_es =
                shared_ptr<MEEventSpace>()
                );

    protected:
        size_t m_n_outcomes;
        size_t m_N;          // total number of examples
        size_t m_n_theta;

        shared_ptr<MEEventSpace> m_es;
        shared_ptr<MEEventSpace> m_heldout_es;
        shared_ptr<ParamsType>   m_params;
        shared_array<double>     m_theta;
        shared_array<double>     m_sigma2;

        size_t eval(const Event::context_type* context, size_t len,
                vector<double>& probs) const;
    private:
        // void save_param_txt(const string& model) const;
};

} // namespace maxent


#endif /* ifndef TRAINER_H */


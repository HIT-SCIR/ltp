/*
 * vi:ts=4:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * gistrainer.hpp  -  a trainer for conditional ME model with GIS algorithm
 *
 * An implementation of Generalized Iterative Scaling.  The reference paper
 * for this implementation was Adwait Ratnaparkhi's tech report at the
 * University of Pennsylvania's Institute for Research in Cognitive Science,
 * and is available at ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z
 *
 * This C++ implementation is originally based on java maxent implementation,
 * with the help of developers from java maxent.
 * see http://maxent.sf.net
 *
 * Copyright (C) 2002 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 31-Dec-2002
 * Last Change : 01-Jul-2004.
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

#ifndef GISTRAINER_H
#define GISTRAINER_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>

#include "trainer.hpp"

namespace maxent{

    class GISTrainer : public Trainer {
        public:
            // GISTrainer();
            // ~GISTrainer();

            void train(size_t iter = 100, double tol = 1.0E-05);

        private:
            void init_trainer();
            double newton(double f_q, double f_ref, size_t i, double tol = 1.0E-6);
#if !defined(_STLPORT_VERSION) && defined(_MSC_VER) && (_MSC_VER >= 1300)
            // for MSVC7's hash_map declaration
            class featid_hasher : public stdext::hash_compare<pair<size_t, size_t> > {
                public:
                    size_t operator()(const pair<size_t, size_t>& p) const {
                        return p.first + p.second;
                    }

                    bool operator()(const pair<size_t, size_t>& k1,
                            const  pair<size_t, size_t>& k2) {
                        return k1 < k2;
                    }
            };
#else
            // for hash_map of GCC & STLPORT
            struct featid_hasher {
                size_t operator()(const pair<size_t, size_t>& p) const {
                    return p.first + p.second;
                }
            };

#endif

            double m_correct_constant;
            shared_ptr<vector<vector<double> > > m_modifiers;
            shared_ptr<vector<vector<double> > > m_observed_expects;
    };

} // namespace maxent

#endif /* ifndef GISTRAINER_H */


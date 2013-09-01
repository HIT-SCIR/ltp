/*
 *
 * Yet another implementation of the Maximum Entropy Model.
 * Support L1-regularized objective, OWLQN and SGD optimization.
 * Support L2-regularized objective, LBFGS optimization.
 *
 * Developed based on the implementation of Tsujii lab
 * ref: http://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/maxent/
 *
 * Author: Jiang Guo (jguo@ir.hit.edu.cn)
 *
 */

#ifndef __MAXENT_H_
#define __MAXENT_H_

#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include <string>
#include "opmath.h"


namespace maxent {

using namespace math::vectype;
using namespace math::minifunc;

class ME_Sample {

    /* Class for training sample */

    public:
        /* discrete-valued features */
        std::vector<std::string> features;

        /* real-valued features */
        std::vector< std::pair<std::string, double> > rvfeatures;

        std::string label;

    public:
        ME_Sample() : label("") {}
        ME_Sample(const std::string& l) : label(l) {}
        ME_Sample(const std::vector<std::string>& vec_features,
                bool with_label = false)
        {
            size_t feature_beg = 0;
            if (with_label)
            {
                label = vec_features[0];
                feature_beg = 1;
            }

            for (size_t i = feature_beg; i < vec_features.size(); ++i)
                add_feature(vec_features[i]);
        }

        void add_feature(const std::string& f)
        {
            features.push_back(f);
        }

        void add_feature(const std::string& f, const double v)
        {
            rvfeatures.push_back(std::pair<std::string, double>(f, v));
        }
};

enum SOLVER_TYPE
{
    L2_LBFGS, // L2-regularized objective, optimized using LBFGS.
    L1_OWLQN, // L1-regularized objective, optimized using OWLQN.
    L1_SGD    // L1-regularized objective, optimized using SGD.
};

struct ME_Parameter
{
    ME_Parameter()
    {
        init_params();
    }

    private:
        void init_params()
        {
            solver_type  = L2_LBFGS;
            sgd_iter     = 30;
            sgd_eta0     = 1;
            sgd_alpha    = 0.85;
            l2_reg       = 1.0;
            l1_reg       = 1.0;
            nheldout     = 0;
        }

    public:
        SOLVER_TYPE solver_type; // either {L2_LBFGS, L1_OWLQN, L1_SGD}

        /* parameters for sgd optimization */
        int         sgd_iter;
        double      sgd_eta0;
        double      sgd_alpha;

        /* coefficients for L1/L2 regularization */
        double      l1_reg;
        double      l2_reg;
        int         nheldout;
};

// typedef struct ME_Parameter ME_Parameter;

class ME_Model {

    /* Class for Maxent model */
    public:

        ME_Model() {}

        ME_Model(ME_Parameter& param) : _param(param) {}
        ME_Model(const std::string& model_path)
        {
            load(model_path);
        }

        void add_training_sample(const ME_Sample& mes);

        int  train();
        std::vector<double> predict(ME_Sample& mes) const;
        typedef std::vector< std::pair<std::string, double> > Prediction;
        void predict(ME_Sample& mes, Prediction& outcome,
                bool sort_result = true) const;

        bool load(const std::string& modelfile);
        bool save(const std::string& modelfile, double weight_cutoff = 0);

        int num_classes() const
        {
            return _num_classes;
        }

        std::string get_class_label(int id) const
        {
            return _set_label.str(id);
        }

        int get_class_id(const std::string& label) const
        {
            return _set_label.id(label);
        }

        /* unused */
        void set_heldout(const int h)
        {
            _param.nheldout = h;
        }

        void set_solver_type(SOLVER_TYPE solver)
        {
            _param.solver_type = solver;
        }

        void set_l1_regularizer(const double v)
        {
            _param.l1_reg = v;
        }

        void set_l2_regularizer(const double v)
        {
            _param.l2_reg = v;
        }

        void release();

    private:
        /* parameters for training */
        ME_Parameter _param;

        struct cmp_outcome
        {
            bool operator()(const std::pair<std::string, double>& lpr,
                    const std::pair<std::string, double>& rpr) const
            {
                return lpr.second > rpr.second;
            }
        };

        struct Sample
        {
            int label;
            std::vector<int> features;
            std::vector< std::pair<int, double> > rvfeatures;

            /* This is for sort samples, so that
             * features could be in increasing order by their ids
             */
            bool operator<(const Sample& x) const
            {
                for (unsigned int i = 0; i < features.size(); ++i)
                {
                    if (i >= x.features.size()) return false;
                    if (features[i] < x.features[i])
                        return true;
                    if (features[i] > x.features[i])
                        return false;
                }
                return false;
            }
        };

        struct ME_Feature
        {
            /*
             * Low 8 bits for the label while
             * High 24 bits for the feature
             */
            enum { MAX_LABEL_TYPES = 255 };

            ME_Feature(const int l, const int f)
                : _body((f << 8) + l)
            {
                assert(l >= 0 && l <= MAX_LABEL_TYPES);
                assert(f >= 0 && f <= 0xffffff);
            }

            int label() const
            {
                return _body & 0xff;
            }

            int feature() const
            {
                return _body >> 8;
            }

            unsigned int body() const
            {
                return _body;
            }

            private:
                unsigned int _body;
        };

        /*
         * structure for bidirectional mefeature<->id mapping
         */
        struct ME_FeatureSet
        {
            std::map<unsigned int, int> mef2id;   /* map from ME_Feature to id */
            std::vector<ME_Feature> id2mef;  /* map from id to ME_Feature */

            int append(const ME_Feature & mef)
            {
                std::map<unsigned int, int>::const_iterator i = mef2id.find(mef.body());

                if (i == mef2id.end())
                {
                    int fid = id2mef.size();
                    id2mef.push_back(mef);
                    mef2id[mef.body()] = fid;

                    return fid;
                }

                return i->second;
            }

            int id(const ME_Feature& mef) const
            {
                std::map<unsigned int, int>::const_iterator i = mef2id.find(mef.body());
                if (i == mef2id.end())
                    return -1;
                return i->second;
            }

            ME_Feature feature(int fid) const
            {
                if (fid >= 0 && fid < (int)id2mef.size())
                    return id2mef[fid];
                throw std::runtime_error("error : feature id beyond the scope of ME_FeatureSet");
            }

            int size() const
            {
                return id2mef.size();
            }

            void clear()
            {
                mef2id.clear();
                id2mef.clear();
            }
        };

        /*
         * structure for str->id mapping, for label-free features
         */
        struct ME_MiniStringSet 
        {
            std::map<std::string, int> str2id;
            ME_MiniStringSet() : _size(0) {}

            int append(const std::string& str)
            {
                std::map<std::string, int>::const_iterator i = str2id.find(str);
                if (i == str2id.end())
                {
                    int sid = _size;
                    _size++;
                    str2id[str] = sid;
                    return sid;
                }
                return i->second;
            }

            int id(const std::string& str) const
            {
                std::map<std::string, int>::const_iterator i = str2id.find(str);
                if (i == str2id.end())
                {
                    return -1;
                }
                return i->second;
            }

            int size() const
            {
                return _size;
            }

            void clear()
            {
                str2id.clear();
                _size = 0;
            }

            std::map<std::string, int>::const_iterator begin() const
            {
                return str2id.begin();
            }

            std::map<std::string, int>::const_iterator end() const
            {
                return str2id.end();
            }

            private:
                int _size;
        };

        /*
         * structure for (bi-directional)str<->id mapping 
         */
        struct ME_StringSet : public ME_MiniStringSet 
        {
            std::vector<std::string> id2str;

            int append(const std::string& str)
            {
                std::map<std::string, int>::const_iterator i = str2id.find(str);
                if (i == str2id.end())
                {
                    int sid = id2str.size();
                    str2id[str] = sid;
                    id2str.push_back(str);
                    return sid;
                }
                return i->second;
            }

            std::string str(const int sid) const
            {
                if (sid >= 0 && sid < (int)id2str.size())
                    return id2str[sid];
                throw std::runtime_error("error : id beyond the scope of StringSet");
            }

            int size() const
            {
                return id2str.size();
            }

            void clear()
            {
                str2id.clear();
                id2str.clear();
            }
        };

        std::vector<Sample> _training_data;
        std::vector<Sample> _heldout_data;  // data for validation while training
        std::vector<double> _vec_lambda;    // parameters for maxent model
        ME_StringSet        _set_label;     // labels appears in training data
        ME_MiniStringSet    _set_feature;   // simple feature (without label)
        ME_FeatureSet       _set_mefeature; // maxent feature (with label)
        int                 _num_classes;

        std::vector<double> _vec_empirical_expectation;
        std::vector<double> _vec_model_expectation;

        /* map from simple feature(id) to maxent feature(id) */
        std::vector< std::vector<int> > _feature2mef;

        double _train_error;    // error rate on training data
        double _heldout_error;  // error rate on heldout data

    private:
        double heldout_likelihood();
        int    classify( // classify a sample with model's lambdas
                const Sample& s,
                std::vector<double>& vp) const;
        int    classify( // classify a sample with self-defined lambdas
                const std::vector<double>& x,
                const Sample& s,
                std::vector<double>& vp) const;
        void   collect_mefeature_set(const int cutoff = 0);
        double update_model_expectation(); // frequently called for computing gradients
        void   init_feature2mef();

        /* I am planning to seperate the optimization methods
         * into an independent class
         */
        void perform_SGD();
        void sgd_apply_penalty(
                const int i,
                const double u,
                std::vector<double>& q);

        void perform_LBFGS();
        void perform_OWLQN();

        double backtracking_line_search(  // for LBFGS
                const Vec& x0,
                const Vec& grad0,
                const double f0,
                const Vec& dx,
                Vec& x,
                Vec& grad1);
        double constrained_line_search(   // for OWLQN
                const Vec& x0,
                const Vec& grad0,
                const double f0,
                const Vec& dx,
                Vec& x,
                Vec& grad1);

        double l1_regularized_func_gradient(
                const Vec& x,
                Vec& grad);

        /*
         * l2-regularizer is supposed to be always turned on
         */
        double l2_regularized_func_gradient(
                const std::vector<double>& x,
                std::vector<double>& grad);

};

} // end namespace maxent

#endif

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "maxent.h"
#include "opmath.h"

namespace maxent {

using namespace std;
using namespace math;

void ME_Model::add_training_sample(const ME_Sample& mes)
{
    Sample s;
    s.label = _set_label.append(mes.label);
    if (s.label > ME_Feature::MAX_LABEL_TYPES)
    {
        cerr << "error: too many types of labels (limit: 255)." << endl;
        exit(1);
    }

    for (vector<string>::const_iterator i = mes.features.begin(); i != mes.features.end(); ++i)
    {
        s.features.push_back(_set_feature.append(*i));
    }

    for (vector< pair<string, double> >::const_iterator i = mes.rvfeatures.begin(); i != mes.rvfeatures.end(); ++i)
    {
        s.rvfeatures.push_back(pair<int, double>(_set_feature.append(i->first), i->second));
    }

    _training_data.push_back(s);
}

int ME_Model::train()
{
    if (_training_data.size() == 0)
    {
        cerr << "error: no training samples." << endl;
        exit(1);
    }

    if (_param.nheldout >= (int)_training_data.size())
    {
        cerr << "error: too much heldout data." << endl;
        exit(1);
    }

    int max_label = 0;
    for (std::vector<Sample>::const_iterator i = _training_data.begin(); i != _training_data.end(); ++i)
    {
        max_label = max(max_label, i->label);
    }
    _num_classes = max_label + 1;

    if (_num_classes != _set_label.size())
    {
        cerr << "[debug] _num_classes != _set_label.size()" << endl;
    }

    for (int i = 0; i < _param.nheldout; ++i)
    {
        _heldout_data.push_back(_training_data.back());
        _training_data.pop_back();
    }

    /* for better feature scanning */
    sort(_training_data.begin(), _training_data.end());

    if ((_param.solver_type == L1_SGD || _param.solver_type == L1_OWLQN)
            && _param.l1_reg > 0)
        cerr << "L1 regularizer = " << _param.l1_reg << endl;
    if (_param.solver_type == L2_LBFGS && _param.l2_reg > 0)
        cerr << "L2 regularizer = " << _param.l2_reg << endl;

    _param.l1_reg /= _training_data.size();
    _param.l2_reg /= _training_data.size();

    cerr << "collecting positive feature set for estimation...";
    collect_mefeature_set();   /* To-do: allow feature cutoff */
    cerr << "[done]" << endl;
    cerr << "number of samples = " << _training_data.size() << endl;
    cerr << "number of features = " << _set_mefeature.size() << endl;

    cerr << "calculating empirical expectation...";
    _vec_empirical_expectation.resize(_set_mefeature.size());
    for (size_t i = 0; i < _set_mefeature.size(); ++i)
    {
        _vec_empirical_expectation[i] = 0.0;
    }

    for (size_t i = 0; i < _training_data.size(); ++i)
    {
        const Sample* s = &_training_data[i];
        for (vector<int>::const_iterator j = s->features.begin(); j != s->features.end(); ++j)
        {
            for (vector<int>::const_iterator k = _feature2mef[*j].begin(); k != _feature2mef[*j].end(); ++k)
            {
                if (_set_mefeature.feature(*k).label() == s->label)
                    _vec_empirical_expectation[*k] += 1.0;
            }
        }

        for (vector< pair<int, double> >::const_iterator j = s->rvfeatures.begin(); j != s->rvfeatures.end(); ++j)
        {
            for (vector<int>::const_iterator k = _feature2mef[j->first].begin(); k != _feature2mef[j->first].end(); ++k)
            {
                if (_set_mefeature.feature(*k).label() == s->label)
                    _vec_empirical_expectation[*k] += j->second;
            }
        }
    }

    for (size_t i = 0; i < _set_mefeature.size(); ++i)
    {
        _vec_empirical_expectation[i] /= _training_data.size();
    }
    cerr << "[done]" << endl;

    _vec_lambda.resize(_set_mefeature.size());
    for (size_t i = 0; i < _vec_lambda.size(); ++i)
    {
        _vec_lambda[i] = 0.0;
    }

    vector<double> x;
    switch (_param.solver_type)
    {
        case L1_SGD:
            cerr << "perform SGD" << endl;
            perform_SGD();   break;
        case L2_LBFGS:
            cerr << "perform LBFGS" << endl;
            perform_LBFGS(); break;
        case L1_OWLQN:
            cerr << "perform OWLQN" << endl;
            perform_OWLQN(); break;
        default:
            cerr << "error: unsupported solver type" << endl;
            break;
    }

    int num_active_features = 0;
    for (size_t i = 0; i < _set_mefeature.size(); ++i)
    {
        if (_vec_lambda[i] != 0)
            num_active_features++;
    }
    cerr << "number of active features = " << num_active_features << endl;

    return 0;
}

vector<double> ME_Model::predict(ME_Sample& mes) const
{
    Sample s;

    for (vector<string>::const_iterator i = mes.features.begin(); i != mes.features.end(); ++i)
    {
        int fid = _set_feature.id(*i);
        if (fid >= 0)
            s.features.push_back(fid);
    }
    for (vector< pair<string, double> >::const_iterator i = mes.rvfeatures.begin(); i != mes.rvfeatures.end(); ++i)
    {
        int fid = _set_feature.id(i->first);
        if (fid >= 0)
            s.rvfeatures.push_back(pair<int, double>(fid, i->second));
    }

    vector<double> vec_prob;
    int l = classify(s, vec_prob);
    mes.label = get_class_label(l);

    return vec_prob;
}

void ME_Model::predict(
        ME_Sample& mes,
        Prediction& outcomes,
        bool sort_result) const
{
    vector<double> vec_prob = predict(mes);

    for (size_t i = 0; i < vec_prob.size(); ++i)
    {
        string label = get_class_label(i);
        outcomes.push_back(make_pair(label, vec_prob[i]));
    }

    if (sort_result)
        sort(outcomes.begin(), outcomes.end(), cmp_outcome());
}


int ME_Model::classify(const Sample& s, vector<double>& vp) const
{
    vp.resize(_num_classes);
    for (size_t i = 0; i < _num_classes; ++i)
    {
        vp[i] = 0.0;
    }

    for (vector<int>::const_iterator i = s.features.begin(); i != s.features.end(); ++i)
    {
        for (vector<int>::const_iterator k = _feature2mef[*i].begin(); k != _feature2mef[*i].end(); ++k)
        {
            int label = _set_mefeature.feature(*k).label();
            vp[label] += _vec_lambda[*k];
        }
    }

    for (vector< pair<int, double> >::const_iterator i = s.rvfeatures.begin(); i != s.rvfeatures.end(); ++i)
    {
        for (vector<int>::const_iterator k = _feature2mef[i->first].begin(); k != _feature2mef[i->first].end(); ++k)
        {
            int label = _set_mefeature.feature(*k).label();
            vp[label] += _vec_lambda[*k] * i->second;
        }
    }

    double sum_prob = 0.0;
    vector<double>::const_iterator pmax = max_element(vp.begin(), vp.end());
    double offset = max(0.0, *pmax - 700);
    for (int label = 0; label < _num_classes; ++label)
    {
        double prod = exp(vp[label] - offset);
        vp[label] = prod;
        sum_prob += prod;
    }

    int max_label = -1;
    double max_prob = -1;
    for (int label = 0; label < _num_classes; ++label)
    {
        vp[label] /= sum_prob;

        if (vp[label] > max_prob)
        {
            max_label = label;
            max_prob  = vp[max_label];
        }
    }

    return max_label;
}

int ME_Model::classify(
        const vector<double>& x,
        const Sample& s,
        vector<double>& vp) const
{
    vp.resize(_num_classes);
    for (size_t i = 0; i < _num_classes; ++i)
    {
        vp[i] = 0.0;
    }

    for (vector<int>::const_iterator i = s.features.begin(); i != s.features.end(); ++i)
    {
        for (vector<int>::const_iterator k = _feature2mef[*i].begin(); k != _feature2mef[*i].end(); ++k)
        {
            int label = _set_mefeature.feature(*k).label();
            vp[label] += x[*k];
        }
    }

    for (vector< pair<int, double> >::const_iterator i = s.rvfeatures.begin(); i != s.rvfeatures.end(); ++i)
    {
        for (vector<int>::const_iterator k = _feature2mef[i->first].begin(); k != _feature2mef[i->first].end(); ++k)
        {
            int label = _set_mefeature.feature(*k).label();
            vp[label] += x[*k] * i->second;
        }
    }

    double sum_prob = 0.0;
    vector<double>::const_iterator pmax = max_element(vp.begin(), vp.end());
    double offset = max(0.0, *pmax - 700);
    for (int label = 0; label < _num_classes; ++label)
    {
        double prod = exp(vp[label] - offset);
        vp[label] = prod;
        sum_prob += prod;
    }

    int max_label = -1;
    double max_prob = -1;
    for (int label = 0; label < _num_classes; ++label)
    {
        vp[label] /= sum_prob;

        if (vp[label] > max_prob)
        {
            max_label = label;
            max_prob  = vp[max_label];
        }
    }

    return max_label;
}

bool ME_Model::load(const std::string& modelfile)
{
    ifstream fp(modelfile.c_str());
    if (!fp)
    {
        cerr << "error: cannot open model:" << modelfile << endl;
        return false;
    }

    _vec_lambda.clear();
    _set_label.clear();
    _set_feature.clear();
    _set_mefeature.clear();

    string buf;
    while (getline(fp, buf))
    {
        string::size_type p1 = buf.find_first_of('\t');
        string::size_type p2 = buf.find_last_of('\t');

        string class_name = buf.substr(0, p1);
        string feature_name = buf.substr(p1+1, p2-p1-1);
        double lambda = atof(buf.substr(p2+1).c_str());

        int label = _set_label.append(class_name);
        int feature = _set_feature.append(feature_name);

        _set_mefeature.append(ME_Feature(label, feature));
        _vec_lambda.push_back(lambda);
    }

    _num_classes = _set_label.size();

    init_feature2mef();

    return true;
}

bool ME_Model::save(const std::string& modelfile,
                double weight_cutoff)
{
    ofstream fp(modelfile.c_str());
    if (!fp)
    {
        cerr << "error: cannot open " << modelfile << endl;
        return false;
    }

    for (map<string, int>::const_iterator i = _set_feature.begin(); i != _set_feature.end(); ++i)
    {
        for (int j = 0; j < _set_label.size(); ++j)
        {
            string label = _set_label.str(j);
            string feature = i->first;

            int fid = _set_mefeature.id(ME_Feature(j, i->second));
            if (fid < 0) continue;
            if (_vec_lambda[fid] == 0) continue;
            if (fabs(_vec_lambda[fid]) < weight_cutoff) continue;

            fp << label << "\t" << feature << "\t" << _vec_lambda[fid] << endl;
        }
    }

    fp.close();

    return true;
}

double ME_Model::heldout_likelihood()
{
    double log_likelihood = 0;
    int n_error = 0;

    for (vector<Sample>::const_iterator i = _heldout_data.begin(); i != _heldout_data.end(); ++i)
    {
        vector<double> vec_prob;
        int label = classify(*i, vec_prob);
        log_likelihood += log(vec_prob[i->label]);

        if (label != i->label)
            n_error++;
    }

    _heldout_error = (double)n_error / _heldout_data.size();
    log_likelihood /= _heldout_data.size();

    return log_likelihood;
}

void ME_Model::collect_mefeature_set(const int cutoff)
{
    map<unsigned int, int> feature_freq;
    if (cutoff > 0)
    {
        /* calculate the feature count */
        for (vector<Sample>::const_iterator i = _training_data.begin(); i != _training_data.end(); ++i)
        {
            for (vector<int>::const_iterator j = i->features.begin(); j != i->features.end(); ++j)
            {
                feature_freq[ME_Feature(i->label, *j).body()]++;
            }

            for (vector< pair<int, double> >::const_iterator j = i->rvfeatures.begin(); j != i->rvfeatures.end(); ++j)
            {
                feature_freq[ME_Feature(i->label, j->first).body()]++;
            }
        }
    }

    /* collect features */
    for (vector<Sample>::const_iterator i = _training_data.begin(); i != _training_data.end(); ++i)
    {
        // days ago, mason told me that Michael Collins is a gay...
        // I feel so sad that I stuck here for two days...
        for (vector<int>::const_iterator j = i->features.begin(); j != i->features.end(); ++j)
        {
            const ME_Feature mefeature(i->label, *j);
            if (cutoff > 0 && feature_freq[mefeature.body()] <= cutoff)
            {
                continue;
            }
            int fid = _set_mefeature.append(mefeature);
        }

        for (vector< pair<int, double> >::const_iterator j = i->rvfeatures.begin(); j != i->rvfeatures.end(); ++j)
        {
            const ME_Feature mefeature(i->label, j->first);
            if (cutoff > 0 && feature_freq[mefeature.body()] <= cutoff)
                continue;
            _set_mefeature.append(mefeature);
        }
    }

    feature_freq.clear();

    init_feature2mef();
}

double ME_Model::update_model_expectation()
{
    /* update the model expectation of features */

    double log_likelihood = 0;
    int n_error = 0;

    /* here is a lesson paid for with many many blood
     * do not use the stupid 'push_back' here!
     */
    _vec_model_expectation.resize(_set_mefeature.size());
    for (size_t i = 0; i < _set_mefeature.size(); ++i)
    {
        _vec_model_expectation[i] = 0;
    }

    for (vector<Sample>::const_iterator i = _training_data.begin(); i != _training_data.end(); ++i)
    {
        vector<double> vec_prob(_num_classes);
        int plabel = classify(*i, vec_prob);

        // increment the log-likelihood
        log_likelihood += log(vec_prob[i->label]);
        if (plabel != i->label) n_error++;

        // calculate the model expectation
        for (vector<int>::const_iterator j = i->features.begin(); j != i->features.end(); ++j)
        {
            for (vector<int>::const_iterator k = _feature2mef[*j].begin(); k != _feature2mef[*j].end(); ++k)
            {
                _vec_model_expectation[*k] += vec_prob[_set_mefeature.feature(*k).label()];
            }
        }

        for (vector< pair<int, double> >::const_iterator j = i->rvfeatures.begin(); j != i->rvfeatures.end(); ++j)
        {
            for (vector<int>::const_iterator k = _feature2mef[j->first].begin(); k != _feature2mef[j->first].end(); ++k)
            {
                _vec_model_expectation[*k] += vec_prob[_set_mefeature.feature(*k).label()] * j->second;
            }
        }
    }

    for (size_t i = 0; i < _set_mefeature.size(); ++i)
    {
        _vec_model_expectation[i] /= _training_data.size();
    }

    _train_error = (double)n_error / _training_data.size();
    log_likelihood /= _training_data.size();

    if (_param.solver_type == L2_LBFGS && (_param.l2_reg > 0))
    {
        for (size_t i = 0; i < _set_mefeature.size(); ++i)
        {
            log_likelihood -= _vec_lambda[i] * _vec_lambda[i] * _param.l2_reg;
        }
    }

    return log_likelihood;
}

void ME_Model::init_feature2mef()
{
    _feature2mef.clear();
    for (size_t i = 0; i < _set_feature.size(); ++i)
    {
        vector<int> vec_l;
        for (int k = 0; k < _num_classes; ++k)
        {
            int fid = _set_mefeature.id(ME_Feature(k, i));
            if (fid >= 0)
                vec_l.push_back(fid);
        }
        _feature2mef.push_back(vec_l);
    }
}

double ME_Model::l1_regularized_func_gradient(
        const Vec & x,
        Vec & grad)
{
    // negtive log-likelihood
    // gradient stays the same with the l2-regularization
    double score = l2_regularized_func_gradient(x.stl_vec(), grad.stl_vec());

    for (size_t i = 0; i < x.size(); ++i)
    {
        score += _param.l1_reg * fabs(x[i]);
    }

    return score;
}

double ME_Model::l2_regularized_func_gradient(
        const std::vector<double> & x,
        std::vector<double> & grad)
{
    if (_set_mefeature.size() != x.size())
    {
        cerr << "error: incompatible vector length." << endl;
        exit(1);
    }

    for (size_t i = 0; i < x.size(); ++i)
    {
        _vec_lambda[i] = x[i];
    }

    double score = update_model_expectation();

    if (_param.solver_type == L2_LBFGS && (_param.l2_reg > 0))
    {
        for (size_t i = 0; i < x.size(); ++i)
        {
            grad[i] = -(_vec_empirical_expectation[i] - _vec_model_expectation[i]
                      - 2 * _param.l2_reg * _vec_lambda[i]);
        }
    }
    else
    {
        for (size_t i = 0; i < x.size(); ++i)
        {
            grad[i] = -(_vec_empirical_expectation[i] - _vec_model_expectation[i]);
        }
    }

    return -score;
}

}

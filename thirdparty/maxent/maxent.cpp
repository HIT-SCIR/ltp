/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 *
 * maxent.cpp  -  A handy command line maxent utility built on top of the
 * maxent library.
 *
 * Copyright (C) 2003 by Zhang Le <ejoy@users.sourceforge.net>
 * Begin       : 02-Sep-2003
 * Last Change : 11-Sep-2004.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>

#if HAVE_GETTIMEOFDAY
    #include <sys/time.h> // for gettimeofday()
#endif

#include <cstdlib>
#include <cassert>
#include <stdexcept> //for std::runtime_error
#include <memory>    //for std::bad_alloc
#include <iostream>
#include <string>
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "line_stream_iterator.hpp"
#include "maxentmodel.hpp"
#include "display.hpp" 
#include "maxent_cmdline.h"

#include "mmapfile.hpp"

#if defined(HAVE_SYSTEM_MMAP)
    #include "line_mem_iterator.hpp"
    #include "token_mem_iterator.hpp"
#endif

bool g_use_mmap = true;

using namespace std;
using namespace maxent;

typedef MaxentModel::context_type me_context_type;
typedef MaxentModel::outcome_type me_outcome_type;

//add_event helper function objects
struct AddEventToModel{
    AddEventToModel(MaxentModel& m)
        :model(m){}

    void operator()(const me_context_type& context, 
            const me_outcome_type& outcome) {
        model.add_event(context, outcome, 1);
    }
    private:
    MaxentModel& model;
};

struct AddHeldoutEventToModel{
    AddHeldoutEventToModel(MaxentModel& m)
        :model(m){}

    void operator()(const me_context_type& context,
            const me_outcome_type& outcome) {
        model.add_heldout_event(context, outcome, 1);
    }
    private:
    MaxentModel& model;
};

struct AddEventToVector{
    typedef vector<pair<me_context_type, me_outcome_type> >  EventVector_;
    AddEventToVector(EventVector_& v)
        :vec(v){}

    void operator()(const me_context_type& context, 
            const me_outcome_type& outcome) {
        vec.push_back(make_pair(context, outcome));
    }
    private:
    EventVector_& vec;
};

bool get_sample(const string& line, me_context_type& context, 
        me_outcome_type& outcome, bool binary_feature) {
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep(" \t");
    tokenizer tokens(line, sep);
    tokenizer::iterator it = tokens.begin();

    outcome = *it++;
    if (outcome.empty())
        return false;

    context.clear();
    if (binary_feature) {
        for (; it != tokens.end(); ++it)
            context.push_back(make_pair(*it, 1.0));
    } else {
        for (; it != tokens.end(); ++it) {
            size_t pos = it->find(':');
            if (pos == string::npos)
                return false;
            context.push_back(make_pair(it->substr(0, pos),
                        atof(it->substr(pos + 1).c_str())));
        }
    }
    return true;
}

#if defined(HAVE_SYSTEM_MMAP)
// the same as the above function, but use the faster token_mem_iterator
// to locate tokens in [begin, end)
bool get_sample(const char* begin, const char* end, me_context_type& context,
        me_outcome_type& outcome, bool binary_feature) {
    token_mem_iterator<> it(begin, end);
    token_mem_iterator<> it_end;

    outcome = string(it->first, it->second - it->first);
    if (outcome.empty())
        return false;
    ++it;

    context.clear();
    if (binary_feature) {
        for (; it != it_end; ++it) {
            context.push_back(make_pair(string(it->first,
                            it->second - it->first), 1.0));
        }
    } else {
        for (; it != it_end; ++it) {
            const char* p = it->first;
            const char* q = it->second;
            while (p < q && *p != ':')
                ++p;
            if (p == q)
                return false;
            context.push_back(make_pair(string(it->first, p - it->first), 
                        atof(string(p + 1, q-p-1).c_str())));
        }
    }
    return true;
}
#endif

// check if all features in a data stream are binary
// it checks for the first non-empty line
// the relative stream position leave unchanged
bool is_binary_feature(const string& file) {
    ifstream is(file.c_str());
    if (!is)
        throw runtime_error("can not open data file to read");

    string line;
    while (getline(is, line)) {
        if (!line.empty()) {
            typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
            boost::char_separator<char> sep(" \t");
            tokenizer tokens(line, sep);
            tokenizer::iterator it = tokens.begin();
            ++it;

            for (;it != tokens.end(); ++it) {
                size_t pos = it->find(':');
                if (pos == string::npos)
                    break;
                try {
                    boost::lexical_cast<double>(it->substr(pos + 1));
                } catch (boost::bad_lexical_cast&) {
                    break;
                }
            }
            return it != tokens.end();
        }
    }
    return true;
}

template <typename Func>
void load_events(const string& file, Func add_event) {
    bool binary_feature = is_binary_feature(file);

    size_t count = 0;
    me_context_type context;
    me_outcome_type outcome;

#if defined(HAVE_SYSTEM_MMAP)
    if (g_use_mmap) {
        MmapFile fm(file.c_str(), "r", 0);
        if (fm.open() == false)
            throw runtime_error("fail to mmap file");

        const char* data = (const char*)fm.addr();
        line_mem_iterator<> line(data, data + fm.size());
        line_mem_iterator<> lend;
        for (; line != lend; ++line) {
            if (line->first == line->second)
                continue; 
            if (!get_sample(line->first, line->second, context,
                        outcome, binary_feature)) {
                char msg[100];
                sprintf(msg, "line [%d] in data file broken.", count);
                throw runtime_error(msg);
            }
            add_event(context, outcome);
            ++count;
            if (count % 1000 == 0) {
                displayA(".");
                if (count % 10000 == 0)
                    displayA(" ");
                if (count % 50000 == 0)
                    display("\t%d samples", count);
            }
        }
    } else {
#endif
        ifstream is(file.c_str());
        if (!is)
            throw runtime_error("can not open data file to read");
        line_stream_iterator<> line(is);
        line_stream_iterator<> lend;
        for (; line != lend; ++line) {
            if (line->empty())
                continue; 
            if (!get_sample(*line, context, outcome, binary_feature)) {
                char msg[100];
                sprintf(msg, "line [%d] in data file broken.", count);
                throw runtime_error(msg);
            }
            add_event(context, outcome);
            ++count;
            if (count % 1000 == 0) {
                displayA(".");
                if (count % 10000 == 0)
                    displayA(" ");
                if (count % 50000 == 0)
                    display("\t%d samples", count);
            }
        }
#if defined(HAVE_SYSTEM_MMAP)
    } // g_use_mmap
#endif

    display("");
}

// perform n-Fold cross_validation, results are printed to stdout
void cross_validation(const string& file, size_t n, int iter, 
        const string& method, double gaussian, bool random) {
    vector<pair<me_context_type, me_outcome_type> > v;
    vector<pair<me_context_type, me_outcome_type> >::iterator it;
    load_events(file, AddEventToVector(v));

    if (v.size() < 5 * n)
        throw runtime_error("data set is too small to perform cross_validation");

    if (random) {
#if HAVE_GETTIMEOFDAY
        timeval t;
        gettimeofday(&t, 0);
        srand48(t.tv_usec);
#endif
        random_shuffle(v.begin(), v.end());
    }

    double total_acc = 0;
    size_t step = v.size() / n;
    for (size_t i = 0; i < n; ++i) {
        MaxentModel m;

        m.begin_add_event();
        m.add_events(v.begin(), v.begin() + i * step);
        m.add_events(v.begin() + (i + 1) * step, v.end());
        m.end_add_event();
        m.train(iter, method, gaussian); 

        size_t correct = 0;
        size_t count = 0;
        for (it = v.begin() + i * step; it != v.begin() + (i + 1) * step;
                ++it) {
            if (m.predict(it->first) == it->second)
                ++correct;
            ++count;
        }

        double acc = double(correct)/count;

        cout << "Accuracy[" << i + 1 << "]: " << 100 * acc  << "%" << endl;
        total_acc += acc;
    }
    cout << n << "-fold Cross Validation Accuracy: " <<
        total_acc * 100 / n << "%" << endl;
}

void predict(const MaxentModel& m, const string& in_file,
        const string& out_file, bool output_prob) {
    ifstream input(in_file.c_str());
    if (!input)
        throw runtime_error("unable to open data file to read");

    ostream* output = 0;
    ofstream os;
    if (!out_file.empty()) {
        os.open(out_file.c_str());
        if (!os)
            throw runtime_error("unable to open data file to write");
        else
            output = &os;
    }

    bool binary_feature = is_binary_feature(in_file);
    size_t correct = 0;
    size_t count = 0;
    me_context_type context;
    me_outcome_type outcome;
    vector<pair<me_outcome_type, double> > outcomes;
    string prediction;
    line_stream_iterator<> line(input);
    line_stream_iterator<> lend;
    if (output)
        output->precision(10);
    for (; line != lend; ++line) {
        if (!get_sample(*line, context, outcome, binary_feature)) {
            char msg[100];
            sprintf(msg, "line [%d] in data file broken.", count);
            throw runtime_error(msg);
        }

        m.eval_all(context, outcomes, false);
        size_t max_i = 0;
        for (size_t i = 1; i < outcomes.size(); ++i)
            if (outcomes[i].second > outcomes[max_i].second)
                max_i = i;

        prediction = outcomes[max_i].first;

        if (prediction == outcome)
            ++correct;

        if (output) {
            if (output_prob) {
                for (size_t i = 0; i < outcomes.size(); ++i)
                    *output << outcomes[i].first << '\t'
                        << outcomes[i].second << '\t';
                *output << endl;
            } else {
                *output << prediction << endl;
            }
        }

        ++count;
    }
    cout << "Accuracy: " << 100.0 * correct/count << "% (" << 
        correct << "/" << count << ")" << endl;
}

int main(int argc,char* argv[]) {
    try {
        gengetopt_args_info args_info;

        /* let's call our CMDLINE Parser */
        if (cmdline_parser (argc, argv, &args_info) != 0)
            return EXIT_FAILURE;

        string model_file;
        string in_file;
        string out_file;
        string test_file;
        string heldout_file;

        if (args_info.model_given)
            model_file = args_info.model_arg;

        if (args_info.output_given) {
            out_file = args_info.output_arg;
        }

        maxent::verbose = args_info.verbose_flag;

        if (args_info.inputs_num > 0) {
            in_file = args_info.inputs[0];
            if (args_info.inputs_num > 1)
                test_file = args_info.inputs[1];
        } else {
            cmdline_parser_print_help();
            return EXIT_FAILURE;
        }

        string estimate = "lbfgs";
        if (args_info.gis_given)
            estimate = "gis";

        if (args_info.heldout_given)
            heldout_file = args_info.heldout_arg;

        g_use_mmap = (args_info.nommap_flag) ? false : true;

        if (args_info.cv_arg > 0) {
            cross_validation(in_file, args_info.cv_arg, args_info.iter_arg,
                    estimate, args_info.gaussian_arg,
                    args_info.random_flag);
        } else if (args_info.predict_given) {
            if (model_file == "")
                throw runtime_error("model name not given");

            MaxentModel m;
            m.load(model_file);
            predict(m, in_file, out_file, args_info.detail_flag);
        } else { // training mode
            MaxentModel m;
            m.begin_add_event();
            display("Loading training events from %s", in_file.c_str());
            load_events(in_file, AddEventToModel(m));
            if (!heldout_file.empty()) {
                display("Loading heldout events from %s", heldout_file.c_str());
                load_events(heldout_file, AddHeldoutEventToModel(m));
            }
            m.end_add_event(args_info.cutoff_arg);
            m.train(args_info.iter_arg, estimate, args_info.gaussian_arg);

            if (!test_file.empty())
                predict(m, test_file, out_file, args_info.detail_flag);

            if (model_file != "")
                m.save(model_file, args_info.binary_flag);
            else
                cerr << "Warning: model name not given, no model saved" << endl;
        }
    } catch (std::bad_alloc& e) {
        cerr << "std::bad_alloc caught: out of memory" << endl;
        return EXIT_FAILURE;
    } catch (std::runtime_error& e) {
        cerr << "std::runtime_error caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (std::exception& e) {
        cerr << "std::exception caught:" << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception caught!" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "maxent.h"

using namespace std;
using namespace maxent;

vector<string> split(const string & line) 
{
    vector<string> vs;
    istringstream is(line);
    string w;
    while (is >> w)
    {
        vs.push_back(w);
    }
    return vs;
}

void train(ME_Model & model, const string & input, const string & model_path)
{
    ifstream ifile(input.c_str());

    if (!ifile)
    {
        cerr << "error: cannot open " << input << endl; 
        exit(1);
    }

    string line;
    while (getline(ifile, line))
    {
        vector<string> vs = split(line);
        ME_Sample mes(vs, true);
        model.add_training_sample(mes);
    }

    model.train();
    model.save(model_path);
}

void exit_with_help()
{
    cerr << "Usage: train_exe [options] training_set_file model_file" << endl
         << "options:" << endl
         << "-s type : set type of solver (default 2)" << endl
         << "\t0 -- L1 Regularized maxent optimized with owlqn" << endl
         << "\t1 -- L1 Regularized maxent optimized with sgd"   << endl
         << "\t2 -- L2 Regularized maxent optimized with lbfgs" << endl
         << "-ci coef : set regularized coefficient. $i could be"
            << "1/2 to indicate L1/L2 regularizer (default 1.0)" << endl
         << "-i sgd_iter : set number of iteration for SGD (default 30)" << endl
         << "-e sgd_eta0 : set initial learning rate for SGD (default 1)" << endl
         << "-a sgd_alpha : set decay coefficient of learning rate for SGD" << endl
         << "-h heldout : set number of heldout data (default 0)" << endl << endl
         << "Example: ./me_train -s 0 -h 100 sample.train model" << endl;
    exit(1);
}

void parse_cmd_line(
        int argc, char** argv,
        ME_Parameter & param,
        string & input_file_name,
        string & model_file_name)
{
    // default settings
    int i;

    for (i = 1; i < argc; ++i)
    {
        if (argv[i][0] != '-') break;
        if (++i > argc)
            exit_with_help();

        switch (argv[i-1][1])
        {
            case 's':
                switch (atoi(argv[i]))
                {
                    case 0: param.solver_type = L1_OWLQN; break;
                    case 1: param.solver_type = L1_SGD;   break;
                    case 2: param.solver_type = L2_LBFGS; break;
                    default:
                        cerr << "unknown solver type: " << argv[i-1]
                             << " " << argv[i] << endl;
                        exit_with_help();
                        break;
                }
                break;

            case 'c':
                switch (atoi(&argv[i-1][2]))
                {
                    case 1: param.l1_reg = atof(argv[i]); break;
                    case 2: param.l2_reg = atof(argv[i]); break;
                    default:
                        cerr << "unknown regularizer: -" << argv[i-1] << endl;
                        exit_with_help();
                        break;
                }
                break;

            case 'i':
                param.sgd_iter = atoi(argv[i]);
                break;

            case 'e':
                param.sgd_eta0 = atof(argv[i]);
                break;

            case 'a':
                param.sgd_alpha = atof(argv[i]);
                break;

            case 'h':
                param.nheldout = atoi(argv[i]);
                break;

            default:
                cerr << "unknown option: -" << argv[i-1][1] << endl;
                exit_with_help();
                break;
        }
    }

    if (i >= argc) exit_with_help();
    input_file_name = argv[i];

    ++i;
    if (i >= argc) exit_with_help();
    model_file_name = argv[i];
}

int main(int argc, char** argv)
{
    /*
     * Params: input_file_name, model_file_name
     * Options: solver_type, regularize_coef, heldout
     *
     */
    string input_path;
    string model_path;

    ME_Parameter param;

    parse_cmd_line(argc, argv, param, input_path, model_path);

    ME_Model m(param);

    // cerr << "Training" << endl;
    // train(m, "./srl_data/train.noun.txt", reg, reg_coef, opt, m_path);
    train(m, input_path, model_path);

    return 0;

    // cout << "[Testing]" << endl;
    // validate(m,  "./srl_data/devel.noun.txt");
}

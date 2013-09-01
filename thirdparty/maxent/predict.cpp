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

void validate(const ME_Model & model, 
              const string & input_file_name,
              const string & output_file_name)
{
    ifstream ifile(input_file_name.c_str());
    ofstream ofile(output_file_name.c_str());

    if (!ifile)
    {
        cerr << "error: cannot open " << input_file_name << endl;
        exit(1);
    }
    if (!ofile)
    {
        cerr << "error: cannot open " << output_file_name << endl;
        exit(1);
    }

    int n_correct = 0;
    int n_total   = 0;

    string line;
    while (getline(ifile, line))
    {
        vector<string> vs = split(line);
        ME_Sample mes(vs, true);
        model.predict(mes);

        ofile << mes.label << endl;

        if (mes.label == vs[0])  n_correct++;
        n_total++;
    }

    double accuracy = (double)n_correct / n_total;
    cout << "accuracy = " << n_correct << " / " << n_total
         << " = " << accuracy << endl;
}

void exit_with_help()
{
    cerr << "Usage: test_exe model_file input_file output_file" << endl;

    exit(1);
}

int main(int argc, char** argv)
{
    /*
     * Params: model_file_name, input_file_name, output_file_name
     *
     */
    if (argc < 4)
    {
        exit_with_help();
    }

    string model_path  = argv[1];
    string input_path  = argv[2];
    string output_path = argv[3];

    ME_Model m;
    m.load(model_path);

    validate(m, input_path, output_path);

    return 0;
}

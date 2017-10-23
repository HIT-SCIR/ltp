/*
 * File name    : DepSRL.h
 * Author       : msmouse
 * Create Time  : 2009-09-19
 * Remark       : feature selection, post-process, result generation
 *
 * Updates by   : jiangfeng
 *
 */

#ifndef _DEP_SRL_
#define _DEP_SRL_

#include <vector>
#include <utility>
#include <string>

#include "Pi/config/SrlPiConfig.h"
#include "Srl/config/SrlSrlConfig.h"
#include "Pi/model/SrlPiModel.h"
#include "Srl/model/SrlSrlModel.h"
#include "structure/WordEmbBuilder.h"
#include "mutex"

class DepSRL {

    public:
        DepSRL() {}

        ~DepSRL()
        {
            if(m_resourceLoaded) {
                ReleaseResource();
            }
        }

        /* Load necessary resources into memory
         *  -> semantic role classifier
         *  -> predicate recognizer
         */
        int LoadResource(const string &ConfigDir = "ltp_data/srl_data/");

        /* Release all resources
         */
        int ReleaseResource();

        /* Produce DepSRL result for a sentence
         */
        int GetSRLResult(
                const vector<string> &words,
                const vector<string> &POSs,
                const vector< pair<int, string> > &parse,
                vector< pair< int, vector< pair<string, pair< int, int > > > > > &vecSRLResult
                );


    private:

        /* 3.form the SRL result, based on predict result from maxent model
         */
        int FormResult(
                const vector<string> &words,
                const vector<string> &POSs,
                const vector<int>    &VecAllPredicates,
                SrlPiSample& sentence,
                vector< pair< int, vector< pair< string, pair< int, int > > > > > &vecSRLResult
                );

        void ProcessOnePredicate(
                const vector<string>& vecWords,
                const vector<string>& vecPos,
                int intPredicates,
                const vector<string>& args,
                const vector< pair<int, int> >& childArea,
                vector< pair< string, pair< int, int > > > &vecResultForOnePredicate
                );

    private:
        /*-----for form result-----*/
        void GetChildArea(SrlPiSample &sentence, vector<pair<int, int>> &childArea);
        void ProcessCollisions(int intPredicates, vector< pair< string, pair< int, int > > > &ResultForOnePredicate);
        /*-----for form result-----*/
    private:
        /*-------------------------for post process-----------------------------*/

        void QTYArgsProcess(const vector<string>& vecPos, vector< pair<string, pair<int, int> > >& results) const;

        /*-------------------------for post process-----------------------------*/

    private:
        bool             m_resourceLoaded;
        SrlPiBaseConfig  piConfig;
        SrlSrlBaseConfig srlConfig;
        SrlSrlModel * srl_model;
        PiModel * pi_model;
        unordered_map<string, vector<float>> embedding;
        static std::mutex mtx; // to fix dynet single CG constrain.
    private:
        void manageConfigPath(ModelConf &config, const string &dirPath);

};

#endif

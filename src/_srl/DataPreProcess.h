/*
 * File Name     : DataPreProcess.h
 * Author        : Frumes
 * Create Time   : 2006Äê12ÔÂ31ÈÕ
 * Project Name  : NewSRLBaseLine
 * Remark        : get data from IR-LTP platform
 *
 */


#ifndef __LTP_PROPRECESS__
#define __LTP_PROPRECESS__

#include "MyTree.h"

class DataPreProcess
{
    public:
        DataPreProcess(const LTPData* ltpData);
        ~DataPreProcess();

    private:
        void BuildStruct(const LTPData* ltpData);
        void DestroyStruct();
        void MapNEToCons();

    private:
        string SingleNE(int intBeg, int intEnd) const;
        string ExternNE(int intBeg, int intEnd) const;

    public:

        const LTPData*  m_ltpData;
        vector<string>  m_vecNE;
        MyTree*         m_myTree;
        int             m_intItemNum; //the Chinese word numbers after segmentation
};

#endif


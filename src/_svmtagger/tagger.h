/*
 * Copyright (C) 2004 Jesus Gimenez, Lluis Marquez and Senen Moya
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifndef TAGGER_H
#include "marks.h"
#include "hash.h"
#include "list.h"
#include "weight.h"
#include "dict.h"
#include "stack.h"
#include "swindow.h"
#include "common.h"

#define BUSHOUNUM 215
#define HANZINUM  20902
struct models_t{
    weightRepository *wr,*wr2,*wrUnk,*wrUnk2;
    hash_t *bias,*bias2,*biasUnk,*biasUnk2;
    simpleList featureList;
    simpleList featureListUnk;
};


class tagger
{
    private:
        //Flow Control
        char flow[10];
        int  taggerStrategy,taggerNumLaps,taggerWinIndex,taggerWinLength;
        float  taggerKFilter,taggerUFilter;
        char taggerBackupDict[150],taggerModelName[150];
        int bushou[BUSHOUNUM];     //新加存放部首

        struct stack_t *stk;
        models_t  *taggerModelList;
        models_t  *taggerModelRunning;
        dictionary *d;
        swindow *sw;
        weight_node_t *weightUnk;
        hash_t *hashBs;            //新加，汉字与unicode的对应

        void init_bs(const char*szResPath);         //新加，初始化部首数组
        void init_hashBs(const char*szResPath);     //新加，初始化汉字与unicode 对应表
        int taggerRightSense();
        int taggerLeftSense();

        void taggerSumWeight(weightRepository *wRep,hash_t *bias,weight_node_t *weight,int numMaybe, int *max);
        void taggerGenerateScore(nodo *elem,int direction);

        weight_node_t *taggerCreateWeightNodeArray(int numMaybe,int index);
        weight_node_t *taggerInitializeWeightNodeArray(int numMaybe,weight_node_t *w);
        weight_node_t *taggerCreateWeightUnkArray(char *name);
        hash_t *taggerCreateBiasHash(char *name);
        void taggerLoadModels(models_t *model, int taggerNumModel,const char *szResPath);

        void taggerStadistics(int numWords, int numSentences, double realTime,double usrTime, double sysTime);
        void taggerShowVerbose(int num,int isEnd);


        int taggerRightSenseSpecialForUnknown();
        int taggerLeftSenseSpecialForUnknown();
        void taggerDoNormal(int *numWords, int *numSentences);
        void taggerDoSpecialForUnknown(int *numWords, int *numSentences);
        void taggerDoNTimes(int *numWords, int *numSentences,int laps);

    public:
        void taggerRun();
        void taggerPutFlow(char *inFlow);
        void taggerPutBackupDictionary(char *dictName);
        void taggerPutStrategy(int num);
        void taggerPutWinLength(int l);
        void taggerPutWinIndex(int i);
        void taggerPutKWeightFilter(float kfilter);
        void taggerPutUWeightFilter(float ufilter);
        void taggerInit(const char *szResPath);
            void taggerInitSw(char **infile, char **outfile,int i);
        tagger(char *model,const char*szResPath);
        ~tagger();
};


#define TAGGER_H
#endif

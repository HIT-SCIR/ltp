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

#ifndef DICT_H

#define TAMTXT 100

struct dataDict{
    char wrd[TAMTXT];
    int  numWrd;
    int  numMaybe;
    simpleList maybe;
};

struct infoDict{
    char txt[TAMTXT];
    int  num;
};

class  dictionary
{
    private:
        hash_t d;
        //FILE *in;

        // FILE *openFile(char *name, char mode[]);
        void dictLoad(FILE *in);
        void dictCreate(FILE *f,int offset, int limit);
        void dictIncInfo(dataDict *elem, char *pos);

        int readInt(FILE *in);
        infoDict *readData(FILE *in);

    public:
        void dictAddBackup(char *name);
        int getElement(char *key);
        char *getElementWord(int ptr);
        int getElementNumWord(int ptr);
        int getElementNumMaybe(int ptr);
        simpleList *getElementMaybe(int ptr);
        char *getMFT(int w);
        char *getAmbiguityClass(int w);

        hash_t *dictFindAmbP(int *numPOS);
        hash_t *dictFindUnkP(int *numPOS);
        void dictRepairFromFile(char *fileName);
        void dictRepairHeuristic(float dratio);

        dictionary(char *name, char *backup);
        dictionary(char *name);
        dictionary(char *name,int limInf, int limSup);
        ~dictionary();

        void dictWrite(char *outName);
};

#define DICT_H
#endif

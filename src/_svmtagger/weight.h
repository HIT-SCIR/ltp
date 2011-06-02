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

#ifndef WEIGHT_H
#include <stdio.h>

typedef struct weight_node_t {
    char pos[5];
    long double data;
} weight_node_t;

class  weightRepository
{
    private:
        hash_t wr;

        char *wrGetMergeInput(hash_t *tptr);
        FILE *openFile(char *name, char mode[]);
        void wrReadMergeModel(FILE *in,float filter);
        char wrSaltarBlancs(FILE *in,char c,int jmp);
        void wrAddPOS(int obj, char* pos, long double weight);
    public:
        long double wrGetWeight(char *feature,char *pos);
        void wrAdd(char *feature, char* pos, long double weight);
        void wrWrite(char *outName);
        void wrWriteHash(hash_t *tptr,FILE *f,char separador);
        weightRepository(char *fileName,float filter);
        weightRepository();
        ~weightRepository();
};

#define WEIGHT_H
#endif

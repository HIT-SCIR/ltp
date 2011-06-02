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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hash.h"
#include "weight.h"

float absolut(float f)
{
    if (f < 0) return (-1)*f;
    else return f;
}

/***********************************************************

  A WeightRepository Object is a hash with weight_struct_t objects.
  We have the key and another hash inside a weight_struct_t object.
  This hash contains weight_node_t with the POS and weight.

 ***********************************************************/

//Types for WeightRepository

class weight_struct_t {
    public:
        char key[100];
        hash_t *hash;

        ~weight_struct_t()
        {
            hash_destroy(hash);
            delete[] key;
        }
};



/***********************************************************/

char weightRepository::wrSaltarBlancs(FILE *in, char c,int jmp)
{
    while ((c==':') || (c==' ') || (c=='\n' && jmp==1)) c=fgetc(in);
    return c;
}


void weightRepository::wrReadMergeModel(FILE *in,float filter)
{
    char c=fgetc(in),key[200],value[100],*endptr;
    weight_struct_t *obj;
    char garbage[512];

    strcpy(key,"");
    strcpy(value,"");
    while  (!feof(in))
    {
        //c = fgetc(in);
        if (c!='#')
        {
            obj = new weight_struct_t;
            strcpy(obj->key,"");

            while (c!=' ')
            { 
                sprintf(obj->key,"%s%c",obj->key,c);
                c=fgetc(in);
            }

            //fprintf(stderr,"%s\t",obj->key);

            obj->hash  = new hash_t;
            hash_init(obj->hash,10);

            while ((c!='\n') && (!feof(in)))
            {
                weight_node_t *w = new weight_node_t;

                c = wrSaltarBlancs(in,c,0);
                strcpy(w->pos,""); strcpy(value,"");
                while ((c!=':') && (!feof(in)))
                { sprintf(w->pos,"%s%c",w->pos,c);
                    c=fgetc(in);
                }

                c = wrSaltarBlancs(in,c,0);

                while ((c!=' ') && (c!='\n') && (!feof(in)) )
                { sprintf(value,"%s%c",value,c);
                    c=fgetc(in);
                }

                //w->data=strtod(value,&endptr); //atof(value);
                //w->data=_atold(value);
                w->data=atof(value);
                if ( absolut(w->data) > absolut(filter) )  
                    hash_insert(obj->hash,w->pos,(long) w);
                else delete w;
            }

            c = wrSaltarBlancs(in,c,1);

            hash_insert(&wr,obj->key, (long) obj);
        }
        else 
        { fgets(garbage,512,in); //while(c=fgetc(in)!='\n');
            c = fgetc(in);
        }
    }
}


long double weightRepository::wrGetWeight(char *feature,char *pos)
{
    int h = hash_lookup(&wr,feature);
    if (h!=HASH_FAIL)
    {
        weight_struct_t *obj = (weight_struct_t *)h;
        int w = hash_lookup(obj->hash,pos);

        if (w!=HASH_FAIL)
        { weight_node_t *ret = (weight_node_t *)w;
            return ret->data;
        }
    }
    return 0;
}

weightRepository::weightRepository(char *fileName,float filter)
{
    FILE *in;
    if ((in = fopen(fileName, "rt"))== NULL)
    {
        fprintf(stderr, "Error opening weightRepository: %s. It's going to work without it.\n",fileName);
        exit(0);
    }
    hash_init(&wr,10000);
    wrReadMergeModel(in,filter);
    fclose(in);
}

weightRepository::weightRepository()
{
    hash_init(&wr,10000);
}

weightRepository::~weightRepository()
{
    hash_destroy(&wr);
}

/*******************************************************/

void weightRepository::wrAddPOS(int obj, char* pos, long double weight)
{
    weight_struct_t *wst = (weight_struct_t *)obj;
    int x = hash_lookup( wst->hash, pos);

    if (x==HASH_FAIL)
    {
        //Insertamos Nueva POS
        weight_node_t *w = new weight_node_t;
        strcpy(w->pos,pos);
        w->data=weight;
        hash_insert( wst->hash,w->pos,(long) w);
    }
    else
    { //Si POS ya esta, incrementamos el peso
        weight_node_t *wnt = (weight_node_t *)x;
        wnt->data = wnt->data + weight;
    }
}

void weightRepository::wrAdd(char *feature, char* pos, long double weight)
{
    weight_struct_t *obj = (weight_struct_t *)hash_lookup(&wr,feature);

    if ( (long) obj == HASH_FAIL)
    {
        // Creamos nueva entrada en WeightRepository
        obj = new weight_struct_t;
        strcpy(obj->key,feature);
        obj->hash  = new hash_t;
        hash_init(obj->hash,10);
        wrAddPOS((long)obj,pos,weight);
        hash_insert(&wr,obj->key, (long) obj);
    }
    else 		wrAddPOS((long)obj,pos,weight);
}

/*******************************************************/

void weightRepository::wrWrite(char *outName)
{
    //int ret=0;
    weight_struct_t *wst;
    FILE *f;

    if ((f = fopen(outName, "w"))== NULL)
    {
        fprintf(stderr, "Error opening file: %s\n",outName);
        exit(0);
    }

    hash_t *tptr = &wr;

    hash_node_t *node, *last;
    int i;

    for (i=0; i<tptr->size; i++)
    {
        node = tptr->bucket[i];
        while (node != NULL)
        {
            last = node;
            node = node->next;
            //fprintf(f,"%s\n",last->key);
            wst = (weight_struct_t *) last->data;
            //fprintf(f,"%s",wst->key);
            //wrWriteHash(wst->hash,f,' ');
            char *mrg = wrGetMergeInput(wst->hash);
            if (strcmp(mrg,"")!=0) fprintf(f,"%s%s\n",wst->key,mrg);
            delete mrg;
        } //while
    }//for
    fclose (f);
}



char *weightRepository::wrGetMergeInput(hash_t *tptr)
{
    char *out = new char[1000];
    weight_node_t *wnt;
    hash_node_t **old_bucket, *old_hash, *tmp;
    int old_size, h, i;

    old_bucket=tptr->bucket;
    old_size=tptr->size;
    strcpy (out,"");

    for (i=0; i<old_size; i++)
    {
        old_hash=old_bucket[i];
        while(old_hash)
        {
            tmp=old_hash;
            old_hash=old_hash->next;
            wnt = (weight_node_t *) tmp->data;
            //fprintf(f,"%c%s %2.10f",separador,wnt->pos,(float)wnt->data);
            if ((float)wnt->data!=0) sprintf(out,"%s %s:%.17E",out,wnt->pos, (float) wnt->data);
        } //while
    } //for
    return out;
}

void weightRepository::wrWriteHash(hash_t *tptr,FILE *f, char separador)
{
    weight_node_t *wnt;
    hash_node_t **old_bucket, *old_hash, *tmp;
    int old_size, h, i;
    int cont=0;

    old_bucket=tptr->bucket;
    old_size=tptr->size;

    for (i=0; i<old_size; i++)
    {
        old_hash=old_bucket[i];
        while(old_hash)
        {
            tmp=old_hash;
            old_hash=old_hash->next;
            wnt = (weight_node_t *) tmp->data;
            if (separador == '\n' && cont==0) fprintf(f,"%s %2.10f",wnt->pos,(float)wnt->data);
            else 	fprintf(f,"%c%s %2.10f",separador,wnt->pos,(float)wnt->data);
            cont++;
        } /* while */
    } /* for */
}


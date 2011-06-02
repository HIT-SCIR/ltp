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

#include <sys/types.h>
//#include <regex.h>
#include <boost/cregex.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "hash.h"
#include "list.h"
#include "dict.h" 
#include "weight.h"
#include "stack.h"
#include "swindow.h"
#include "er.h"
#include "common.h"
#include "marks.h"

using namespace boost;

/*****************************************************************
 * Config elements
 *****************************************************************/

/*****************************************************************
 * Feature Generation
 *****************************************************************/

int swindow::checkHanzi(char* wrd)
{
    int flag = 1;

    if(erLookRegExp2(&erStartCap,wrd))
        flag = 0;
    else if(erLookRegExp2(&erStartLower,wrd))
        flag = 0;
    else if(erLookRegExp2(&erStartNumber,wrd))
        flag = 0;
    else if(erLookRegExp2(&erAllUp,wrd))
        flag = 0;
    else if(erLookRegExp2(&erAllLow,wrd))
        flag = 0;
    else if(erLookRegExp2(&erContainCap,wrd))
        flag = 0;
    else if(erLookRegExp2(&erContainPeriod,wrd))
        flag = 0;
    else if(erLookRegExp2(&erContainComma,wrd))
        flag = 0;
    else if(erLookRegExp2(&erContainNum,wrd))
        flag = 0;
    else if(erLookRegExp2(&erContainCaps,wrd))
        flag = 0;
    else
        flag = 1;
    return flag;				
}
int swindow::checkEnglish(char* wrd)
{
    int flag = 0;
        
        if(erLookRegExp2(&erStartCap,wrd))
            flag = 1;
        else if(erLookRegExp2(&erStartLower,wrd))
            flag = 1;
        else if(erLookRegExp2(&erAllUp,wrd))
            flag = 1;
        else if(erLookRegExp2(&erAllLow,wrd))
            flag = 1;
        else if(erLookRegExp2(&erContainCap,wrd))
            flag = 1;
        else if(erLookRegExp2(&erContainPeriod,wrd))
            flag = 1;
        else if(erLookRegExp2(&erContainComma,wrd))
            flag = 1;
        else if(erLookRegExp2(&erContainCaps,wrd))
            flag = 1;
        else
            flag = 0;
                return flag;				
}
//寻找部首
int swindow::find(int num,int bushou[],int numbs)
{
    int i;
    int answer;
    for (i = 0; i<numbs ; i++)
    {
        if (num < bushou[i])
        {
            answer=i-1;
            break;
        }
        else if(num == bushou[i])
        {
            answer=i;
            break;
        }
    }
    if(i == numbs)
        answer = bushou[numbs-1];
    //printf("%d\n",answer);
    return answer;	
}

//加入部首前缀化特征
void swindow::winPushPreBushouFeature(char *wrd,stack_t *pila, int longitud, int bushou[], int numbs, hash_t* hashBs)
{
    int len = strlen(wrd);
    int prebushou[6];
    int hashnum;
    char key[3];
    int j = 0;
    int k =0 ;
    int i=0;

    if(checkHanzi(wrd))
    {
        for ( i=0; i< longitud * 2 ; i++)
        {
            if (len > i) 
            {
                key[j] = wrd[i];
                j++;
                if (((i+1)%2 == 0) && (i!=0))       //可能是一个汉字了
                {
                    key[j] = '\0';
                    hashnum = hash_lookup(hashBs, key);
                    prebushou[k] = find(hashnum,bushou,numbs);
                    k++;
                    j = 0;
                }	
            }
            else //如果索取长度大于词长度，不再提取
            {
                break;
            }
        }

        char* feat = new char[strlen(PRE_BUSHOU)+k * sizeof(int)+(k-1) * sizeof(char) + 4];
        strcpy(feat,""); 
        for (i=0; i < k;i++)
        {
            if (i==0)
            {
                if(k == 1)
                    sprintf(feat,"%s%d:%d",PRE_BUSHOU,k,prebushou[i]);
                else
                    sprintf(feat,"%s%d:%d%c",PRE_BUSHOU,k,prebushou[i],'~');
            }	
            else if(i == k-1)
                sprintf(feat,"%s%d",feat,prebushou[i]);
            else
                sprintf(feat,"%s%d%c",feat,prebushou[i],'~');
        }
        //printf("%s\n",feat);
        push(pila,feat);
    }
    //delete[] feat;
    //feat = NULL;
}
//加入部首后缀化特征
void swindow::winPushSufBushouFeature(char *wrd,stack_t *pila, int longitud, int bushou[], int numbs, hash_t* hashBs)
{
    int len = strlen(wrd);
    int prebushou[6];
    int hashnum;
    char key[3];
    int j = 0;
    int k =0 ;
    int i= len - longitud * 2;

    if(checkHanzi(wrd))
    {
        if (i >= 0)
        {
            for ( i=len - longitud * 2; i<= len -1; i++)
            {
                key[j] = wrd[i];
                j++;
                if ((i+1)%2 == 0)       //可能是一个汉字了
                {
                    key[j] = '\0';
                    hashnum = hash_lookup(hashBs, key);
                    prebushou[k] = find(hashnum,bushou,numbs);
                    k++;
                    j = 0;
                }
            }
        } 
        else
        {
            for ( i=0; i<= len -1; i++)
            {
                key[j] = wrd[i];
                j++;
                if ((i+1)%2 == 0)       //可能是一个汉字了
                {
                    key[j] = '\0';
                    hashnum = hash_lookup(hashBs, key);
                    prebushou[k] = find(hashnum,bushou,numbs);
                    k++;
                    j = 0;
                }
            }
        }

        char* feat = new char[strlen(SUF_BUSHOU)+k * sizeof(int)+(k-1) * sizeof(char) + 4];
        strcpy(feat,""); 
        for (i=0; i < k;i++)
        {
            if (i==0)
            {
                if(k == 1)
                    sprintf(feat,"%s%d:%d",SUF_BUSHOU,k,prebushou[i]);
                else
                    sprintf(feat,"%s%d:%d%c",SUF_BUSHOU,k,prebushou[i],'~');
            }
            else if(i == k-1)
                sprintf(feat,"%s%d",feat,prebushou[i]);
            else
                sprintf(feat,"%s%d%c",feat,prebushou[i],'~');
        }
        //printf("%s\n",feat);
        push(pila,feat);
    }

}
//添加重叠特征
void swindow::winPushDoubleFeature(char *wrd,stack_t *pila)
{
    int len = strlen(wrd);
    char *feat = new char[6];
    char key1[3];
    char key2[3];
    char key3[3];
    char key4[3];

    strcpy(feat,"");
    if (len == 6)
    {
        key1[0] = wrd[0];
        key1[1] = wrd[1];
        key1[2] = '\0';
        key2[0] = wrd[2];
        key2[1] = wrd[3];
        key2[2] = '\0';
        key3[0] = wrd[4];
        key3[1] = wrd[5];
        key3[2] = '\0';
        if ((strcmp(key1,key2) == 0)&& ((strcmp(key3,"地") == 0)))
        {
            //strcpy(feat,key1);
            sprintf(feat,"%s%s",feat,key1);
        }
    }
    else if (len == 8)
    {
        key1[0] = wrd[0];
        key1[1] = wrd[1];
        key1[2] = '\0';
        key2[0] = wrd[2];
        key2[1] = wrd[3];
        key2[2] = '\0';
        key3[0] = wrd[4];
        key3[1] = wrd[5];
        key3[2] = '\0';
        key4[0] = wrd[6];
        key4[1] = wrd[7];
        key4[2] = '\0';
        if ((strcmp(key1,key2) == 0) && (strcmp(key3,key4) == 0))
        {
            //strcat(key1,key3);
            //strcpy(feat,key1);
            sprintf(feat,"%s%s",feat,key1);
            sprintf(feat,"%s%s",feat,key3);
        }
        else if ((strcmp(key1,key3) == 0) && (strcmp(key2,key4) == 0))
        {
            //strcat(key1,key2);
            //strcpy(feat,key1);
            sprintf(feat,"%s%s",feat,key1);
            sprintf(feat,"%s%s",feat,key2);
        }
    }
    else
        strcpy(feat,"NULL");
    push(pila,feat);
    //delete[] feat;
    //feat = NULL;
}
void swindow::winPushStartWithLowerFeature(char *wrd,stack_t *pila)
{
    int startlower=0;

    //Comienza por Minuscula?
    if (erLookRegExp2(&erStartLower,wrd)) 
    { 
        startlower = 1;

        char *feat = new char[strlen(START_LOWER_MARK)+4];
        sprintf(feat,"%s:%d",START_LOWER_MARK,startlower);

        push(pila,feat);
    }
}

void swindow::winPushStartWithNumberFeature(char *wrd,stack_t *pila)
{
    int startnumber=0;

    //Comienza por Numero?
    if (erLookRegExp2(&erStartNumber,wrd)) 
    {
        startnumber = 1;
        char *feat = new char[strlen(START_NUMBER_MARK)+4];
        sprintf(feat,"%s:%d",START_NUMBER_MARK,startnumber);

        push(pila,feat);
    }
}

void swindow::winPushSuffixFeature(char *wrd, struct stack_t *pila,int longitud)
{
    //Obtenemos la longitud de la palabra
    //char *feat = new char[longitud+6];
    char *feat = new char[strlen(PREFIX_MARK) + longitud * sizeof(char) + 4];
    int len = strlen(wrd);
    //char *suf = new char[longitud+1];
    char *suf = new char[longitud * sizeof(char) + 2];
    //int a=0;

    strcpy(suf,"");
    for (int i=len-longitud;  i<=len-1; i++)
    {      
        if (i>=0) sprintf(suf,"%s%c",suf,wrd[i]);
        else sprintf(suf,"%s~",suf); 
    }

    sprintf(feat,"%s%d:%s",SUFFIX_MARK,longitud,suf);
    push(pila,feat);
    //delete[] feat;  //modify wlj
    delete[] suf;  //modify wlj
    //feat = NULL;
    suf = NULL;
}

/*
 * void winPushPreffixFeatures ( char *wrd, struct stack_t *pila, int longitud)
 * esta funcion creara las "features" para la palabra desconocida
 * <wrd> y las apilara en en el parametro <pila>
 */
void swindow::winPushPrefixFeature(char *wrd, struct stack_t *pila,int longitud)
{
    //Obtenemos la longitud de la palabra
    //char *feat = new char[6+longitud];
    char *feat = new char[strlen(SUFFIX_MARK) + longitud * sizeof(char) + 4];
    int len = strlen(wrd);
    // char *pref = new char[longitud+1];
    char *pref = new char[longitud * sizeof(char) + 2];

    strcpy(pref,"");
    for (int i=0; i<longitud; i++)
    {
        if (len > i) sprintf(pref,"%s%c",pref,wrd[i]);
        else /*if (i > len-1  )*/  sprintf(pref,"%s~",pref);
    }

    sprintf(feat,"%s%d:%s",PREFIX_MARK,longitud,pref);
    push(pila,feat);   
    //delete[] feat;  //modify wlj
    delete[] pref;  //modify wlj
    //feat = NULL;
    pref = NULL;
}


void swindow::winPushStartWithCapFeature(char *wrd, struct stack_t *pila)
{
    int startcap=0;
    //Comienza por Mayuscula?
    if (erLookRegExp2(&erStartCap,wrd)) 
    {
        startcap = 1;
        char *feat = new char[strlen(START_CAPITAL_MARK)+4];
        sprintf(feat,"%s:%d",START_CAPITAL_MARK,startcap);
        push(pila,feat);

    }
}

void swindow::winPushAllUpFeature(char *wrd,stack_t *pila)
{
    int allup=0;

    //Esta toda la palabra en mayusculas?
    if (erLookRegExp2(&erAllUp,wrd)) 
    {
        allup = 1;
        char *feat = new char[strlen(ALL_UPPER_MARK)+4];  //modify wlj
        sprintf(feat,"%s:%d",ALL_UPPER_MARK,allup);
        push(pila,feat);

    }
}

void swindow::winPushAllLowFeature(char *wrd,stack_t *pila)
{
    int alllow = 0;
    //Esta toda la palabra en minusculas?
    if (erLookRegExp2(&erAllLow,wrd)) 
    {
        alllow = 1;
        char *feat = new char[strlen(ALL_LOWER_MARK)+4]; //modify wlj
        sprintf(feat,"%s:%d",ALL_LOWER_MARK,alllow);
        push(pila,feat);

    }
}

void swindow::winPushContainCapFeature(char *wrd, stack_t *pila)
{
    int containcap = 0;
    if (erLookRegExp2(&erContainCap,wrd)) 
    { 
        containcap = 1;
        char *feat = new char[strlen(CONTAIN_CAP_MARK)+4]; //modify
        sprintf(feat,"%s:%d",CONTAIN_CAP_MARK,containcap);
        push(pila,feat);

    }
}

void swindow::winPushContainCapsFeature(char *wrd, stack_t *pila)
{
    int containcaps = 0;
    if (erLookRegExp2(&erContainCaps,wrd)) 
    {
        containcaps = 1;
        char *feat = new char[strlen(CONTAIN_CAPS_MARK)+4]; //modify
        sprintf(feat,"%s:%d",CONTAIN_CAPS_MARK,containcaps);
        push(pila,feat);

    }
}

void swindow::winPushContainPeriodFeature(char *wrd, stack_t *pila)
{
    int containperiod = 0;
    //Contiene un punto?
    if (erLookRegExp2(&erContainPeriod,wrd)) 
    {
        containperiod = 1;
        char *feat = new char[strlen(CONTAIN_PERIOD_MARK)+4]; //modify
        sprintf(feat,"%s:%d",CONTAIN_PERIOD_MARK,containperiod);
        push(pila,feat);

    }
}

void swindow::winPushContainCommaFeature(char *wrd, stack_t *pila)
{
    int containcomma = 0;
    //Contiene un punto?
    if (erLookRegExp2(&erContainComma,wrd)) 
    {
        containcomma = 1;
        char *feat = new char[strlen(CONTAIN_COMMA_MARK)+4]; //modify
        sprintf(feat,"%s:%d",CONTAIN_COMMA_MARK,containcomma);
        push(pila,feat);

    }
}

void swindow::winPushContainNumFeature(char *wrd, stack_t *pila)
{
    int containnum = 0;
    //Contiene un numero?
    if (erLookRegExp2(&erContainNum,wrd)) 
    {  
        containnum = 1;
        //char *feat = new char[5];
        char *feat = new char[strlen(CONTAIN_NUMBER_MARK)+4];//modify wlj
        sprintf(feat,"CN:%d",containnum);
        push(pila,feat);

    }
}

void swindow::winPushMultiwordFeature(char *wrd, stack_t *pila)
{
    int multiword = 0;
    //Es una palabra multiple?
    if (erLookRegExp2(&erMultiWord,wrd)) 
    {
        multiword = 1;
        char *feat = new char[strlen(MULTIWORD_MARK)+4];  //modify
        sprintf(feat,"MW:%d",multiword);
        push(pila,feat);

    }
}

void swindow::winPushLetterFeature(char *wrd , stack_t *pila, int position,int where)
{
    //char *feature = new char[12];
    char *feature = new char[strlen(CHAR_A_MARK) + sizeof(char) * 1 + 4];          //modify
    if (COUNTING_FROM_END==where) 
    {
        sprintf(feature,"%s%d:%c",CHAR_Z_MARK,position,wrd[strlen(wrd)-position]);  
    }
    else 
    {
        sprintf(feature,"%s%d:%c",CHAR_A_MARK,position,wrd[position-1]);
    }

    push (pila,feature);
    //delete[] feature;
    // feature = NULL;
}

void swindow::winPushLenghtFeature(char *wrd, stack_t *pila)
{
    //Obtenemos la longitud de la palabra
    int len = strlen(wrd);

    //Longitud de la palabra
    char *feat = new char[strlen(LENGTH_MARK)+4];  //modify
    //char *feat = new char[6]; //modify wlj
    sprintf(feat,"%s:%d",LENGTH_MARK,len);
    push(pila,feat);
    //delete[] feat;  //modify wlj
    //feat = NULL;
}


/*
 * void winPushUnkownoFeatures ( char *wrd, struct stack_t *pila)
 * esta funcion creara las "features" para la palabra desconocida
 * <wrd> y las apilara en en el parametro <pila>
 */
void swindow::winPushUnknownFeatures(char *wrd, struct stack_t *pila)
{
    int startcap=0,allup=0,alllow=0,wordlength=0,containnum=0,multiword=0,containcap=0,containcaps=0,containperiod=0;

    //Obtenemos la longitud de la palabra
    int len = strlen(wrd);
    char ant[10]="";

    //Creamos el prefijo de longitud 2
    char *feat = new char[5];
    if (len > 1) sprintf(ant,"%c%c",wrd[0],wrd[1]);
    else sprintf(ant,"%c~",wrd[0]);
    sprintf(feat,"a2:%s",ant);
    push(pila,feat);

    //Generamos el prefijo de longitud 3
    feat = new char[6];
    if (len > 2) sprintf(ant,"%c%c%c",wrd[0],wrd[1],wrd[2]);
    else sprintf(ant,"%s~",ant);
    sprintf(feat,"a3:%s",ant);
    push(pila,feat);

    //Generamos el prefijo de longitud 4
    feat = new char[7];
    if (len > 3) sprintf(ant,"%c%c%c%c",wrd[0],wrd[1],wrd[2],wrd[3]);
    else sprintf(ant,"%s~",ant);
    sprintf(feat,"a4:%s",ant);
    push(pila,feat);

    //Generamos el sufijo de longitud 2
    feat = new char[6];
    if (len > 1) sprintf(feat,"z2:%c%c",wrd[len-2],wrd[len-1]);
    else sprintf(feat,"z2:~%c","",wrd[len-1]);
    push(pila,feat);

    //generamos el sufijo de longitud 3
    feat = new char[7];
    if (len > 2) sprintf(feat,"z3:%c%c%c",wrd[len-3],wrd[len-2],wrd[len-1]);
    else if (len > 1) sprintf(feat,"z3:~%c%c",wrd[len-2],wrd[len-1]);
    else sprintf(feat,"z3:~~%c",wrd[len-1]);
    push(pila,feat);

    //generamos el sufijo de longitud 4
    feat = new char[8];
    if (len > 3) sprintf(feat,"z4:%c%c%c%c",wrd[len-4],wrd[len-3],wrd[len-2],wrd[len-1]); //strcpy(prefix4,substr(wrd, 0, 4));
    else if (len > 2) sprintf(feat,"z4:~%c%c%c",wrd[len-3],wrd[len-2],wrd[len-1]);
    else if (len > 1) sprintf(feat,"z4:~~%c%c",wrd[len-2],wrd[len-1]);
    else sprintf(feat,"z4:~~~%c",wrd[len-1]);
    push(pila,feat);

    //Comienza por Mayuscula?
    if (erLookRegExp2(&erStartCap,wrd)) startcap = 1;
    feat = new char[3];
    sprintf(feat,"A:%d",startcap);
    push(pila,feat);

    //Esta toda la palabra en mayusculas?
    if (erLookRegExp2(&erAllUp,wrd)) allup = 1;
    feat = new char[4];
    sprintf(feat,"AA:%d",allup);
    push(pila,feat);

    //Esta toda la palabra en minusculas?
    if (erLookRegExp2(&erAllLow,wrd)) alllow = 1;
    feat = new char[4];
    sprintf(feat,"aa:%d",alllow);
    push(pila,feat);

    //Longitud de la palabra
    feat = new char[4];
    sprintf(feat,"L:%d",len);
    push(pila,feat);

    if (erLookRegExp2(&erContainCap,wrd)) containcap = 1;
    feat = new char[4];
    sprintf(feat,"CA:%d",containcap);
    push(pila,feat);

    if (erLookRegExp2(&erContainCaps,wrd)) containcaps = 1;
    feat = new char[5];
    sprintf(feat,"CAA:%d",containcaps);
    push(pila,feat);

    //Contiene un punto?
    if (erLookRegExp2(&erContainPeriod,wrd)) containperiod = 1;
    feat = new char[5];
    sprintf(feat,"CP:%d",containperiod);
    push(pila,feat);

    //Contiene un numero?
    if (erLookRegExp2(&erContainNum,wrd)) containnum = 1;
    feat = new char[5];
    sprintf(feat,"CN:%d",containnum);
    push(pila,feat);

    //Es una palabra multiple?
    if (erLookRegExp2(&erMultiWord,wrd))  multiword = 1;
    feat = new char[6];
    sprintf(feat,"MW:%d",multiword);
    push(pila,feat);

    //Letra por la que empieza la palabra
    feat = new char[6];
    sprintf(feat,"c1:%c",wrd[0]);
    push(pila,feat);

    //Letra por la que acaba la palabra
    feat = new char[6];
    sprintf(feat,"cn:%c",wrd[len-1]); //charn = wrd[len-1]; //substr(wrd, len-1, 1);
    push(pila,feat);
}

/*
 * void winPushSwnFeature (struct stack_t *pila)
 * Recibe como parametro <pila>, donde se apilara la "feature"
 * Swn.Swn es el elemento final de frase que puede ser
 * ! ? o .
 */
void swindow::winPushSwnFeature(struct stack_t *pila)
{
    char *feature = new char[10];
    //int len = strlen(last->wrd);
    //char *feature = new char[strlen(SLASTW) + sizeof(char)* len + 4];

    sprintf(feature,"Swn:%s",last->wrd);
    //printf("%s\n",last->wrd);
    push(pila,feature);
    //delete[] feature;  //modify wlj
    //feature = NULL;
}


/*
 * void winPushAmbiguityFeature(void *ptr, dictionary *d, stack_t *pila, int direction)
 * Genera el atributo que representa la ambiguedad de una palabra.
 * Recibe como parametros:
 *      ptr, que es un puntero a un nodo de la lista de atributos (nodo_feature_list)
 *           aunque se recibe como un void*.
 *      d,   es el diccionario con el que estamos trabajarando
 *      pila,es la pila donde apilaremos el atributo generado
 *      direction, es la direccion en que estamos recorriendo el corpus (LEFT_TO_RIGHT
 *           o RIGHT_TO_LEFT).
 */
void swindow::winPushAmbiguityFeature(void *ptr,dictionary *d,struct stack_t *pila,int direction)
{
    char value[100];
    nodo_feature_list *p = (nodo_feature_list *)ptr;
    nodo *pn;
    simpleList *list;
    int w,*num,ret=0;
    infoDict *pInfoDict;

    strcpy(value,"");

    char *feature = new char[100];
    strcpy(feature,"");

    num = (int *) p->l.getIndex();
    sprintf(value,"%s%d:",p->mark,*num);
    pn = get(*num, direction);
    if (pn!=NULL)
    {

        w = d->getElement(pn->wrd);
        if (w!=HASH_FAIL)
        {
            list = (simpleList *) d->getElementMaybe(w);
            int numMaybe = d->getElementNumMaybe(w);
            while (ret>=0)
            {
                pInfoDict = (infoDict *) list->getIndex();
                numMaybe--;
                if (numMaybe>0) sprintf(value,"%s%s~",value,pInfoDict->txt);
                else sprintf(value,"%s%s",value,pInfoDict->txt);
                ret=list->next();
            }
            list->setFirst();
        }
        else sprintf(value,"%s%s",value,"UNKNOWN"); //is unknown word
    }
    else sprintf(value,"%s%s",value,EMPTY_POS);

    strcpy(feature,value);
    push (pila,feature);
    //delete[] feature; //modify wlj
}


/*
 * void winPushMFTFeature(void *ptr, dictionary *d, stack_t *pila, int direction)
 * Genera el atributo con la "Most Frequent Tag", la etiqueta mas frecuente.
 * Recibe como parametros:
 *      ptr, que es un puntero a un nodo de la lista de atributos (nodo_feature_list)
 *           aunque se recibe como un void*.
 *      d,   es el diccionario con el que estamos trabajarando
 *      pila,es la pila donde apilaremos el atributo generado
 *      direction, es la direccion en que estamos recorriendo el corpus (LEFT_TO_RIGHT
 *           o RIGHT_TO_LEFT).
 */
void swindow::winPushMFTFeature(void *ptr,dictionary *d,struct stack_t *pila,int direction)
{
    char value[100],mft[5];
    nodo_feature_list *p = (nodo_feature_list *)ptr;
    nodo *pn = NULL;                  //modify
    simpleList *list;
    int w,*num = NULL,max=0,ret=0;
    infoDict *pInfoDict = NULL;

    strcpy(value,"");
    //strcpy(feature,"");

    num = (int *) p->l.getIndex();
    sprintf(value,"%s%d:",p->mark,*num);
    pn = get(*num, direction);
    if (pn!=NULL)
    {
        w = d->getElement(pn->wrd);
        if (w!=HASH_FAIL)
        {
            list = (simpleList *) d->getElementMaybe(w);
            int numMaybe = d->getElementNumMaybe(w);
            while (ret>=0)
            {
                pInfoDict = (infoDict *) list->getIndex();
                numMaybe--;
                if (pInfoDict->num>max) strcpy(mft,pInfoDict->txt);
                ret=list->next();
            }
            list->setFirst();
            sprintf(value,"%s%s",value,mft);
        }
        else sprintf(value,"%s%s",value,"UNKNOWN"); //is unknown word
    }
    else sprintf(value,"%s%s",value,EMPTY_POS);
    //char *feature = new char[strlen(value)+1];
    char *feature = new char[strlen(value)+2];  //modify wlj
    strcpy(feature,value);
    push (pila,feature);
    //delete[] feature;   //modify wlj
    //feature = NULL;
}


/*
 * void winPushMaybeFeature(void *ptr, dictionary *d, stack_t *pila, int direction)
 * Genera tantos atributos "maybe" como posibles POS pueda tener la palabra, y los
 * apila en <pila>.
 * Recibe como parametros:
 *      ptr, que es un puntero a un nodo de la lista de atributos (nodo_feature_list)
 *           aunque se recibe como un void*.
 *      d,   es el diccionario con el que estamos trabajarando
 *      pila,es la pila donde apilaremos el atributo generado
 *      direction, es la direccion en que estamos recorriendo el corpus (LEFT_TO_RIGHT
 *           o RIGHT_TO_LEFT).
 */
void swindow::winPushMaybeFeature(void *ptr,dictionary *d,struct stack_t *pila,int direction)
{
    char value[100],txt[5];
    nodo_feature_list *p = (nodo_feature_list *)ptr;
    nodo *pn = NULL;
    simpleList *list = NULL;
    int w,*num = NULL,ret=0;
    infoDict *pInfoDict = NULL;
    char *feature = NULL;

    strcpy(value,"");
    num = (int *) p->l.getIndex();
    sprintf(txt,"%s%d~",p->mark,*num);
    pn = get(*num, direction);
    if (pn!=NULL)
    {
        w = d->getElement(pn->wrd);

        if (w!=HASH_FAIL)
        {
            list = (simpleList *) d->getElementMaybe(w);

            while (ret>=0)
            {
                feature = new char[10];
                strcpy(feature,"");
                pInfoDict = (infoDict *) list->getIndex();
                sprintf(feature,"%s%s:1",txt,pInfoDict->txt);
                push(pila,feature);
                //delete[] feature;
                ret=list->next();
            }
            list->setFirst();
        }
        else
        {
            feature = new char[15];
            sprintf(feature,"%s%s:1",txt,"UNKNOWN"); //is unknown word
            push(pila,feature);
            // delete[] feature;
        }
    }
    else
    {
        feature = new char[10];
        sprintf(feature,"%s%s:1",txt,EMPTY_POS);
        push(pila,feature);
        // delete[] feature;
    }
    //delete[] feature;
    //feature = NULL;
}


/*
 * void winPushPosFeature(void *ptr, dictionary *d, stack_t *pila, int direction)
 * Genera un atributo con la POS de algunos elementos de la ventana.
 * Recibe como parametros:
 *      ptr, que es un puntero a un nodo de la lista de atributos (nodo_feature_list)
 *           aunque se recibe como un void*.
 *      d,   es el diccionario con el que estamos trabajarando
 *      pila,es la pila donde apilaremos el atributo generado
 *      direction, es la direccion en que estamos recorriendo el corpus (LEFT_TO_RIGHT
 *           o RIGHT_TO_LEFT).
 */
void swindow::winPushPosFeature(void *ptr,dictionary *d, struct stack_t *pila,int direction)
{
    char value[100]="",name[100]="",txt[100]="";
    nodo_feature_list *p = (nodo_feature_list *)ptr;
    nodo *pn = NULL;
    infoDict *pInfoDict = NULL;
    //char *feature;

    int end=1,ret=1,w,*num;

    while (end>=0)
    {
        ret=1;
        num = (int *) p->l.getIndex();
        if (strcmp(name,EMPTY)==0) sprintf(name,"%s%d",p->mark,*num); //AKI3
        else sprintf(name,"%s,%d",name,*num);
        pn = get(*num, direction);

        if (pn==NULL) strcpy(txt,EMPTY_POS);
        else if ( (strcmp(pn->pos,EMPTY)==0) || (*num==0) )  //AKI3
        {

            w = d->getElement(pn->wrd);

            if (w!=HASH_FAIL)
            {
                simpleList *list = (simpleList *) d->getElementMaybe(w);
                int numMaybe = d->getElementNumMaybe(w);
                strcpy(txt,EMPTY);
                while (ret>=0)
                {
                    pInfoDict = (infoDict *) list->getIndex();
                    numMaybe--;
                    if (numMaybe>0) sprintf(txt,"%s%s_",txt,pInfoDict->txt);
                    else sprintf(txt,"%s%s",txt,pInfoDict->txt);
                    ret=list->next();
                }
                list->setFirst();
            }
            else strcpy(txt,"UNKNOWN"); //is unknown word
        }
        else strcpy(txt,pn->pos); //AKI3

        if (strcmp(value,EMPTY)==0) sprintf(value,"%s",txt); //AKI3
        else sprintf(value,"%s~%s",value,txt);

        end = p->l.next();
    }
    p->l.setFirst();
    sprintf(name,"%s:%s",name,value);

    //printf("%s %d\n",name,strlen(name));
    char *feature = new char[strlen(name)+2];

    if (feature == NULL)
    {
        exit(0);
    }

    strcpy (feature,name);
    //fprintf(stderr,"%s\n",feature);
    push (pila,feature);
    //delete[] feature;   //modify wlj
    //feature = NULL;
}

/*
 * void winPushPOSFeature(void *ptr, dictionary *d, stack_t *pila, int direction)
 * Genera un atributo con la palabra de algunos elementos de la ventana.
 * Recibe como parametros:
 *      ptr, que es un puntero a un nodo de la lista de atributos (nodo_feature_list)
 *           aunque se recibe como un void*.
 *      d,   es el diccionario con el que estamos trabajarando
 *      pila,es la pila donde apilaremos el atributo generado
 *      direction, es la direccion en que estamos recorriendo el corpus (LEFT_TO_RIGHT
 *           o RIGHT_TO_LEFT).
 */
void swindow::winPushWordFeature(void *ptr,dictionary *d, struct stack_t *pila,int direction)
{
    char value[200],name[200],txt[100];
    nodo_feature_list *p = (nodo_feature_list *)ptr;
    nodo *pn=NULL;
    //char *feature;

    int *num = (int *) p->l.getIndex();
    pn = get(*num, direction);

    if (pn==NULL) strcpy(value,EMPTY_WORD);
    else strcpy(value,pn->wrd);
    sprintf(name,"%s%d",p->mark,*num);

    while (p->l.next()>=0)
    {
        num = (int *) p->l.getIndex();
        sprintf(name,"%s,%d",name,*num);
        pn = get(*num, direction);

        if (pn==NULL) strcpy(txt,EMPTY_WORD);
        else strcpy(txt,pn->wrd);
        sprintf(value,"%s~%s",value,txt);
    }
    p->l.setFirst();
    sprintf(name,"%s%s%s",name,":",value);

    // printf("%s %d\n",name,strlen(name));
    char *feature = new char[(strlen(name)+2) * sizeof(char)];  //modify
    if(feature == NULL)
    {
        exit(0);
    }
    strcpy (feature,name);
    push(pila,feature);
    //delete[] feature; //modify wlj
    //feature = NULL;
}


/****************************************************************************/

/*
 * void deleteList()
 * Elimina todas las palabras existentes en la ventana
 */
void swindow::deleteList()
{
    if (first==NULL) return;
    while (first->next!=NULL)
    {
        first = first->next;
        delete first->previous->stackScores;
        delete first->previous;
    }
    delete last->stackScores;
    delete last;
    first=NULL;
    last=NULL;
    index=NULL;
}


void swindow::init()
{
    iniGeneric();
}

int swindow::iniGeneric()
{
    index = NULL;
    beginWin = NULL;
    endWin = NULL;
    first = NULL;
    last = NULL;
    numObj = 0;
    posBegin = posIndex;
    posEnd = posIndex;

    int ret = iniList();
    endWin = last;
    if (ret>0) readSentence();

    if (ret==-1) return -1;
    else if (ret==0) posEnd = posIndex+last->ord;
    else posEnd=posIndex+ret;

    beginWin = first;

    return ret;
}

int swindow::iniList()
{
    int j=0,ret=1;

    for(j=posIndex; ((j<lengthWin) && (ret>0)); j++) ret = readInput();

    //ret >1 correct
    //     0 if end of sentence
    //    -1 if there aren't words
    //    -2 if end of file
    if (ret>0) ret=j-posIndex-1;

    return ret;
}

/****************************************************************************/

int swindow::readSentence()
{
    int ret=1;
    while (ret>0) ret = readInput();
    return ret;
}

/****************************************************************************/

/* Read one line from corpus and add node to list
   Return 1 if it's ok
   0 if end of sentence
   -1 if there aren't more words
   -2 if end of file   */
/*int swindow::readInput()
  {
  if (feof(input)) return -2;

  char value[2][100]={EMPTY,EMPTY};
  int i=0,w=0,ret=1,isCom=0,addComAtEnd=0;
  char ant='q',c = fgetc(input);

  while ((!feof(input)) && (c!='\n'))
  {
  if (i<2 && ant=='#' && c=='#')
  { 
  char garbage[512];
  fgets(garbage,512,input);
  w=0;
  ret = 1;
  i=0;
  ant='q';
  strcpy(value[0],EMPTY);
  strcpy(value[1],EMPTY);
  c = fgetc(input);
  }
  if ((w==0) && (c==' ' || c=='\t' || c==32))
  {
//value[w][i]='\0';
i=0;
ret = 1;
w=1;
ant='q';
c = fgetc(input);
}
sprintf(value[w],"%s%c",value[w],c);
//value[w][i]=c;
i++;
ant=c;
c = fgetc(input);

}
value[w][i]='\0';

if ((strlen(value[0])<=0) && (!isCom)) return -1;

winAdd(value[0],value[1]);

if ((strcmp("。",value[0])==0) || (strcmp("？",value[0])==0) || (strcmp("！",value[0])==0)) return 0;
return 1;
}*/
int swindow::readInput()
{
    if (countOfinput >= num)
    {
        if ((strcmp("。",input[countOfinput-1])!=0) || (strcmp("？",input[countOfinput-1])!=0) || (strcmp("！",input[countOfinput-1])!=0))
            winAdd("。",EMPTY);
        return -2;
    }
    char value[2][100]={EMPTY,EMPTY};
    strcpy(value[0],input[countOfinput]);
    //printf("%s %d\n",value[0],countOfinput);
    countOfinput ++;

    if (strlen(value[0])<=0) return -1;

    winAdd(value[0],value[1]);

    if ( (strcmp("。",value[0])==0) || (strcmp("？",value[0])==0) || (strcmp("！",value[0])==0)) 
    {
        if((countOfinput < num) && strcmp("”",input[countOfinput])==0)
        {
            strcpy(value[0],input[countOfinput]);
            //printf("%s %d\n",value[0],countOfinput);
            //countOfinput ++;

            if (strlen(value[0])<=0) return -1;

            winAdd(value[0],value[1]);  
        }
        return 0;
    }
    return 1;
}

int swindow::winAdd(char *wrd, char *com)
{
    nodo *aux = new nodo;
    if(numObj == 0)
    {
        aux->previous=NULL;
        first = aux;
        last = aux;
        index = aux;
    }
    else
    {
        aux->previous = last;
        last->next = aux;
        last = aux;
    }
    aux->ord = numObj;
        int flag = checkEnglish(wrd);
        if (flag)
        {
            strcpy(aux->wrd,"@WS");
        } 
        else
        {
            int erRet=erLookRegExp(wrd);
                switch (erRet)
                {
                    case CARD: strcpy(aux->wrd,"@CARD"); break;
                    case CARDSEPS: strcpy(aux->wrd,"@CARDSEPS"); break;
                    case CARDPUNCT: strcpy(aux->wrd,"@CARDPUNCT"); break;
                    case CARDSUFFIX: strcpy(aux->wrd,"@CARDSUFFIX"); break;
                    default: strcpy(aux->wrd,wrd);
                }
        }

    strcpy(aux->realWrd,wrd);
    strcpy(aux->posOld,EMPTY);
    strcpy(aux->pos,EMPTY);
    strcpy(aux->comment,com);
    aux->stackScores = new stack_t;
    init_stack(aux->stackScores);
    aux->weight = 0;
    aux->weightOld = 0;
    aux->next=NULL;
    numObj++;
    return numObj;
}



/****************************************************************/
/****************************************************************/

swindow::~swindow()
{
    deleteList();
}

swindow::swindow(char**in,char **out,int i)
{
    input = in;
    output = out;
    num = i;
    int j;
    /*for (j=0;j<num;j++)
      {
      printf("%s\n",input[j]);
      }*/
    countOfinput = 0;
    countOfoutput = 0;
    lengthWin = 5;
    posIndex = 2;

    init();
}

swindow::swindow(char **in,char **out,int number, int position,int i)
{

    input = in;
    output = out;
    num = i;
    countOfinput = 0;
    countOfoutput = 0;

    if ((number<3) || (number<=position))
    { fprintf(stderr,"\nWindow Length can not be first or last element.\nLength should be greater than \"Interest Point Position\" or 3.\n");
        exit(0);
    }

    lengthWin = number;
    posIndex = position-1;

    init();
}

swindow::swindow(char **in,char **out,int number,int i)
{

    input = in;
    output = out;
    num = i;
    countOfinput = 0;
    countOfoutput = 0;
    lengthWin = number;
    posIndex = number/2;

    init();
}

/****************************************************************/
/****************************************************************/

/* Move Interest Point to next element */
int swindow::next()
{
    int ret = -1; // readInput();
    if ((ret==-1) && (endWin->next!=NULL)) ret=1;

    if  ((index==NULL) || (index->next==NULL)) return -1;
    if  ((posIndex>=posEnd) && (ret==-1)) 
    {
        if(num == 2)
        {
            index = index->next;
            return 0;
        }
        else
            return -1;

    }

    if  ((posIndex<posEnd) && (ret==-1)) posEnd--;
    if ((posEnd==lengthWin-1) && (ret!=-1)) endWin = endWin->next;

    if (posBegin==0) beginWin = beginWin->next;
    else if  ((posIndex>=posBegin) && (posBegin>0)) posBegin--;

    index = index->next;
    return 0;
}

/* Move Interest Point to previous element */
int swindow::previous()
{
    if ((index==NULL) || (index->previous==NULL)) return -1;

    if  ((posBegin==0) && (beginWin->previous!=NULL)) beginWin = beginWin->previous;
    else if  (posIndex>posBegin) posBegin++;

    if  (posEnd<lengthWin-1) posEnd++;
    else endWin = endWin->previous;

    index = index->previous;
    return 0;
}

/* Get Interest Point */
nodo *swindow::getIndex()
{
    return index;
}

nodo *swindow::get(int position,int direction)
{
    nodo *aux=NULL;
    int i=0;

    if (position == 0) return index;
    if (direction==2) position = -position;
    if ( (numObj == 0)
            || ((position<0) && (posIndex+position+1<posBegin))
            || ((position>0) && (posIndex+position>posEnd)) )
        return NULL;

    aux = index;

    while (i!=position)
    {
        if (position>0)
        { 
            i++;
            if (aux->next != NULL) aux = aux->next;
            else 	return NULL;
        }
        else
        { 
            i--;
            if (aux->previous != NULL) aux = aux->previous;
            else return NULL;
        }
    }

    return aux;
}

/* Show list elements */
int swindow::show()
{
    if (first==NULL) return 0;

    //char result[100];
    nodo *actual=first;

    //strcpy(result,"");
    //sprintf(result,"%s%c%s",actual->realWrd,'/',actual->pos);
    //printf("%s\n",result);
    //strcpy(output[countOfoutput],result);
    if (countOfoutput < num)
    {
        strcpy(output[countOfoutput],actual->pos);
        //printf("%d %s\n",countOfoutput,output[countOfoutput]);
        countOfoutput++;
    }

    while (actual->next!=NULL)
    {
        actual=actual->next;
        //sprintf(result,"%s%c%s",actual->realWrd,'/',actual->pos);
        //sprintf(output[countOfoutput],"%s%c%s\n",actual->realWrd,'/',actual->pos);
        //strcpy(output[countOfoutput],result);
        if (countOfoutput < num)
        {
            strcpy(output[countOfoutput],actual->pos);
            //printf("%d %s\n",countOfoutput,output[countOfoutput]);
            countOfoutput++;
        }


    }
    return 0;
}

void swindow::putLengthWin(int l)
{
    lengthWin = l;
}

void swindow::putIndex(int i)
{
    posIndex = i;
}

/*
 * action with value 0 to put max score
 * action with value 1 to reset values
 * action with value 2 to restore old value (last lap value)
 *
 */
int swindow::winMaterializePOSValues(int action)
{
    if (first==NULL) return 0;

    int inicio=1;
    weight_node_t *w,max;
    nodo *actual=first;

    while (actual!=NULL)
    {

        switch (action)
        {
            case 0: //PUT MAX
                inicio = 1;
                while(!empty(actual->stackScores))
                {
                    w = (weight_node_t *) pop(actual->stackScores);

                    if (inicio || w->data>max.data)
                    { max.data=w->data;
                        strcpy(max.pos,w->pos);
                        inicio = 0;
                    }
                    delete w;
                }
                actual->weight=max.data;
                strcpy(actual->pos,max.pos);
                //Added for 2 laps tagging
                actual->weightOld=max.data;
                strcpy(actual->posOld,max.pos);
                break;
            case 1: //RESET VALUES
                strcpy(actual->pos,"");
                actual->weight=0;
                break;
            case 2: //PUT OLD
                strcpy(actual->pos,actual->posOld);
                actual->weight=actual->weightOld;
                break;
        }
        actual=actual->next;
    }
    return 0;
}


/*
 * int winExistUnkWord(int direction, dictionary *d)
 * Esta funcion comprueba si hay parabras desconocidas.
 * En caso de que el parametro direction sea:
 *   LEFT_TO_RIGHT - mira si hay desconocidas a la
 *                   derecha del punto de interes de la ventana.
 *   RIGHT_TO_LEFT - mira si hay desconocidas a la izquierda
 *                   del punto de interes de la ventana.
 * Esta funcion devuelve:
 *   un entero >=0, si no hay desconocidas
 *              -1, si hay desconocidas
 */
int swindow::winExistUnkWord(int direction, dictionary *d)
{
    nodo *aux=index;
    int ret=0,i=posIndex;

    if (index==NULL) return 1;
    aux = index;

    while (ret>=0)
    {
        switch (direction)
        {
            case LEFT_TO_RIGHT:
                if (aux->next==NULL || aux==endWin) ret=-1;
                else aux = aux->next;
                if (d->getElement(aux->wrd)==HASH_FAIL)	return -1;
                i++;
                break;
            case RIGHT_TO_LEFT:
                if (aux->previous==NULL || aux==beginWin) ret=-1;
                else aux = aux->previous;
                if (d->getElement(aux->wrd)==HASH_FAIL) return -1;
                i--;
                break;
        }
    }
    return 0;
}


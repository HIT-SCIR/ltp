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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <unistd.h>
//#include <ntdef.h>

//#include <windows.h>

#include "common.h"

/*
 * FILE *openFile(char *name, char mode[])
 * Abre un fichero con el nombre <name> y en el modo <mode> 
 * (r lectura, w escritura, a actualización ...).
 * Devuelve el puntero al fichero
 * En caso de no poder abrir un fichero, termina la ejecucion
 */
FILE *openFile(char *name, char mode[])
{
    FILE *f;
        if ((f = fopen(name, mode))== NULL)
        {
            fprintf(stderr, "Error opening file: %s\n",name);
                exit(0);
        }
    return f;
}


void generateFileName(char *name, char *added,int numModel, int direction, int what, char *type, char *out)
{
    strcpy(out,name);	
        if (strcmp(added,"")!=0 && added!=NULL)
        { 
            sprintf(out,"%s.",out);
                for (int i=0; i < strlen(added); i++)
                {
                    if (added[i]==':') sprintf(out,"%s%s",out,"DOSPUNTS");
                    else if (added[i]=='\'') // && added[i+1]=='\'') 
                        sprintf(out,"%s%s",out,"COMETA");
                    else sprintf(out,"%s%c",out,added[i]);
                }
            // if (strcmp(added,":")==0) 	 sprintf(out,"%s.DOSPUNTS",out,added);
            // else if (strcmp(added,"''")==0)  sprintf(out,"%s.COMETES",out,added);
            // else sprintf(out,"%s.%s",out,added);
        }
    if (what==UNKNOWN) 	sprintf(out,"%s.UNK",out);
    else sprintf(out,"%s",out);
        if (numModel>=0) sprintf(out,"%s.M%d",out,numModel);
            if (direction==LEFT_TO_RIGHT) sprintf(out,"%s.LR",out);
            else if (direction==RIGHT_TO_LEFT) sprintf(out,"%s.RL",out);
                if (type!=NULL) sprintf(out,"%s.%s",out,type);
}

void showProcess(int num,int isEnd)
{
    if (isEnd) { fprintf(stderr,".%d sentences [DONE]\n\n",num); return; }
    else if (num%100==0) fprintf(stderr,"%d",num);
    else if (num%10==0) fprintf(stderr,".");
}

void showProcessDone(int num,int freq, int isEnd, char *what)
{
    if (isEnd) { fprintf(stderr,".%d %s [DONE]\n",num,what); return; }
    else if (num%freq==0) fprintf(stderr,".");
}

/* 
 * int goToWord(FILE *f, int offset) 
 * Lee <offset> lineas del canal o fichero <f>
 * Retorna -1 si encuentra eof
 * retorna el numero de lineas leidas si todo va bien         
 */
int goToWord(FILE *f, int offset)
{
    int cont=0;
        
        while (!feof(f) && cont<offset)
        {
            if (fgetc(f)=='\n') cont++;
        }
    
        if (feof(f)) return -1;
            return cont;
}

/*
 * int readString(FILE *f, char *out)
 * Lee un String entre espacios o espacios y fines de linea
 *    FILE *f - es fichero o canal de donde leeremos el String
 *    char *out - es un parametro de salida, contendra el 
 *                String que leeremos
 * Retorna -1 si encuentra eof
 * retorna  0 si todo va bien
 */
int readString(FILE *f, char *out)
{
    if (feof(f)) return -1;
        
            char c = fgetc(f);
            strcpy(out,"");
            
            while (!feof(f) && c==' ' && c=='\n') c=fgetc(f);
                
                    while (!feof(f) && c!=' ' && c!='\n')
                    { 
                        sprintf(out,"%s%c",out,c);
                            c=fgetc(f);
                    }
    if (feof(f) && strlen(out)==0) return -1;
        return 0;	
}

/*
 * int readTo(FILE *f, char endChar, chas endLine, char *out)
 * Lee un del canal o fichero <f>, el String leido sera devuelto como el
 * parametro de salida <out>. Para leer el String se leera hasta encontrar
 * el <endChar> o el caracter <endLine>
 * Retorna 0 si encuentra <endLine> 
 * retorna -1 si eof
 * retorn 1 si todo va bien y encuentra <endChar>
 */
int readTo(FILE *f, char endChar, char endLine, char *out)
{
    strcpy(out,"");
        char c = endChar+1;
        while (!feof(f) && c!=endChar && (endLine==0 || c!=endLine))
        { 
            c=fgetc(f);      
                if (c!=endChar && c!=endLine) sprintf(out,"%s%c",out,c);
        }
    if (feof(f)) return -1;
        if (c==endLine) return 0;
            return 1;  
}

//void saltarLinea(FILE *f)
//{
//	char c=' ';
//	while (c!='\n') c=fgetc(f);
//}

/*******************************************************/

void qsort(int a[], int lo, int hi) {
    int h, l, p, t;
        
        if (lo < hi) {
            l = lo;
                h = hi;
                p = a[hi];
                
                do {
                    while ((l < h) && (a[l] <= p))
                        l = l+1;
                            while ((h > l) && (a[h] >= p))
                                h = h-1;
                                    if (l < h) {
                                        t = a[l];
                                            a[l] = a[h];
                                            a[h] = t;
                                    }
                } while (l < h);
            
                t = a[l];
                a[l] = a[hi];
                a[hi] = t;
                
                qsort(a, lo, l-1);
                qsort(a, l+1, hi);
        } // if
}

/*
  static void main() {
  const int n = 20;
  int a[n];
  initialize(a, n);
  cout << "a = "; print(a, n);
  qsort(a, 0, n-1);
  cout << "a = "; print(a, n);
  }
  */


void showTime(char *what, /*clock_t start,clock_t end,*/ double real, double utime, double stime)
//struct  tms tbuff1, struct tms tbuff2)
{
    //double real  = ((double)(end-start))/CLOCKS_PER_SECOND;
    //double utime = ((double)tbuff2.tms_utime-(double)tbuff1.tms_utime)/CLOCKS_PER_SECOND;
    //double stime = ((double)tbuff2.tms_stime-(double)tbuff1.tms_stime)/CLOCKS_PER_SECOND;
    
        char message[200]="";
        sprintf(message,"%s: [ Real Time = %5.3lf secs.( %5.3lf usr + %5.3lf sys = %5.3lf CPU Time) ]\n",what,real,utime,stime,utime+stime);
        fprintf(stderr,"%s",message);
}



int buscarMenorEnString(char *szIn,char *szMenor,int *iMenor)
{
    char szString[10];
        char *szTemp = new char[strlen(szIn)+1];
        int  iString;
        
        if (strcmp(szIn,"")==0 || szIn==NULL) return 1;
            
                strcpy(szTemp,szIn);
                if (*iMenor==-1)
                    sscanf(szIn,"%s%d",szMenor,iMenor);
                else 
                {
                    sscanf(szIn,"%s%d",szString,&iString);
                        if (strcmp(szString,szMenor)<0)
                        {
                            strcpy(szMenor,szString);
                                *iMenor = iString;
                        }
                }
    
        int cont=0;
        int i;
        for (i=0; cont<2 && i<strlen(szTemp) ;i++)
        {
            if (szTemp[i]==' ') cont++;
        }
    
        return buscarMenorEnString(szTemp+i,szMenor,iMenor);
}

int ordenarStringPorParejas(char *szIn, char *szOut, int depth, char *szInicial)
{
    char szMenor[10],*p;
        char szTempMenor[10];
        char *szTemp = new char[strlen(szIn)+1];
        int  i=0,iMenor = -1;
        
        // fprintf(stderr,"1 in: %s  out: %s\n",szIn,szOut);
        if (strcmp(szIn,"")==0 || szIn==NULL
                || szInicial==NULL || szIn>(szInicial+strlen(szInicial))) return depth;
            if (depth==0) strcpy(szOut,"");
                
                    buscarMenorEnString(szIn,szMenor,&iMenor);  
                    sprintf(szTempMenor,"%s %d",szMenor,iMenor);
                    p = strstr(szIn,szTempMenor);
                    
                    strcpy(szTemp,"");
                    // Copiamos string szIn sin pareja menor
                    while (i<strlen(szIn) && p!=NULL)
                    {
                        if (&szIn[i]<p || &szIn[i]>(p+strlen(szTempMenor)))
                        {
                            sprintf(szTemp,"%s%c",szTemp,szIn[i]);
                        }
                        i++;
                    }
    
        if (strlen(szOut)==0) sprintf(szOut,"%s %d",szMenor,iMenor);
        else  sprintf(szOut,"%s %s %d",szOut,szMenor,iMenor);
            
                //fprintf(stderr,"2 in: %s  out: %s menor: %s temp: %s  ",szIn,szOut,szMenor,szTemp); 
                return   ordenarStringPorParejas(szTemp,szOut,depth+1,szInicial);
                
}


int obtainMark(FILE *channel,char *mark, int es_primera)
{
    //fprintf(stderr,"->");
    //if (!es_primera) while (fgetc(channel)!='\n'); 
    
        int ret;
        strcpy(mark,"");
        while (strlen(mark)==0) ret = readTo(channel,'(','\n',mark);
            //if (ret==0 && strlen(mark)==0) ret = -1;
            //fprintf(stderr,"[%s]",mark);
            if (ret==-1) return -1;
            else return ret;
                
                    /*    int i=0,ret=0;
                          char c='q',tmp[TAM_MARK]="";
                          
                          while ( (!feof(channel)) && (c!='(') && (c!='\n') && (c!=' ') )
                          {
                          c=fgetc(channel);
                    //if (c!=' ' && c!='(' && c!='\n') 
                    tmp[i]=c;
                    i++;
                    }
                    
                    tmp[i-1]='\0';
                    strcpy(mark,tmp);
                    if (strlen(mark)==0 || feof(channel)) return -1;
                    else if (c=='(') return 1;
                    else return 0;
                    
                    //    return atoi(num);
                    */
}

int obtainAtrInt(FILE *channel,int *endAtr)
{
    int i=0;
        char c=' ',num[5]="";
        
        while ( (!feof(channel)) && (c!='(') && (c!=',') && (c!=')') )
        {
            c=fgetc(channel);
                if ((c!='(') && (c!=')')) num[i]=c;
                    i++;
        }
    if (c==')') *endAtr=1;
        num[i]='\0';
            return atoi(num);
}


void createFeatureList(char *name,simpleList *featureList)
{
    int *i,endAtr,cont=0;
        char c;
        int ret = 1;
        //char temp[100];
        nodo_feature_list *data;
        
        FILE *f;
        if ((f = fopen(name, "rt"))== NULL)
        {
            fprintf(stderr, "Error opening file %s!!",name);
                exit(0);
        }
    
        //Insert feature Swn
        data = new nodo_feature_list; //odoConfigList;
    strcpy(data->mark,"Swn"); // = CHAR_NULL;
    data->n = 0;
        featureList->add(data);
        //fprintf(stderr,"(%s)",data->mark);
        
        char temp[10];
        ret = obtainMark(f,temp,TRUE);
        while (ret!=-1)
        {
            //fprintf(stderr,"@");
            data = new nodo_feature_list;
                strcpy(data->mark,temp); //c=c;
            //fprintf(stderr,"(%s)",data->mark);
            
                endAtr=0;
                cont=0;
                //fprintf(stderr,"\n%s(",temp);
                while (endAtr==0 && ret!=0)
                {
                    i = new int;
                        *i = obtainAtrInt(f,&endAtr);
                        data->l.add(i);
                        //fprintf(stderr,"%d,",*i);
                        cont++;
                }
            data->n = cont;
                featureList->add(data);
                strcpy(temp,"");
                ret = obtainMark(f,temp,FALSE);
        }
    
        fclose(f);      
}


void removeFiles(char *path, int type,int numModel, int direction, int verbose)
{
    char remove[200];
        switch (type)
        {
            case RM_TEMP_FILES: 
                                if (verbose==TRUE) 
                                    fprintf(stderr,"DELETING temporal files.\n",numModel);
                                        sprintf(remove,"rm %s*POS",path);
                                        system(remove);
                                        sprintf(remove,"rm %s*SVM",path);
                                        system(remove);
                                        sprintf(remove,"rm %s*SAMPLES",path);
                                        system(remove);
                                        sprintf(remove,"rm %s*MAP",path);
                                        system(remove);
                                        break;
            case RM_MODEL_FILES:
                                if (direction==LEFT_TO_RIGHT || direction==LR_AND_RL)
                                {
                                    if (verbose==TRUE) 
                                        fprintf(stderr,"DELETING files for MODEL %d in LEFT TO RIGHT sense,\n",numModel);
                                            sprintf(remove,"rm %s*M%d.LR.MRG",path,numModel);
                                            system(remove);
                                            sprintf(remove,"rm %s*M%d.LR.B",path,numModel);
                                            system(remove);
                                }
                                if (direction==RIGHT_TO_LEFT || direction==LR_AND_RL)
                                {
                                    if (verbose==TRUE) 
                                        fprintf(stderr,"DELETING files for MODEL %d in RIGHT TO LEFT sense,\n",numModel);
                                            sprintf(remove,"rm %s*M%d.RL.MRG",path,numModel);
                                            system(remove);
                                            sprintf(remove,"rm %s*M%d.RL.B",path,numModel);
                                            system(remove);
                                }
                                sprintf(remove,"rm %s*A%d",path,numModel);
                                    system(remove);
                                    break;
        }
}

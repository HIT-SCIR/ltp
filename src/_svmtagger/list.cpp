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

//#include <stdio.h>
#include "list.h"
#include "common.h"

/****************************************************************************
 *
 * Simple List
 *
 ****************************************************************************/

void simpleList::deleteList()
{
    if (first==NULL) return;
    listNode *aux=first;
    while (first->next!=NULL)
    {
        first = first->next;
        delete aux;
    }
    delete last;
    numObj = 0;
    first = NULL;
    last = NULL;
    index = NULL;
}

simpleList::~simpleList()
{
    deleteList();
}

simpleList::simpleList()
{
    numObj = 0;
    first = NULL;
    last = NULL;
    index = NULL;
}

/*Move Interest Point to next element */
int simpleList::next()
{
    if  ((index==NULL) || (index->next==NULL)) return -1;
    index = index->next;
    return 0;
}

/* Move Interest Point to previous element */
int simpleList::previous()
{
    if ((index==NULL) || (index->previous==NULL)) return -1;
    index = index->previous;
    return 0;
}

/* Get Interest Point */
void *simpleList::getIndex()
{
    return index->data;	  
}

/* Get Interest Point */
void *simpleList::getFirst()
{
    return first->data;
}

void *simpleList::getLast()
{
    return last->data;
}

void simpleList::setFirst()
{
    index = first;
}

void *simpleList::get(int position)
{
    listNode *aux;
    int i;

    if (numObj == 0 || position >= numObj)
        return NULL;

    aux = first;

    for(i=0; i<position; i++)
    {
        if(aux->next != NULL) aux = aux->next;
        else return NULL;
    }
    return aux->data;
}

/* Show list elements */
int simpleList::show()
{
    //fprintf(stderr, "\nShow list: ");
    if (first==NULL) return 0;

    listNode *actual=first;

    //printf("(%d wrd=%s POS=%s)\n",actual->ord,actual->wrd,actual->pos);
    while (actual->next!=NULL)
    {
        actual=actual->next;
        //fprintf (stderr,".");
        //printf("(%d wrd=%s POS=%s)\n",actual->ord,actual->wrd,actual->pos);
    }
    return 0;
}

int simpleList::add(void *object)
{
    listNode *aux = new listNode;

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
    //aux->key = new char[strlen(key)+1];
    //strcpy(aux->key,key);
    aux->ord = numObj;
    aux->data = object;
    aux->next=NULL;
    numObj++;
    return numObj;
}


int simpleList::delIndex()
{
    listNode *aux = index;

    if(numObj == 0) return -1;

    if (index==last && index==first)
    {
        first = aux->next;
        aux->previous = NULL;
        index = first;
        last = aux->previous;
        last->next = NULL;
        index = last;
    }
    else if (index==first)
    {

        first = aux->next;
        first->previous = NULL;
        index = first;
    }
    else if (index==last)
    {
        last = aux->previous;
        last->next = NULL;
        index = last;
    }
    else
    {
        aux->previous->next = aux->next;
        aux->next->previous = aux->previous;
    }

    numObj--;
    delete aux;
    return numObj;
}


int simpleList::isEmpty()
{
    if (numObj == 0 || first == NULL) return TRUE;
    else return FALSE;

}





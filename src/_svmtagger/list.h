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

#ifndef SIMPLELIST_H

struct listNode{
    int ord;
    void *data;
    listNode *next;
    listNode *previous;
    //char *key;      
};


class simpleList
{

    private:
        //List Control
        listNode *first;
        listNode *last;
        listNode *index;
        int numObj;

    public:
        ~simpleList();
        simpleList();
        void deleteList();
        int next();
        int previous();
        void setFirst();
        void *get(int position);
        void *getIndex();
        void *getFirst();
        void *getLast();
        int show();
        int add(void *object);
        int delIndex();
        int isEmpty();
};

#define SIMPLELIST_H
#endif

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

#ifndef STACK_H
#define STACK_H

#define STACKSIZE	1000

typedef enum {_FALSE = 0,_TRUE = 1} _boolean;

typedef void *element_type;

struct stack_t {
    int top;
    element_type items[STACKSIZE];
};

_boolean empty(struct stack_t *ps);
void init_stack(struct stack_t *ps);
element_type pop(struct stack_t *ps);
void push(struct stack_t *ps, element_type x);
element_type stack_top(struct stack_t *ps);

#endif

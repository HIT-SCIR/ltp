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
#include "stack.h"

/****************************************************************************
 * empty -- Indica si la pila est?vac?a o no
 *
 * Par?metros:
 *		*ps: puntero a la pila
 * Devuelve:
 *		TRUE si est?vac?a FALSE si no lo est?****************************************************************************/
_boolean empty(struct stack_t *ps)
{
    return((_boolean)(ps->top == -1));
}
/****************************************************************************
 * init_stack -- Inicializa la pila
 *
 * Par?metros:
 *		*ps: puntero a la pila
 ****************************************************************************/
void init_stack(struct stack_t *ps)
{
    ps->top = -1;
}
/****************************************************************************
 * pop -- Extrae el elemento del top de la pila si no est?vac?a
 *
 * Par?metros:
 *		*ps: puntero a la pila
 * Devuelve:
 *		El elemento del top de la pila si no est?vac?a
 ****************************************************************************/
element_type pop(struct stack_t *ps)
{
    if (empty(ps)) 	return NULL;
    return(ps->items[ps->top--]);
}
/****************************************************************************
 * push -- Coloca un valor en la pila
 *
 * Par?metros:
 *		*ps: puntero a la pila
 *		x: valor a colocar en la pila
 ****************************************************************************/
void push(struct stack_t *ps, element_type x)
{
    if (ps->top == STACKSIZE -1) {
        fprintf(stderr,"Error: Stack Overflow. %d %d\n",ps->top,STACKSIZE-1);
        exit(1);
    }
    else
        ps->items[++(ps->top)] = x;
}
/****************************************************************************
 * stack_top -- Devuelve sin quitarlo de la pila el elemento que est?en el
 *					top de la misma, si no est?vac?a
 *
 * Par?metros:
 *		*ps: puntero a la pila
 * Devuelve:
 *		El elemento del top de la pila si no est?vac?a
 si esta vacia devuelve NULL
 ****************************************************************************/
element_type stack_top(struct stack_t *ps)
{
    if (empty(ps)) 	return NULL;
    return(ps->items[ps->top]);
}
/****************************************************************************/

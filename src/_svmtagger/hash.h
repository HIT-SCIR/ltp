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

#ifndef __SVMTAGGER_HASH_H__
#define __SVMTAGGER_HASH_H__

#include <stdio.h>

#ifdef __cplusplus

extern "C" {
#endif

    typedef struct hash_t
    {
        struct hash_node_t **bucket;        /* array of hash nodes */
        int size;                           /* size of the array */
        int entries;                        /* number of entries in table */
        int downshift;                      /* shift cound, used in hash function */
        int mask;                           /* used to select bits for hashing */
    } hash_t;


    typedef struct hash_node_t {
        int data;                           /* data in hash node */
        const char * key;                   /* key for hash lookup */
        struct hash_node_t *next;           /* next node in hash chain */
    } hash_node_t;

#define HASH_FAIL -1

    void hash_init(hash_t *, int);

    int hash_lookup (const hash_t *, const char *);

    int hash_insert (hash_t *, const char *, int);

    int hash_delete (hash_t *, const char *);

    void hash_destroy(hash_t *);

    char *hash_stats (hash_t *);

    void hash_print(hash_t *tptr,FILE *f);

#ifdef __cplusplus
}
#endif

#endif

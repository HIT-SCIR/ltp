/*
 * File Name     : MyTree.h
 * Author        : Frumes, hjliu
 *
 * Create Time   : 2006Äê12ÔÂ31ÈÕ
 * Project Name  £ºNewSRLBaseLine
 *
 */

#ifndef _MY_TREE_
#define _MY_TREE_
#pragma warning(disable:4786)

#include <stdlib.h>
#include "MyStruct.h"
#include "ConstVar.h"

class MyTree
{
    public:
        MyTree(const LTPData* ltpData);
        ~MyTree();

        int  GetRootID() const;
        void GetNodeValue(DepNode& depNode, int nodeID) const;
        int  GetLeftChild(int nodeID) const;
        int  GetRightChild(int nodeID) const;
        int  GetLeftSib(int nodeID) const;
        int  GetRightSib(int nodeID) const;
        void GetAllSibs(int nodeID, deque<int>& dequeSibs) const;
        void GetAllNodePath(int intCurPdID, vector<string>& vecPath) const;
        void GetFamilyShip(string& strFSship, int nodeID1, int nodeID2) const;
        int  GetRCParent(int nodeID1, int nodeID2) const;
        bool IsRoot(int nodeID) const;
        bool IsLeaf(int nodeID) const;

    private:
        // build and destroy the the tree
        bool BuildDepTree(const LTPData* ltpData);
        void InitTree(const LTPData* ltpData);
        bool UpdateTree();
        void ClearTree();

        // the family members relationship
        bool IsParent(int parentID, int childID) const;
        bool IsChild(int childID, int parentID) const;
        bool IsSibling(int nodeID1, int nodeID2) const;
        bool IsAncestor(int anceID, int postID) const;
        bool IsPosterity(int postID, int anceID) const;


        // other operation
        void GetNodeValue(
                DepNode& depNode,
                const DepTree& depTree,
                int nodeID) const;
        bool IsLeaf(
                const DepTree& depTree,
                int rootID) const;
        void UpdateNodePS(
                DepTree& depTree,
                int nodeID,
                int childNodeID);
        void CopyAllNodePS(const DepTree& depTree);

    public:
        DepTree m_depTree;

    private:
        int m_rootID;
};

#endif


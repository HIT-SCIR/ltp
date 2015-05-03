/*
 * File Name     : MyTree.cpp
 * Author        : Frumes, hjliu
 *
 * Create Time   : 2006Äê12ÔÂ31ÈÕ
 * Project Name  £ºNewSRLBaseLine
 *
 */
#include <iostream>
#include <algorithm>
#include "MyTree.h"

using namespace std;

MyTree::MyTree(const LTPData* ltpData)
{
    BuildDepTree(ltpData);
}

MyTree::~MyTree()
{
    ClearTree();
}

int MyTree::GetRootID() const
{
    return m_rootID;
}

// The interface: return the depNode with index nodeID
void MyTree::GetNodeValue(DepNode& depNode, 
                          int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));
    depNode = m_depTree.vecDepNode.at(nodeID);
}


// Left child: the left child but near to the current node
int MyTree::GetLeftChild(const int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    deque<int> dequeChildren;
    deque<int>::iterator itChildren;
    int leftChild = I_NULL_ID;

    DepNode depNode;
    GetNodeValue(depNode, nodeID);
    dequeChildren = depNode.dequeChildren;
    itChildren = dequeChildren.begin();
    while(itChildren != dequeChildren.end())
    {
        if(*itChildren < nodeID)
        {
            leftChild = *itChildren;
        }
        else // child node id greater than the parent id
        {
            break;
        }

        ++ itChildren;
    }

    return leftChild;
}

// Right child: the right child but near to the current node
int MyTree::GetRightChild(const int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    deque<int> dequeChildren;
    deque<int>::iterator itChildren;
    int rightChild = I_NULL_RIGHT;

    DepNode depNode;
    GetNodeValue(depNode, nodeID);
    dequeChildren = depNode.dequeChildren;
    itChildren = dequeChildren.begin();
    while(itChildren != dequeChildren.end())
    {
        if(*itChildren > nodeID)
        {
            //greater than parent node id
            rightChild = *itChildren;
            break;
        }

        ++ itChildren;
    }

    return rightChild;
}

int MyTree::GetLeftSib(const int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    int leftID = I_NULL_ID;
    DepNode depNode;
    GetNodeValue(depNode, nodeID);

    int parentID = depNode.parent;
    if(parentID < 0)
    {    //process punctuation or root node
        return leftID;
    }

    GetNodeValue(depNode, parentID);
    deque<int> dequeChildren = depNode.dequeChildren;
    deque<int>::iterator itDequeChildren;

    itDequeChildren = dequeChildren.begin();
    while(itDequeChildren != dequeChildren.end())
    {
        if(*itDequeChildren < nodeID)
        {
            leftID = *itDequeChildren;
        }
        else
        {
            break;
        }

        ++ itDequeChildren;
    }

    return leftID;
}

int MyTree::GetRightSib(const int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    int rightID = I_NULL_RIGHT;
    DepNode depNode;
    GetNodeValue(depNode, nodeID);

    int parentID = depNode.parent;
    if(parentID < 0)
    {    //process punctuation or root node
        return rightID;
    }

    GetNodeValue(depNode, parentID);
    deque<int> dequeChildren = depNode.dequeChildren;
    deque<int>::iterator itDequeChildren;

    itDequeChildren = dequeChildren.begin();
    while(itDequeChildren != dequeChildren.end())
    {
        if(*itDequeChildren > nodeID)
        {
            rightID = *itDequeChildren;
            break;
        }
        ++ itDequeChildren;
    }

    return rightID;
}

void MyTree::GetAllSibs(
        const int nodeID,
        deque<int>& dequeSibs) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    DepNode depNode;
    GetNodeValue(depNode, nodeID);

    int parentID = depNode.parent;
    if(parentID < 0)
    {    //punctuation or root
        return;
    }

    GetNodeValue(depNode, parentID);
    dequeSibs = depNode.dequeChildren;

    //delete the current node
    deque<int>::iterator itDequeSibs;
    itDequeSibs = std::find(dequeSibs.begin(), dequeSibs.end(), nodeID);
    if (itDequeSibs != dequeSibs.end())
    {
        dequeSibs.erase(itDequeSibs);
    }
}

// Set the path feature of every node for current predicate
void MyTree::GetAllNodePath(
        const int intCurPdID,
        vector<string>& vecPath) const
{
    assert((intCurPdID < m_depTree.nodeNum) && (intCurPdID >= 0));

    string strRootPath;
    string strCurRel;
    int intCurNodeID;
    int intParentID;

    //initial the path and predicate path
    char str[16];
    vecPath.clear();
    vecPath.resize(m_depTree.nodeNum, S_NULL_STR);

    // itoa(intCurPdID, str, I_RADIX); //pd node: intCurPdID
    sprintf(str, "%d", intCurPdID);
    vecPath.at(intCurPdID) = str;
    strRootPath = str;

    //get the root path and update the path from pd to root
    string strCur;
    intCurNodeID = intCurPdID;
    intParentID = intCurPdID;
    // while(!IsRoot(intParentID))
    while(1)
    {    //the predicate may not be punctuation
        intParentID = m_depTree.vecDepNode.at(intCurNodeID).parent;
        if(intParentID < 0){
            intParentID = intCurNodeID;
            break;
        }
        intCurNodeID = intParentID;
        // itoa(intCurNodeID, str, I_RADIX);
        sprintf(str, "%d", intCurNodeID);

        strCur = str;
        strRootPath = strCur + S_PATH_DOWN +strRootPath;
        vecPath.at(intCurNodeID) = strRootPath;        
    }
    vecPath.at(intParentID) = strRootPath; //the intParentID is RootID

    //visit the tree using DWS(Width First Search)
    queue<int> queDepNode;
    deque<int> dequeChildren;
    deque<int>::iterator itDequeChildren;
    string strParentPath;
    string strCurNodePath;

    //get the children of root, and push them to the queue
    dequeChildren = m_depTree.vecDepNode.at(intParentID).dequeChildren; 
    itDequeChildren = dequeChildren.begin();
    while(itDequeChildren != dequeChildren.end())
    {
        queDepNode.push(*itDequeChildren);
        ++ itDequeChildren;
    }

    while(!queDepNode.empty())
    {
        //pop the front element of the queue
        intCurNodeID = queDepNode.front();
        queDepNode.pop();

        //check whether current node is along the path: from pd to root
        //if no, update the current node path
        if(!vecPath.at(intCurNodeID).compare(S_NULL_STR))
        {
            intParentID = m_depTree.vecDepNode.at(intCurNodeID).parent;
            strParentPath = vecPath.at(intParentID);
            // itoa(intCurNodeID, str, I_RADIX);
            sprintf(str, "%d", intCurNodeID);

            strCur = str;
            strCurNodePath = strCur + S_PATH_UP + strParentPath;
            vecPath.at(intCurNodeID) = strCurNodePath;
        }

        dequeChildren = m_depTree.vecDepNode.at(intCurNodeID).dequeChildren;
        itDequeChildren = dequeChildren.begin();
        while(itDequeChildren != dequeChildren.end())
        {
            queDepNode.push(*itDequeChildren);
            ++ itDequeChildren;
        }
    }
}

// Get the familyship of nodeID1 and nodeID2
void MyTree::GetFamilyShip(
        string& strFShip,
        int nodeID1,
        int nodeID2) const
{
    assert((nodeID1 < m_depTree.nodeNum) && (nodeID1 >= 0));
    assert((nodeID2 < m_depTree.nodeNum) && (nodeID2 >= 0));

    if(IsParent(nodeID1, nodeID2))
    {
        strFShip = S_FMS_PARENT;
    }
    else if(IsChild(nodeID1, nodeID2))
    {
        strFShip = S_FMS_CHILD;
    }
    else if(IsSibling(nodeID1, nodeID2))
    {
        strFShip = S_FMS_SIBLING;
    }
    else if(IsAncestor(nodeID1, nodeID2))
    {
        strFShip = S_FMS_ANCESTOR;
    }
    else if(IsPosterity(nodeID1, nodeID2))
    {
        strFShip = S_FMS_POSTERITY;
    }
    else
    {
        strFShip = S_FMS_OTHER;
    }
}

// Get the recent common parent
int MyTree::GetRCParent(int nodeID1, int nodeID2) const
{
    assert((nodeID1 < m_depTree.nodeNum) && (nodeID1 >= 0));
    assert((nodeID2 < m_depTree.nodeNum) && (nodeID2 >= 0));

    //if nodeID1 or nodeID2 is punctuation
    if ( (m_depTree.vecDepNode.at(nodeID1).parent == I_PUN_PARENT_ID) ||
         (m_depTree.vecDepNode.at(nodeID2).parent == I_PUN_PARENT_ID) )
    {
        return I_NULL_RCP;
    }

    int high1 = 0;
    int high2 = 0;
    int parent1 = nodeID1;
    int parent2 = nodeID2;

    //calculate the high of nodeID1 and nodeID2
    while (!IsRoot(parent1))
    {
         parent1 = m_depTree.vecDepNode.at(parent1).parent;
         high1++;
    }
    while (!IsRoot(parent2))
    {
        parent2 = m_depTree.vecDepNode.at(parent2).parent;
        high2++;
    }

    //move low node above
    parent1 = nodeID1;
    parent2 = nodeID2;
    if (high1 > high2)
    {
        for(int i = 0; i < (high1 - high2); i++)
        {
            parent1 = m_depTree.vecDepNode.at(parent1).parent;
        }
    }
    else
    {
        for(int i = 0; i < (high2 - high1); i++)
        {
            parent2 = m_depTree.vecDepNode.at(parent2).parent;
        }
    }

    //move tow node together
    while (parent1 != parent2)
    {
        parent1 = m_depTree.vecDepNode.at(parent1).parent;
        parent2 = m_depTree.vecDepNode.at(parent2).parent;
    }

    return parent1;

}

bool MyTree::IsRoot(const int nodeID) const
{
    return m_rootID == nodeID;
}

bool MyTree::IsLeaf(const int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));

    if(m_depTree.vecDepNode.at(nodeID).dequeChildren.empty())
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// The interface, build the depTree using parent and relation information
bool MyTree::BuildDepTree(const LTPData* ltpData)
{
    InitTree(ltpData);

    return UpdateTree();
}

// Update current node using the child's constituent
void MyTree::UpdateNodePS(
        DepTree& depTree,
        const int nodeID,
        const int childNodeID)
{
    int begin = depTree.vecDepNode.at(nodeID).constituent.first;
    int end   = depTree.vecDepNode.at(nodeID).constituent.second;
    int childBeg = depTree.vecDepNode.at(childNodeID).constituent.first;
    int childEnd = depTree.vecDepNode.at(childNodeID).constituent.second;

    pair<int, int> pairPs;
    pairPs.first = (begin < childBeg) ? begin : childBeg;
    pairPs.second = (end > childEnd)  ? end : childEnd;
    depTree.vecDepNode.at(nodeID).constituent = pairPs;
}

// Copy the Nodes position of depTree1 to depTree2
void MyTree::CopyAllNodePS(const DepTree& depTree)
{
    for(int i = 0; i < m_depTree.nodeNum; i++)
    {
        m_depTree.vecDepNode.at(i).constituent = depTree.vecDepNode.at(i).constituent;
    }
}

// Initial the Dependency Tree, but the consituent position may
void MyTree::InitTree(const LTPData* ltpData)
{
    int index;
    vector<int>::const_iterator itParent;
    vector<string>::const_iterator itRelation;

    index = 0;
    m_rootID = I_NULL_ID;
    itParent = ltpData->vecParent.begin();
    itRelation = ltpData->vecRelation.begin();
    while(itParent != ltpData->vecParent.end())
    {
        DepNode depNode;

        depNode.parent = *itParent;
        depNode.relation = *itRelation;
        depNode.id = index;
        depNode.constituent.first = index;
        depNode.constituent.second = index;

        m_depTree.vecDepNode.push_back(depNode);
        //if relation is "HED", it is the root, else root is -1
        if(!depNode.relation.compare(S_ROOT_REL)) 
        {
            m_rootID = index;
        }

        ++ itParent;
        ++ itRelation;
        ++ index;
    }
    m_depTree.nodeNum = index;

    //get the children for every node
    // size_t id = 0;
    for(size_t id = 0; id <m_depTree.nodeNum; id++)
    {
        index =  m_depTree.vecDepNode.at(id).parent;
        if(index >= 0) //except the root node and punc nodes
        {
            m_depTree.vecDepNode.at(index).dequeChildren.push_back(id);
        }
    }
}

// Update the consituent position for each depNode
bool MyTree::UpdateTree()
{
    if(m_rootID == I_NULL_ID)
    {   //if there isn`t verb in the sentence, do nothing
        return 0;
    }

    //a temp copy, used for update
    DepTree updateTree = m_depTree; 
    int rootID = m_rootID;

    //iterate until the root's  constituent is updated
    while(!IsLeaf(updateTree, rootID))
    {
        vector<DepNode>::iterator itDepNode;
        int curIndex = 0;

        //check if the node is leaf, if yes update it's constituent and it's parent's
        itDepNode = updateTree.vecDepNode.begin();
        while(itDepNode != updateTree.vecDepNode.end())
        {
            deque<int>::size_type childNum = (*itDepNode).dequeChildren.size(); //children number

            //scan the children, if leaf then update, else push_back
            for(deque<int>::size_type n = 0; n < childNum; n++)
            {
                int firstChildID = (*itDepNode).dequeChildren.front();
                (*itDepNode).dequeChildren.pop_front();

                if(IsLeaf(updateTree,firstChildID))
                {   //the node of id(depChildren[n]) in updateTree is leaf
                    UpdateNodePS(updateTree, curIndex, firstChildID);
                }
                else
                {   //push the child back 
                    (*itDepNode).dequeChildren.push_back(firstChildID);
                }
            }

            ++ itDepNode;
            ++ curIndex; //next node
        } //interior while

        //for debug
        //string strTemp;

    } //exterior while

    //update the m_depTree using the updateTree
    CopyAllNodePS(updateTree);

    return 1;
}

// Clear the Tree
void MyTree::ClearTree()
{
    m_depTree.vecDepNode.clear();
    m_depTree.nodeNum = 0;
    m_rootID = I_NULL_ID;
}

bool MyTree::IsLeaf(const DepTree& depTree, 
                    int rootID) const
{
    DepNode depNode;
    GetNodeValue(depNode, depTree, rootID);

    if(depNode.dequeChildren.empty())
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// Check if node1 is parent of node2
bool MyTree::IsParent(int parentID, 
                      int childID) const
{
    assert((parentID < m_depTree.nodeNum) && (parentID >= 0));
    assert((childID < m_depTree.nodeNum) && (childID >= 0));

    int newParentID = m_depTree.vecDepNode.at(childID).parent;
    if (newParentID < 0)
    {    //root or punctuaion
        return 0;
    }

    if(parentID == newParentID)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// Check if node1 is child of node2
bool MyTree::IsChild(int childID, 
                     int parentID) const
{
    assert((parentID < m_depTree.nodeNum) && (parentID >= 0));
    assert((childID < m_depTree.nodeNum) && (childID >= 0));

    if(IsParent(parentID, childID))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// Check if node1 is sibling of node2
bool MyTree::IsSibling(const int nodeID1, 
                       const int nodeID2) const
{
    assert((nodeID1 < m_depTree.nodeNum) && (nodeID1 >= 0));
    assert((nodeID2 < m_depTree.nodeNum) && (nodeID2 >= 0));

    deque<int> dequeSibs;
    deque<int>::iterator itDequeSibs;

    GetAllSibs(nodeID2, dequeSibs);

    itDequeSibs = std::find(dequeSibs.begin(), dequeSibs.end(), nodeID1);
    if(itDequeSibs != dequeSibs.end())
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// check if node anceID is ancesstor of node postID
bool MyTree::IsAncestor(int anceID, 
                        int postID) const
{
    assert((anceID < m_depTree.nodeNum) && (anceID >= 0));
    assert((postID < m_depTree.nodeNum) && (postID >= 0));

    int parentID = m_depTree.vecDepNode.at(postID).parent;

    if (parentID < 0)
    {    //root or punctation node
        return 0;
    }

    while(!IsRoot(parentID))
    {
        if(anceID == parentID)
        {
            return 1;
        }
        if(parentID < 0) break; // Added by Carl at 2009.09.29
        parentID = m_depTree.vecDepNode.at(parentID).parent;
    }

    return 0;
}

// check if node postID is posterity of node anceID
bool MyTree::IsPosterity(int postID, 
                         int anceID) const
{
    assert((anceID < m_depTree.nodeNum) && (anceID >= 0));
    assert((postID < m_depTree.nodeNum) && (postID >= 0));

    if(IsAncestor(anceID, postID))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void MyTree::GetNodeValue(DepNode& depNode, 
                          const DepTree& depTree, 
                          int nodeID) const
{
    assert((nodeID < m_depTree.nodeNum) && (nodeID >= 0));
    depNode = depTree.vecDepNode.at(nodeID);
}



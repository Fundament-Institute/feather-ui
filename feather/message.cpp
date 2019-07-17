// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/outline.h"
#include "feather/rtree.h"
#include "util.h"

static_assert(sizeof(fgMessageResult) <= sizeof(double));
static_assert(sizeof(fgMessage) <= sizeof(double) * 4);

extern "C" void DrawRTNode(fgVec parent, const struct FG__ROOT* root, fgRTNode* node, const fgMessage* msg)
{
  parent += node->area.topleft;
  node = node->children;

  while(node)
  {
    if(node->document)
    {
      fgMessage m = *msg;
      m.draw.area = node->area + parent;
      fgSendMessage(root, node->document, &m);
    }
    else
    {
      DrawRTNode(parent, root, node, msg);
      node = node->sibling;
    }
  }
}

extern "C" fgMessageResult fgDefaultBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  switch(msg->type)
  {
  case FG_MSG_ADD_CHILD:
    //LLAdd<fgDocumentNode>(msg->child.node, node->children);
    (*node->outline->layout)(msg->child.node, 0, 0, msg->child.scale, msg->child.dpi);
    break;
  case FG_MSG_REMOVE_CHILD:
    //LLRemove<fgDocumentNode>(msg->child.node, node->children);
    //msg->child.node->next = msg->child.node->prev = 0;
    (*node->outline->layout)(msg->child.node, 0, node->outline, msg->child.scale, msg->child.dpi);
    break;
  case FG_MSG_DRAW:
    // Do a partial pre-order traversal of the R-tree that terminates on a leaf with a valid document node
    DrawRTNode(msg->draw.area.topleft, root, node->rtnode, msg);
    break;
  case FG_MSG_STATE_GET:
  {
    unsigned int err = fgStandardResolver(node->outline, msg->subtype, msg->getState.value, msg->getState.id);
    if(err == ~0)
      err = (*node->outline->stateresolver)(node->outline->data, msg->subtype, msg->getState.value, msg->getState.id);
    if(err == ~0)
      err = (*node->outline->auxresolver)(node->outline->auxdata, msg->subtype, msg->getState.value, msg->getState.id);
    return fgMessageResult{ (fgError)err };
  }
  case FG_MSG_STATE_SET:
    if(unsigned int err = (*node->outline->stateresolver)(node->outline->data, msg->subtype, const_cast<fgCalcNode*>(&msg->setState.value), msg->setState.id); err == ~0)
      return fgMessageResult{ -1 };
    break;
  case FG_MSG_TEXT_SCALE:
  case FG_MSG_DPI:
    return node->outline->parent ? (*node->outline->parent->behavior)(root, node, msg) : fgMessageResult{ -1 };
  }
  return fgMessageResult{ 0 };
}

extern "C" fgMessageResult fgSendMessage(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  return (*node->outline->behavior)(root, node, msg);
}
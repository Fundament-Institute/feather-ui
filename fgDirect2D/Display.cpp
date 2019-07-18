// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#include "Display.h"
#include "Context.h"
#include "feather/outline.h"

using namespace D2D;

fgMessageResult D2D::DisplayBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  DisplayData& data = *reinterpret_cast<DisplayData*>(node->outline->auxdata);
  fgMessageResult r;

  switch(msg->type)
  { // Note that we do not process drawing here - we control when we call the draw function.
  case FG_MSG_ADD_CHILD:
    r = fgDefaultBehavior(root, node, msg);
    if(msg->child.node->backend)
      delete reinterpret_cast<Context*>(msg->child.node->backend);
    msg->child.node->backend = new Context(root, msg->child.node, node->outline); // Create the window and attach our node to it, but don't show it (we don't know where it is yet)
    return r;
  case FG_MSG_REMOVE_CHILD:
    if(msg->child.node->backend) // Destroy the window 
      delete reinterpret_cast<Context*>(msg->child.node->backend);
    return fgDefaultBehavior(root, node, msg);
  case FG_MSG_TEXT_SCALE:
    r.scale.scale = data.scale;
    return r;
  case FG_MSG_DPI:
    r.dpi.dpi = &data.dpi;
    return r;
  }

  return fgDefaultBehavior(root, node, msg);
}
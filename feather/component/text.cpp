// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/component/text.h"
#include "feather/outline.h"
#include "feather/root.h"

extern "C" fgMessageResult fgTextBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto data = reinterpret_cast<fgTextData*>(node->outline->data);
  auto state = reinterpret_cast<fgTextState*>(node->state);

  switch(msg->type)
  {
  case FG_MSG_CONSTRUCT:
    if(!msg->construct.ptr)
      return fgMessageResult{ .construct = { sizeof(fgTextState) } };

    *reinterpret_cast<fgTextState*>(msg->construct.ptr) = { 0 };
    break;
  case FG_MSG_DRAW:
    (*root->backend->drawFont)(root->backend, msg->draw.data, state->font, state->layout, &msg->draw.area, data->color, node->outline->lineHeight, data->letterSpacing, data->blur, data->aa);
    break;
  }

  return fgMessageResult{ 0 };
}
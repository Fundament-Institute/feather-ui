// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/component/box.h"
#include "feather/outline.h"
#include "feather/root.h"

fgMessageResult fgBoxBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto data = reinterpret_cast<fgBoxData*>(node->outline->data);
  switch(msg->type)
  {
  case FG_MSG_DRAW:
    (*root->backend->drawRect)(root->backend, msg->draw.data, &msg->draw.area, &data->corners, data->fillColor, data->border, data->borderColor, data->blur);
    break;
  }

  return fgMessageResult{ 0 };
}
// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/component/box.h"
#include "feather/outline.h"
#include "feather/root.h"
#include <unordered_map>
#include <algorithm>
#include <string>

extern "C" fgMessageResult fgBoxBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto data = reinterpret_cast<fgBoxData*>(node->outline->statedata);
  switch(msg->type)
  {
  case FG_MSG_DRAW:
    (*root->backend->drawRect)(root->backend, msg->draw.data, &msg->draw.area, &data->corners, data->fillColor, data->border, data->borderColor, data->blur, 0);
    break;
  }

  return fgDefaultBehavior(root, node, msg);
}

static const std::unordered_map<std::string, unsigned int> box_state_map = {
  { "fillcolor", FG_EVENT_FILL_COLOR },
  { "border", FG_EVENT_BORDER },
  { "bordercolor", FG_EVENT_BORDER_COLOR },
  { "blur", FG_EVENT_BLUR },
};

static unsigned int box_fields[] = { FG_EVENT_FILL_COLOR, FG_EVENT_BORDER, FG_EVENT_BORDER_COLOR, FG_EVENT_BLUR };

extern "C" unsigned int fgBoxResolver(void* ptr, unsigned int index, fgCalcNode* out, const char* id)
{
  if(id)
  {
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = box_state_map.find(raw.c_str()); i != box_state_map.end())
      return i->second;
    return ~0;
  }
  if(!ptr)
  {
    out->value.p = box_fields;
    return sizeof(box_fields) / sizeof(unsigned int);
  }

  auto data = reinterpret_cast<fgBoxData*>(ptr);
  switch(index)
  {
  case FG_EVENT_FILL_COLOR:
    if(out->type == FG_CALC_NONE)
    {
      out->value.i = data->fillColor.color;
      return out->type = FG_CALC_INT;
    }
    if(out->type != FG_CALC_INT)
      break;
    data->fillColor.color = out->value.i;
    return FG_CALC_INT;
  case FG_EVENT_BORDER:
    if(out->type == FG_CALC_NONE)
    {
      out->value.d = data->border;
      return out->type = FG_CALC_FLOAT;
    }
    if(out->type != FG_CALC_FLOAT)
      break;
    data->border = out->value.d;
    return FG_CALC_FLOAT;
  case FG_EVENT_BORDER_COLOR:
    if(out->type == FG_CALC_NONE)
    {
      out->value.i = data->borderColor.color;
      return out->type = FG_CALC_INT;
    }
    if(out->type != FG_CALC_INT)
      break;
    data->borderColor.color = out->value.i;
    return FG_CALC_INT;
  case FG_EVENT_BLUR:
    if(out->type == FG_CALC_NONE)
    {
      out->value.d = data->blur;
      return out->type = FG_CALC_FLOAT;
    }
    if(out->type != FG_CALC_FLOAT)
      break;
    data->blur = out->value.d;
    return FG_CALC_FLOAT;
  }

  return ~0;
}
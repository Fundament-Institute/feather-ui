// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/component/window.h"
#include "feather/outline.h"
#include <unordered_map>
#include <algorithm>
#include <string>

static const std::unordered_map<std::string, unsigned int> window_state_map = {
  { "title", FG_WINDOW_TITLE },
  { "icon", FG_WINDOW_ICON },
  { "left", FG_WINDOW_LEFT },
  { "top", FG_WINDOW_TOP },
  { "right", FG_WINDOW_RIGHT },
  { "bottom", FG_WINDOW_BOTTOM },
  { "state", FG_WINDOW_STATE },
};

static unsigned int window_fields[] = { FG_WINDOW_TITLE, FG_WINDOW_ICON };

extern "C" unsigned int fgWindowResolver(void* ptr, unsigned int index, fgCalcNode* out, const char* id)
{
  if(id)
  {
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = window_state_map.find(raw.c_str()); i != window_state_map.end())
      return i->second;
    return ~0;
  }
  if(!ptr)
  {
    out->value.p = window_fields;
    return sizeof(window_fields) / sizeof(unsigned int);
  }

  auto data = reinterpret_cast<fgWindowData*>(ptr);
  switch (index)
  {
  case FG_WINDOW_TITLE:
    if (out->type == FG_CALC_NONE)
    {
      out->value.s = data->title;
      return out->type = FG_CALC_STRING;
    }
    if(out->type != FG_CALC_STRING)
      break;
    data->title = out->value.s;
    return FG_CALC_STRING;
  case FG_WINDOW_ICON:
    if (out->type == FG_CALC_NONE)
    {
      out->value.p = (void*)data->icon;
      return out->type = FG_CALC_STRING;
    }
    if(out->type != FG_CALC_STRING)
      break;
    data->icon = out->value.s;
    return FG_CALC_STRING;
  }

  return ~0;
}

extern "C" fgMessageResult fgWindowBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto state = reinterpret_cast<fgWindowState*>(node->state);
  auto data = reinterpret_cast<fgWindowData*>(node->outline->statedata);

  switch(msg->type)
  {
  case FG_MSG_CONSTRUCT:
    if(!msg->construct.ptr)
      return fgMessageResult{ .construct = { sizeof(fgWindowState) } };

    *reinterpret_cast<fgWindowState*>(msg->construct.ptr) = { 0 };
    break;
  }

  return fgDefaultBehavior(root, node, msg);
}

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
  { "rect", FG_WINDOW_RECT },
  { "state", FG_WINDOW_STATE },
};

static const char* window_fields[] = { "title", "icon" };

unsigned char fgWindowResolver(void* ptr, const char* id, fgCalcResult* out, int* n)
{
  if(!id)
  {
    out->p = window_fields;
    *n = sizeof(window_fields) / sizeof(const char*);
  }
  else
  {
    auto data = reinterpret_cast<fgWindowData*>(ptr);
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = window_state_map.find(raw.c_str()); i != window_state_map.end())
    {
      switch(i->second)
      {
      case FG_WINDOW_TITLE:
        if(!n)
        {
          out->s = data->title;
          return FG_CALC_STRING;
        }
        if(*n != FG_CALC_STRING)
          return ~0;
        data->title = out->s;
        return FG_CALC_STRING;
      case FG_WINDOW_ICON:
        if(!n)
        {
          out->p = (void*)data->icon;
          return FG_CALC_STRING;
        }
        if(*n != FG_CALC_STRING)
          return ~0;
        data->icon = out->s;
        return FG_CALC_STRING;
      default:
        return ~0;
      }
    }
    else
      return ~0;
  }
  return 0;
}

fgMessageResult fgWindowBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto state = reinterpret_cast<fgWindowState*>(node->state);
  auto data = reinterpret_cast<fgWindowData*>(node->outline->data);

  switch(msg->type)
  {
  case FG_MSG_CONSTRUCT:
    if(!msg->construct.ptr)
      return fgMessageResult{ .construct = { sizeof(fgWindowState) } };

    *reinterpret_cast<fgWindowState*>(msg->construct.ptr) = { 0 };
    break;
  case FG_MSG_STATE_GET:
  case FG_MSG_STATE_SET:
  {
    int subtype = msg->subtype;

    if(!subtype)
    {
      std::string raw(msg->getState.id);
      std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

      if(auto i = window_state_map.find(raw.c_str()); i != window_state_map.end())
        subtype = i->second;
    }

    if(msg->type == FG_MSG_STATE_GET)
    {
      switch(subtype)
      {
      case FG_WINDOW_RECT:
        msg->getState.value->p = &state->area;
        break;
      case FG_WINDOW_STATE:
        msg->getState.value->i = state->flags;
        break;
      default:
        return fgDefaultBehavior(root, node, msg);
      }
      return fgMessageResult{ 0 };
    }
    else
    {
      switch(subtype)
      {
      case FG_WINDOW_RECT:
        state->area = *reinterpret_cast<fgRect*>(msg->setState.value.p);
        break;
      case FG_WINDOW_STATE:
        state->flags = msg->setState.value.i;
        break;
      case FG_WINDOW_TITLE:
        data->title = msg->setState.value.s;
        break;
      case FG_WINDOW_ICON:
        data->icon = msg->setState.value.p;
        break;
      default:
        return fgMessageResult{ -1 };
      }
      return fgMessageResult{ 0 };
    }
  }
    break;
  }

  return fgDefaultBehavior(root, node, msg);
}

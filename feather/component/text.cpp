// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/component/text.h"
#include "feather/outline.h"
#include "feather/root.h"
#include <unordered_map>
#include <algorithm>
#include <string>

extern "C" fgMessageResult fgTextBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg)
{
  auto data = reinterpret_cast<fgTextData*>(node->outline->statedata);
  auto state = reinterpret_cast<fgTextState*>(node->state);

  switch(msg->type)
  {
  case FG_MSG_CONSTRUCT:
    if(!msg->construct.ptr)
      return fgMessageResult{ .construct = { sizeof(fgTextState) } };

    *reinterpret_cast<fgTextState*>(msg->construct.ptr) = { 0 };
    break;
  case FG_MSG_DRAW:
    if(!state->font)
      state->font = (*root->backend->createFont)(root->backend, data->family, data->weight, data->italic, (unsigned int)node->outline->fontSize, fgVec { 96, 96 });
    if(!state->layout)
    {
      fgRect area = msg->draw.area;
      state->layout = (*root->backend->fontLayout)(root->backend, state->font, data->text, &area, node->outline->lineHeight, data->letterSpacing, state->layout, fgVec{ 96, 96 });
    }
    (*root->backend->drawFont)(root->backend, msg->draw.data, state->font, state->layout, &msg->draw.area, data->color, node->outline->lineHeight, data->letterSpacing, data->blur, data->aa);
    break;
  }

  return fgMessageResult{ 0 };
}

static const std::unordered_map<std::string, unsigned int> text_state_map = {
  { "font-family", FG_TEXT_FAMILY, },
  { "font-weight", FG_TEXT_WEIGHT },
  { "letter-spacing", FG_TEXT_LETTER_SPACING },
  { "text", FG_EVENT_TEXT },
  { "fillcolor", FG_EVENT_FILL_COLOR },
  { "blur", FG_EVENT_BLUR },
};

static unsigned int text_fields[] = { FG_TEXT_FAMILY, FG_TEXT_WEIGHT, FG_TEXT_LETTER_SPACING, FG_EVENT_TEXT, FG_EVENT_FILL_COLOR, FG_EVENT_BLUR };

extern "C" unsigned int fgTextResolver(void* ptr, unsigned int index, fgCalcNode* out, const char* id)
{
  if(id)
  {
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = text_state_map.find(raw.c_str()); i != text_state_map.end())
      return i->second;
    return ~0;
  }
  if(!ptr)
  {
    out->value.p = text_fields;
    return sizeof(text_fields) / sizeof(unsigned int);
  }

  auto data = reinterpret_cast<fgTextData*>(ptr);
  switch(index)
  {
  case FG_TEXT_FAMILY:
    if(out->type == FG_CALC_NONE)
    {
      out->value.s = data->family;
      return out->type = FG_CALC_STRING;
    }
    if(out->type != FG_CALC_STRING)
      break;
    data->family = out->value.s;
    return FG_CALC_STRING;
  case FG_TEXT_ITALIC:
    if(out->type == FG_CALC_NONE)
    {
      out->value.i = data->italic;
      return out->type = FG_CALC_INT;
    }
    if(out->type != FG_CALC_INT)
      break;
    data->italic = out->value.i != 0;
    return FG_CALC_INT;
  case FG_TEXT_WEIGHT:
    if(out->type == FG_CALC_NONE)
    {
      out->value.i = data->weight;
      return out->type = FG_CALC_INT;
    }
    if(out->type != FG_CALC_INT)
      break;
    data->weight = out->value.i;
    return FG_CALC_INT;
  case FG_TEXT_LETTER_SPACING:
    if(out->type == FG_CALC_NONE)
    {
      out->value.d = data->letterSpacing;
      return out->type = FG_CALC_FLOAT;
    }
    if(out->type != FG_CALC_FLOAT)
      break;
    data->letterSpacing = out->value.d;
    return FG_CALC_FLOAT;
  case FG_EVENT_TEXT:
    if(out->type == FG_CALC_NONE)
    {
      out->value.s = data->text;
      return out->type = FG_CALC_STRING;
    }
    if(out->type != FG_CALC_STRING)
      break;
    data->text = out->value.s;
    return FG_CALC_STRING;
  case FG_EVENT_FILL_COLOR:
    if(out->type == FG_CALC_NONE)
    {
      out->value.i = data->color.color;
      return out->type = FG_CALC_INT;
    }
    if(out->type != FG_CALC_INT)
      break;
    data->color.color = out->value.i;
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
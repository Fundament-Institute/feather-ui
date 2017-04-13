// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSkin.h"
#include "fgAll.h"
#include "feathercpp.h"
#include "bss-util/cStr.h"
#include "bss-util/cXML.h"

using namespace bss_util;

void fgSkinBase_WriteFloat(cStr& s, float abs)
{
  float i;
  if(modf(abs, &i) == 0.0f)
    return fgSkinBase_WriteInt(s, fFastTruncate(i));
  int len = snprintf(0, 0, "%.2f", abs) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%.1f", abs); // snprintf lies about how many characters were written
  s.resize(strlen(s));
}
void fgSkinBase_WriteInt(cStr& s, int64_t i)
{
  int len = snprintf(0, 0, "%lli", i) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%lli", i);
  s.resize(strlen(s));
}
void fgSkinBase_WriteHex(cStr& s, uint64_t i)
{
  int len = snprintf(0, 0, "%08llX", i) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%08llX", i);
  s.resize(strlen(s));
}

void fgSkinBase_WriteAbsXML(cStr& s, float abs, short unit)
{
  fgSkinBase_WriteFloat(s, abs);
  if(!abs)
    return;
  switch(unit)
  {
  case FGUNIT_SP: s += "sp"; break;
  case FGUNIT_EM: s += "em"; break;
  case FGUNIT_PX: s += "px"; break;
  }
}
void fgSkinBase_WriteCoordXML(cStr& s, const Coord& coord, short unit)
{
  if(coord.rel == 0.0f)
    fgSkinBase_WriteAbsXML(s, coord.abs, unit);
  else if(coord.abs == 0.0f)
  {
    fgSkinBase_WriteFloat(s, coord.rel * 100);
    s += "%";
  }
  else
  {
    fgSkinBase_WriteFloat(s, coord.abs);
    s += ":";
    fgSkinBase_WriteFloat(s, coord.rel);
  }
}
cStr fgSkinBase_WriteCVecXML(const CVec& vec, short units)
{
  cStr s;
  fgSkinBase_WriteCoordXML(s, vec.x, (units&FGUNIT_X_MASK) >> FGUNIT_X);
  s += ' ';
  fgSkinBase_WriteCoordXML(s, vec.y, (units&FGUNIT_Y_MASK) >> FGUNIT_Y);
  return s;
}

cStr fgSkinBase_WriteAbsRectXML(const AbsRect& r, short units)
{
  cStr s;
  fgSkinBase_WriteAbsXML(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgSkinBase_WriteAbsXML(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgSkinBase_WriteAbsXML(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgSkinBase_WriteAbsXML(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

cStr fgSkinBase_WriteCRectXML(const CRect& r, short units)
{
  cStr s;
  fgSkinBase_WriteCoordXML(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgSkinBase_WriteCoordXML(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgSkinBase_WriteCoordXML(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgSkinBase_WriteCoordXML(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

void fgSkinBase_WriteTransformXML(cXMLNode* node, const fgTransform& tf, short units)
{
  static const CVec EMPTYVEC = { 0 };
  if(memcmp(&tf.area, &CRect_EMPTY, sizeof(CRect)) != 0)
    node->AddAttribute("area")->String = fgSkinBase_WriteCRectXML(tf.area, units);
  if(tf.rotation != 0)
    fgSkinBase_WriteFloat(node->AddAttribute("rotation")->String, tf.rotation);
  if(memcmp(&tf.center, &EMPTYVEC, sizeof(CVec)) != 0)
    node->AddAttribute("center")->String = fgSkinBase_WriteCVecXML(tf.center, units);
}
void fgSkinBase_WriteFlagsXMLIterate(cStr& s, const char* type, fgFlag flags, bool remove)
{
  const fgFlag MAXBITS = 3;
  const fgFlag END = (sizeof(fgFlag) << 3) - MAXBITS;
  const char* str;

  for(fgFlag index = 0; index < END; ++index)
  {
    fgFlag bits = 0;
    for(fgFlag i = 0; i < MAXBITS; ++i)
      bits |= (1 << (i + index));
    for(fgFlag i = MAXBITS; i-- > 0;)
    {
      if((str = fgroot_instance->backend.fgFlagMap(type, bits&flags)) != 0)
      {
        s += remove ? "|-" : "|";
        s += str;
        index += i;
        break;
      }
      bits ^= (1 << (i + index));
    }
  }
}

void fgSkinBase_WriteFlagsXML(cXMLNode* node, const char* type, fgFlag flags)
{
  fgFlag def = fgGetTypeFlags(type);
  fgFlag rm = def & (~flags);
  fgFlag add = (~def) & flags;

  cStr s;
  fgSkinBase_WriteFlagsXMLIterate(s, type, add, false);
  fgSkinBase_WriteFlagsXMLIterate(s, type, rm, true);
  if(s.length() > 0)
    node->AddAttribute("flags")->String = s.c_str() + 1; // strip initial '|'
}
void fgSkinBase_WriteStyleAttributesXML(cXMLNode* node, fgStyle& s, fgSkinBase* root)
{
  static const char* COLORATTR[8] = { "color", "placecolor", "cursorcolor", "selectcolor", "hovercolor", "dragcolor", "edgecolor", "dividercolor" };

  fgStyleMsg* cur = s.styles;
  while(cur)
  {
    switch(cur->msg.type)
    {
    case FG_SETSKIN:
      if(cur->msg.p)
        node->AddAttribute("skin")->String = reinterpret_cast<fgSkin*>(cur->msg.p)->name;
      break;
    case FG_SETALPHA:
      fgSkinBase_WriteFloat(node->AddAttribute("alpha")->String, cur->msg.f);
      break;
    case FG_SETMARGIN:
      node->AddAttribute("margin")->String = fgSkinBase_WriteAbsRectXML(*(AbsRect*)cur->msg.p, cur->msg.subtype);
      break;
    case FG_SETPADDING:
      node->AddAttribute("padding")->String = fgSkinBase_WriteAbsRectXML(*(AbsRect*)cur->msg.p, cur->msg.subtype);
      break;
    case FG_SETTEXT:
      switch(cur->msg.subtype & 3)
      {
      case FGTEXTFMT_UTF8:
        node->AddAttribute((cur->msg.subtype&FGTEXTFMT_PLACEHOLDER_UTF8) ? "placeholder" : "text")->String = (const char*)cur->msg.p;
        break;
      case FGTEXTFMT_UTF16:
        node->AddAttribute((cur->msg.subtype&FGTEXTFMT_PLACEHOLDER_UTF8) ? "placeholder" : "text")->String = (const char*)cur->msg.p;
        break;
      case FGTEXTFMT_UTF32:
        node->AddAttribute((cur->msg.subtype&FGTEXTFMT_PLACEHOLDER_UTF8) ? "placeholder" : "text")->String = (const char*)cur->msg.p;
        break;
      }
      break;
    case FG_SETCOLOR:
      if(cur->msg.subtype < 8)
        fgSkinBase_WriteHex(node->AddAttribute(COLORATTR[cur->msg.subtype])->String, cur->msg.u);
      break;
    case FG_SETFONT:
      if(cur->msg.p)
      {
        _FG_FONT_DATA* p = root->GetFont(cur->msg.p);
        if(p)
        {
          cStr& s = node->AddAttribute("font")->String;
          fgSkinBase_WriteInt(s, p->size);
          if(p->weight != 400)
          {
            s += ' ';
            fgSkinBase_WriteInt(s, p->weight);
          }
          if(p->italic)
            s += " italic";
          s += p->family;
        }
      }
      break;
    case FG_SETLINEHEIGHT:
      fgSkinBase_WriteFloat(node->AddAttribute("lineheight")->String, cur->msg.f);
      break;
    case FG_SETLETTERSPACING:
      fgSkinBase_WriteFloat(node->AddAttribute("letterspacing")->String, cur->msg.f);
      break;
    case FG_SETVALUE:
      switch(cur->msg.subtype)
      {
      case FGVALUE_FLOAT:
        fgSkinBase_WriteFloat(node->AddAttribute("value")->String, cur->msg.f);
        break;
      case FGVALUE_INT64:
        fgSkinBase_WriteInt(node->AddAttribute("value")->String, cur->msg.i);
        break;
      }
      break;
    case FG_SETUV:
      node->AddAttribute("uv")->String = fgSkinBase_WriteCRectXML(*(CRect*)cur->msg.p, 0);
      break;
    case FG_SETASSET:
      if(cur->msg.p)
      {
        _FG_ASSET_DATA* p = root->GetAsset(cur->msg.p);
        if(p)
          node->AddAttribute("asset")->String = p->file;
      }
      break;
    case FG_SETRANGE:
      fgSkinBase_WriteFloat(node->AddAttribute("range")->String, cur->msg.f);
      break;
    case FG_SETOUTLINE:
      fgSkinBase_WriteFloat(node->AddAttribute("outline")->String, cur->msg.f);
      break;
    case FG_SETUSERDATA:
      node->AddAttribute((const char*)cur->msg.p)->String = (const char*)cur->msg.p2;
      break;
    case FG_SETDIM:
      switch(cur->msg.subtype)
      {
      case FGDIM_MAX:
        fgSkinBase_WriteFloat(node->AddAttribute("max-width")->String, cur->msg.f);
        fgSkinBase_WriteFloat(node->AddAttribute("max-height")->String, cur->msg.f2);
        break;
      case FGDIM_MIN:
        fgSkinBase_WriteFloat(node->AddAttribute("min-width")->String, cur->msg.f);
        fgSkinBase_WriteFloat(node->AddAttribute("min-height")->String, cur->msg.f2);
        break;
      }
      break;
    }
    cur = cur->next;
  }
}
void fgSkinBase_WriteElementAttributesXML(cXMLNode* node, fgSkinElement& e, fgSkinBase* root)
{
  if(e.order != 0)
    fgSkinBase_WriteInt(node->AddAttribute("order")->String, e.order);
  fgSkinBase_WriteFlagsXML(node, e.type, e.flags);
  fgSkinBase_WriteStyleAttributesXML(node, e.style, root);
  fgSkinBase_WriteTransformXML(node, e.transform, e.units);
}
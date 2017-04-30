// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSkin.h"
#include "fgAll.h"
#include "feathercpp.h"
#include "bss-util/cStr.h"
#include "bss-util/cXML.h"

using namespace bss_util;

void fgStyle_WriteFloat(cStr& s, float abs)
{
  float i;
  if(std::modf(abs, &i) == 0.0f)
    return fgStyle_WriteInt(s, fFastTruncate(i));
  int len = snprintf(0, 0, "%.2f", abs) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%.1f", abs); // snprintf lies about how many characters were written
  s.resize(strlen(s));
}
void fgStyle_WriteInt(cStr& s, int64_t i)
{
  int len = snprintf(0, 0, "%lli", i) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%lli", i);
  s.resize(strlen(s));
}
void fgStyle_WriteHex(cStr& s, uint64_t i)
{
  int len = snprintf(0, 0, "%08llX", i) + 1;
  int start = s.size();
  s.resize(start + len);
  snprintf(s.UnsafeString() + start, len, "%08llX", i);
  s.resize(strlen(s));
}

void fgStyle_WriteAbs(cStr& s, float abs, short unit)
{
  fgStyle_WriteFloat(s, abs);
  if(!abs)
    return;
  switch(unit)
  {
  case FGUNIT_SP: s += "sp"; break;
  case FGUNIT_EM: s += "em"; break;
  case FGUNIT_PX: s += "px"; break;
  }
}
void fgStyle_WriteCoord(cStr& s, const Coord& coord, short unit)
{
  if(coord.rel == 0.0f)
    fgStyle_WriteAbs(s, coord.abs, unit);
  else if(coord.abs == 0.0f)
  {
    fgStyle_WriteFloat(s, coord.rel * 100);
    s += "%";
  }
  else
  {
    fgStyle_WriteFloat(s, coord.abs);
    s += ":";
    fgStyle_WriteFloat(s, coord.rel);
  }
}
cStr fgStyle_WriteCVec(const CVec& vec, short units)
{
  cStr s;
  fgStyle_WriteCoord(s, vec.x, (units&FGUNIT_X_MASK) >> FGUNIT_X);
  s += ' ';
  fgStyle_WriteCoord(s, vec.y, (units&FGUNIT_Y_MASK) >> FGUNIT_Y);
  return s;
}

cStr fgStyle_WriteAbsRect(const AbsRect& r, short units)
{
  cStr s;
  fgStyle_WriteAbs(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgStyle_WriteAbs(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgStyle_WriteAbs(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgStyle_WriteAbs(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

cStr fgStyle_WriteCRect(const CRect& r, short units)
{
  cStr s;
  fgStyle_WriteCoord(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgStyle_WriteCoord(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgStyle_WriteCoord(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgStyle_WriteCoord(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

void fgSkinBase_WriteTransform(cXMLNode* node, const fgTransform& tf, short units)
{
  static const CVec EMPTYVEC = { 0 };
  if(memcmp(&tf.area, &CRect_EMPTY, sizeof(CRect)) != 0)
    node->AddAttribute("area")->String = fgStyle_WriteCRect(tf.area, units);
  if(tf.rotation != 0)
    fgStyle_WriteFloat(node->AddAttribute("rotation")->String, tf.rotation);
  if(memcmp(&tf.center, &EMPTYVEC, sizeof(CVec)) != 0)
    node->AddAttribute("center")->String = fgStyle_WriteCVec(tf.center, units);
}
void fgStyle_WriteFlagsIterate(cStr& s, const char* type, const char* divider, fgFlag flags, bool remove)
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
        s += divider;
        if(remove) s += "-";
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
  fgStyle_WriteFlagsIterate(s, type, "|", add, false);
  fgStyle_WriteFlagsIterate(s, type, "|", rm, true);
  if(s.length() > 0)
    node->AddAttribute("flags")->String = s.c_str() + 1; // strip initial '|'
}
void fgSkinBase_WriteStyleAttributesXML(cXMLNode* node, fgStyle& s, fgSkinBase* root)
{
  static const char* COLORATTR[10] = { "color", "placecolor", "cursorcolor", "selectcolor", "hovercolor", "dragcolor", "edgecolor", "dividercolor", "columndividercolor", "rowevencolor" };

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
      fgStyle_WriteFloat(node->AddAttribute("alpha")->String, cur->msg.f);
      break;
    case FG_SETMARGIN:
      node->AddAttribute("margin")->String = fgStyle_WriteAbsRect(*(AbsRect*)cur->msg.p, cur->msg.subtype);
      break;
    case FG_SETPADDING:
      node->AddAttribute("padding")->String = fgStyle_WriteAbsRect(*(AbsRect*)cur->msg.p, cur->msg.subtype);
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
        fgStyle_WriteHex(node->AddAttribute(COLORATTR[cur->msg.subtype])->String, cur->msg.u);
      break;
    case FG_SETFONT:
      if(cur->msg.p)
      {
        _FG_FONT_DATA* p = root->GetFont(cur->msg.p);
        if(p)
        {
          cStr& s = node->AddAttribute("font")->String;
          fgStyle_WriteInt(s, p->size);
          if(p->weight != 400)
          {
            s += ' ';
            fgStyle_WriteInt(s, p->weight);
          }
          if(p->italic)
            s += " italic";
          s += p->family;
        }
      }
      break;
    case FG_SETLINEHEIGHT:
      fgStyle_WriteFloat(node->AddAttribute("lineheight")->String, cur->msg.f);
      break;
    case FG_SETLETTERSPACING:
      fgStyle_WriteFloat(node->AddAttribute("letterspacing")->String, cur->msg.f);
      break;
    case FG_SETVALUE:
      switch(cur->msg.subtype)
      {
      case FGVALUE_FLOAT:
        fgStyle_WriteFloat(node->AddAttribute("value")->String, cur->msg.f);
        break;
      case FGVALUE_INT64:
        fgStyle_WriteInt(node->AddAttribute("value")->String, cur->msg.i);
        break;
      }
      break;
    case FG_SETUV:
      node->AddAttribute("uv")->String = fgStyle_WriteCRect(*(CRect*)cur->msg.p, 0);
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
      fgStyle_WriteFloat(node->AddAttribute("range")->String, cur->msg.f);
      break;
    case FG_SETOUTLINE:
      fgStyle_WriteFloat(node->AddAttribute("outline")->String, cur->msg.f);
      break;
    case FG_SETUSERDATA:
      node->AddAttribute((const char*)cur->msg.p)->String = (const char*)cur->msg.p2;
      break;
    case FG_SETDIM:
      switch(cur->msg.subtype)
      {
      case FGDIM_MAX:
        fgStyle_WriteFloat(node->AddAttribute("max-width")->String, cur->msg.f);
        fgStyle_WriteFloat(node->AddAttribute("max-height")->String, cur->msg.f2);
        break;
      case FGDIM_MIN:
        fgStyle_WriteFloat(node->AddAttribute("min-width")->String, cur->msg.f);
        fgStyle_WriteFloat(node->AddAttribute("min-height")->String, cur->msg.f2);
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
    fgStyle_WriteInt(node->AddAttribute("order")->String, e.order);
  fgSkinBase_WriteFlagsXML(node, e.type, e.flags);
  fgSkinBase_WriteStyleAttributesXML(node, e.style, root);
  fgSkinBase_WriteTransform(node, e.transform, e.units);
}
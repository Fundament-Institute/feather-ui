// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "feathercpp.h"
#include "bss-util/cXML.h"
#include "bss-util/cTrie.h"

#include <fstream>
#include <sstream>

using namespace bss_util;

void fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void fgLayout_Destroy(fgLayout* self)
{
  fgSkinBase_Destroy(&self->base);
  fgStyle_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cArraySort();
}
FG_UINT fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgClassLayoutArray&)self->layout).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* fgLayout_GetLayout(const fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}

void fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  fgSkinElement_Init(&self->layout, type, flags, transform, units, order);
  self->name = fgCopyText(name, __FILE__, __LINE__);
  self->id = 0;
  memset(&self->children, 0, sizeof(fgVector));
  memset(&self->userdata, 0, sizeof(fgVector));
}
void fgClassLayout_Destroy(fgClassLayout* self)
{
  fgSkinElement_Destroy(&self->layout);
  if(self->name) fgFreeText(self->name, __FILE__, __LINE__);
  if(self->id) fgFreeText(self->id, __FILE__, __LINE__);
  reinterpret_cast<fgKeyValueArray&>(self->userdata).~cDynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cArraySort();
}

void fgClassLayout_AddUserString(fgClassLayout* self, const char* key, const char* value)
{
  size_t i = ((fgKeyValueArray&)(self->userdata)).AddConstruct();
  self->userdata.p[i].key = fgCopyText(key, __FILE__, __LINE__);
  self->userdata.p[i].value = fgCopyText(value, __FILE__, __LINE__);
}

FG_UINT fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgClassLayoutArray&)self->children).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child)
{
  return DynArrayRemove((fgClassLayoutArray&)self->children, child);
}
fgClassLayout* fgClassLayout_GetChild(const fgClassLayout* self, FG_UINT child)
{
  return self->children.p + child;
}
void fgLayout_LoadFileUBJSON(fgLayout* self, const char* file)
{
}
void fgLayout_LoadUBJSON(fgLayout* self, const char* data, FG_UINT length)
{

}

void fgLayout_SaveFileUBJSON(fgLayout* self, const char* file)
{

}

void fgClassLayout_LoadAttributesXML(fgClassLayout* self, const cXMLNode* cur, int flags, fgSkinBase* root, const char* path)
{
  fgStyle_ParseAttributesXML(&self->layout.style, cur, flags, root, path, &self->id, (fgKeyValueArray*)&self->userdata);
}

void fgClassLayout_LoadLayoutXML(fgClassLayout* self, const cXMLNode* cur, fgLayout* root, const char* path)
{
  for(size_t i = 0; i < cur->GetNodes(); ++i)
  { 
    const cXMLNode* node = cur->GetNode(i);
    fgTransform transform = { 0 };
    int type = fgStyle_NodeEvalTransform(node, transform);
    fgFlag rmflags = 0;
    fgFlag flags = fgSkinBase_ParseFlagsFromString(node->GetAttributeString("flags"), &rmflags);
    flags = (flags | fgGetTypeFlags(node->GetName()))&(~rmflags);

    fgTransform TF_SEPERATOR = { { 0,0,0,0,0,1.0,0,0 }, 0,{ 0,0,0,0 } };
    if(!STRICMP(node->GetName(), "menuitem") && !node->GetNodes() && !node->GetAttributeString("text")) // An empty menuitem is a special case
      fgClassLayout_AddChild(self, "Element", "Submenu$seperator", FGELEMENT_IGNORE, &TF_SEPERATOR, 0, (int)node->GetAttributeInt("order"));
    else
    {
      FG_UINT index = fgClassLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, (int)node->GetAttributeInt("order"));
      fgClassLayout* layout = fgClassLayout_GetChild(self, index);
      fgClassLayout_LoadAttributesXML(layout, node, flags, &root->base, path);
      fgClassLayout_LoadLayoutXML(layout, node, root, path);
    }
  }
}
bool fgLayout_LoadStreamXML(fgLayout* self, std::istream& s, const char* path)
{
  cXML xml(s);
  size_t nodes = xml.GetNodes();
  for(size_t i = 0; i < nodes; ++i) // Load any skins contained in the layout first
    if(!STRICMP(xml.GetNode(i)->GetName(), "fg:skin"))
      fgSkinBase_ParseNodeXML(&self->base, xml.GetNode(i));

  const cXMLNode* root = xml.GetNode("fg:Layout");
  if(!root)
    return false;

  fgStyle_ParseAttributesXML(&self->style, root, 0, &self->base, 0, 0, 0); // This will load and apply skins to the layout base

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const cXMLNode* node = root->GetNode(i);
    fgTransform transform = { 0 };
    short type = fgStyle_NodeEvalTransform(node, transform);
    fgFlag rmflags = 0;
    fgFlag flags = fgSkinBase_ParseFlagsFromString(node->GetAttributeString("flags"), &rmflags);
    flags = (flags | fgGetTypeFlags(node->GetName()))&(~rmflags);
    FG_UINT index = fgLayout_AddLayout(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, (int)node->GetAttributeInt("order"));
    fgClassLayout* layout = fgLayout_GetLayout(self, index);
    fgClassLayout_LoadAttributesXML(layout, node, flags, &self->base, path);
    fgClassLayout_LoadLayoutXML(layout, node, self, path);
  }

  return true;
}
char fgLayout_LoadFileXML(fgLayout* self, const char* file)
{
  cStr path(file);
  path.ReplaceChar('\\', '/');
  char* dir = strrchr(path.UnsafeString(), '/');
  if(dir) // If we find a /, we chop off the rest of the string AFTER it, so it becomes a valid directory path.
    dir[1] = 0;
  std::ifstream fs(file, std::ios_base::in|std::ios_base::binary);
  return fgLayout_LoadStreamXML(self, fs, path.c_str());
}

char fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length)
{
  std::stringstream ss(std::string(data, length));
  return fgLayout_LoadStreamXML(self, ss, 0);
}

void fgLayout_WriteFloat(cStr& s, float abs)
{
  int len = snprintf(0, 0, "%f", abs);
  int start = s.size();
  s.resize(len + start);
  snprintf(s.UnsafeString(), len, "%f", abs);
}
void fgLayout_WriteInt(cStr& s, int64_t i)
{
  int len = snprintf(0, 0, "%lli", i);
  int start = s.size();
  s.resize(len + start);
  snprintf(s.UnsafeString(), len, "%lli", i);
}
void fgLayout_WriteHex(cStr& s, uint64_t i)
{
  int len = snprintf(0, 0, "%08llX", i);
  int start = s.size();
  s.resize(len + start);
  snprintf(s.UnsafeString(), len, "%08llX", i);
}

void fgLayout_WriteAbsXML(cStr& s, float abs, short unit)
{
  fgLayout_WriteFloat(s, abs);
  if(!abs)
    return;
  switch(unit)
  {
  case FGUNIT_SP: s += "sp"; break;
  case FGUNIT_EM: s += "em"; break;
  case FGUNIT_PX: s += "px"; break;
  }
}
void fgLayout_WriteCoordXML(cStr& s, const Coord& coord, short unit)
{
  if(coord.rel == 0.0f)
    fgLayout_WriteAbsXML(s, coord.abs, unit);
  else if(coord.abs == 0.0f)
  {
    fgLayout_WriteFloat(s, coord.rel * 100);
    s += "%";
  }
  else
  {
    fgLayout_WriteFloat(s, coord.abs);
    s += ":";
    fgLayout_WriteFloat(s, coord.rel);
  }
}
cStr fgLayout_WriteCVecXML(const CVec& vec, short units)
{
  cStr s;
  fgLayout_WriteCoordXML(s, vec.x, (units&FGUNIT_X_MASK) >> FGUNIT_X);
  s += ' ';
  fgLayout_WriteCoordXML(s, vec.y, (units&FGUNIT_Y_MASK) >> FGUNIT_Y);
  return s;
}

cStr fgLayout_WriteAbsRectXML(const AbsRect& r, short units)
{
  cStr s;
  fgLayout_WriteAbsXML(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgLayout_WriteAbsXML(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgLayout_WriteAbsXML(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgLayout_WriteAbsXML(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

cStr fgLayout_WriteCRectXML(const CRect& r, short units)
{
  cStr s;
  fgLayout_WriteCoordXML(s, r.left, (units&FGUNIT_LEFT_MASK) >> FGUNIT_LEFT);
  s += ' ';
  fgLayout_WriteCoordXML(s, r.top, (units&FGUNIT_TOP_MASK) >> FGUNIT_TOP);
  s += ' ';
  fgLayout_WriteCoordXML(s, r.right, (units&FGUNIT_RIGHT_MASK) >> FGUNIT_RIGHT);
  s += ' ';
  fgLayout_WriteCoordXML(s, r.bottom, (units&FGUNIT_BOTTOM_MASK) >> FGUNIT_BOTTOM);
  return s;
}

void fgLayout_WriteTransformXML(cXMLNode* node, const fgTransform& tf, short units)
{
  static const CVec EMPTYVEC = { 0 };
  node->AddAttribute("area")->String = fgLayout_WriteCRectXML(tf.area, units);
  if(tf.rotation != 0)
    fgLayout_WriteFloat(node->AddAttribute("rotation")->String, tf.rotation);
  if(memcmp(&tf.center, &EMPTYVEC, sizeof(CVec)) != 0)
    node->AddAttribute("center")->String = fgLayout_WriteCVecXML(tf.center, units);
}
void fgLayout_WriteFlagsXMLIterate(cStr& s, const char* type, fgFlag flags, bool remove)
{
  const fgFlag MAXBITS = 3;
  const fgFlag END = (sizeof(fgFlag) << 3) - MAXBITS;
  const char* str;
  for(fgFlag index = 0; index < END; ++index)
  {
    // Checks for a maximum of a 3 bit mask flag combination
    fgFlag bits = 0;
    for(fgFlag i = 0; i < MAXBITS; ++i)
      bits |= (1 << (i + index));
    for(fgFlag i = MAXBITS; i-- > 0;)
    {
      if((str = fgroot_instance->backend.fgFlagMap(type, bits)) != 0)
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

void fgLayout_WriteFlagsXML(cXMLNode* node, const char* type, fgFlag flags)
{
  fgFlag def = fgGetTypeFlags(type);
  fgFlag rm = def & (~flags);
  fgFlag add = (~def) & flags;

  cStr s;
  fgLayout_WriteFlagsXMLIterate(s, type, add, false);
  fgLayout_WriteFlagsXMLIterate(s, type, rm, true);
  if(s.length() > 0)
    node->AddAttribute("flags")->String = s.c_str() + 1; // strip initial '|'
}
void fgLayout_WriteStyleAttributesXML(cXMLNode* node, fgStyle& s, fgSkinBase* root)
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
      fgLayout_WriteFloat(node->AddAttribute("alpha")->String, cur->msg.f);
      break;
    case FG_SETMARGIN:
      node->AddAttribute("margin")->String = fgLayout_WriteAbsRectXML(*(AbsRect*)cur->msg.p, cur->msg.subtype);
      break;
    case FG_SETPADDING:
      node->AddAttribute("padding")->String = fgLayout_WriteAbsRectXML(*(AbsRect*)cur->msg.p, cur->msg.subtype);
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
        fgLayout_WriteHex(node->AddAttribute(COLORATTR[cur->msg.subtype])->String, cur->msg.u);
      break;
    case FG_SETFONT:
      if(cur->msg.p)
      {
        _FG_FONT_DATA* p = root->GetFont(cur->msg.p);
        if(p)
        {

        }
      }
      break;
    case FG_SETLINEHEIGHT:
      fgLayout_WriteFloat(node->AddAttribute("lineheight")->String, cur->msg.f);
      break;
    case FG_SETLETTERSPACING:
      fgLayout_WriteFloat(node->AddAttribute("letterspacing")->String, cur->msg.f);
      break;
    case FG_SETVALUE:
      switch(cur->msg.subtype)
      {
        case FGVALUE_FLOAT:
          fgLayout_WriteFloat(node->AddAttribute("value")->String, cur->msg.f);
          break;
        case FGVALUE_INT64:
          fgLayout_WriteInt(node->AddAttribute("value")->String, cur->msg.i);
          break;
      }
      break;
    case FG_SETUV:
      node->AddAttribute("uv")->String = fgLayout_WriteCRectXML(*(CRect*)cur->msg.p, 0);
      break;
    case FG_SETASSET:
      break;
    case FG_SETRANGE:
      fgLayout_WriteFloat(node->AddAttribute("range")->String, cur->msg.f);
      break;
    case FG_SETOUTLINE:
      fgLayout_WriteFloat(node->AddAttribute("outline")->String, cur->msg.f);
      break;
    case FG_SETUSERDATA:
      node->AddAttribute((const char*)cur->msg.p)->String = (const char*)cur->msg.p2;
      break;
    case FG_SETDIM:
      switch(cur->msg.subtype)
      {
      case FGDIM_MAX:
        fgLayout_WriteFloat(node->AddAttribute("max-width")->String, cur->msg.f);
        fgLayout_WriteFloat(node->AddAttribute("max-height")->String, cur->msg.f2);
        break;
      case FGDIM_MIN:
        fgLayout_WriteFloat(node->AddAttribute("min-width")->String, cur->msg.f);
        fgLayout_WriteFloat(node->AddAttribute("min-height")->String, cur->msg.f2);
        break;
      }
      break;
    }
    cur = cur->next;
  }
}
void fgLayout_WriteElementAttributesXML(cXMLNode* node, fgSkinElement& e)
{
  if(e.order != 0)
    fgLayout_WriteInt(node->AddAttribute("order")->String, e.order);
  fgLayout_WriteFlagsXML(node, e.type, e.flags);
  fgLayout_WriteStyleAttributesXML(node, e.style);
  fgLayout_WriteTransformXML(node, e.transform, e.units);
}
void fgLayout_SaveElementXML(fgLayout* self, fgClassLayout& e, cXMLNode* parent, const char* path)
{
  cXMLNode* node = parent->AddNode(e.layout.type);
  fgLayout_WriteElementAttributesXML(node, e.layout);
  if(e.id)
    node->AddAttribute("id")->String = e.id;
  if(e.name)
    node->AddAttribute("name")->String = e.name;
  for(size_t i = 0; i < e.userdata.l; ++i)
    node->AddAttribute(e.userdata.p[i].key)->String = e.userdata.p[i].value;

  for(size_t i = 0; i < e.children.l; ++i)
    fgLayout_SaveElementXML(self, e.children.p[i], node, path);
}
void fgLayout_SaveStreamXML(fgLayout* self, std::ostream& s, const char* path)
{
  cXML xml;
  xml.AddAttribute("version")->String = "1.0";
  xml.AddAttribute("encoding")->String = "UTF-8";
  
  // Write root element
  cXMLNode* root = xml.AddNode("fg:Layout");
  root->AddAttribute("xmlns:xsi")->String = "http://www.w3.org/2001/XMLSchema-instance";
  root->AddAttribute("xmlns:fg")->String = "featherGUI";
  root->AddAttribute("xsi:schemaLocation")->String = "featherGUI feather.xsd";
  fgLayout_WriteStyleAttributesXML(root, self->style);
  
  // Write all children nodes
  for(size_t i = 0; i < self->layout.l; ++i)
    fgLayout_SaveElementXML(self, self->layout.p[i], root, path);

  // Write any skins that were stored in the root

  xml.Write(s);
}
void fgLayout_SaveFileXML(fgLayout* self, const char* file)
{
  cStr path(file);
  path.ReplaceChar('\\', '/');
  char* dir = strrchr(path.UnsafeString(), '/');
  if(dir) // If we find a /, we chop off the rest of the string AFTER it, so it becomes a valid directory path.
    dir[1] = 0;
  std::ofstream fs(file, std::ios_base::out | std::ios_base::binary);
  return fgLayout_SaveStreamXML(self, fs, path.c_str());
}

void fgLayout_SaveElementXML(fgElement* root, const char* file)
{

}

FG_UINT fgLayout::AddLayout(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order) { return fgLayout_AddLayout(this, type, name, flags, transform, units, order); }
char fgLayout::RemoveLayout(FG_UINT layout) { return fgLayout_RemoveLayout(this, layout); }
fgClassLayout* fgLayout::GetLayout(FG_UINT layout) const { return fgLayout_GetLayout(this, layout); }

void fgLayout::LoadFileUBJSON(const char* file) { fgLayout_LoadFileUBJSON(this, file); }
void fgLayout::LoadUBJSON(const char* data, FG_UINT length) { fgLayout_LoadUBJSON(this, data, length); }
void fgLayout::SaveFileUBJSON(const char* file) { fgLayout_SaveFileUBJSON(this, file); }
char fgLayout::LoadFileXML(const char* file) { return fgLayout_LoadFileXML(this, file); }
char fgLayout::LoadXML(const char* data, FG_UINT length) { return fgLayout_LoadXML(this, data, length); }
void fgLayout::SaveFileXML(const char* file) { fgLayout_SaveFileXML(this, file); }

void fgClassLayout::AddUserString(const char* key, const char* value) { return fgClassLayout_AddUserString(this, key, value); }
FG_UINT fgClassLayout::AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order) { return fgClassLayout_AddChild(this, type, name, flags, transform, units, order); }
char fgClassLayout::RemoveChild(FG_UINT child) { return fgClassLayout_RemoveChild(this, child); }
fgClassLayout* fgClassLayout::GetChild(FG_UINT child) const { return fgClassLayout_GetChild(this, child); }
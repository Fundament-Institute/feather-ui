// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "feathercpp.h"
#include "bss-util/cXML.h"
#include "bss-util/cTrie.h"

#include <fstream>
#include <sstream>

using namespace bss_util;

KHASH_INIT(fgLayoutMap, const char*, fgLayout*, 1, kh_str_hash_funcins, kh_str_hash_insequal);

void fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void fgLayout_Destroy(fgLayout* self)
{
  fgSkinBase_Destroy(&self->base);
  fgStyle_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cArraySort();
  if(self->sublayouts != 0)
  {
    for(khint_t i = 0; i < kh_end(self->sublayouts); ++i)
    {
      if(kh_exist(self->sublayouts, i))
      {
        fgLayout_Destroy(kh_val(self->sublayouts, i));
        fgfree(kh_val(self->sublayouts, i), __FILE__, __LINE__);
        fgFreeText(kh_key(self->sublayouts, i), __FILE__, __LINE__);
      }
    }
    kh_destroy_fgLayoutMap(self->sublayouts);
  }
}
FG_UINT fgLayout_AddChild(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgClassLayoutArray&)self->layout).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char fgLayout_RemoveChild(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* fgLayout_GetChild(const fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}
fgLayout* fgLayout_AddLayout(fgLayout* self, const char* name)
{
  if(!self->sublayouts)
    self->sublayouts = kh_init_fgLayoutMap();

  int r = 0;
  khint_t i = kh_put_fgLayoutMap(self->sublayouts, name, &r);
  if(!r)
  {
    fgLog("Sublayout %s already exists.", name);
    return kh_val(self->sublayouts, i);
  }
  kh_key(self->sublayouts, i) = fgCopyText(name, __FILE__, __LINE__);
  kh_val(self->sublayouts, i) = fgmalloc<fgLayout>(1, __FILE__, __LINE__);
  fgLayout_Init(kh_val(self->sublayouts, i));
  return kh_val(self->sublayouts, i);
}
char fgLayout_RemoveLayout(fgLayout* self, const char* name)
{
  if(!self->sublayouts)
    return 0;
  khint_t i = kh_get_fgLayoutMap(self->sublayouts, name);
  if(i >= kh_end(self->sublayouts) || !kh_exist(self->sublayouts, i))
  {
    fgLog("Attempted to remove nonexistent sublayout %s", name);
    return 0;
  }
  kh_del_fgLayoutMap(self->sublayouts, i);
  return 1;
}
fgLayout* fgLayout_GetLayout(const fgLayout* self, const char* name)
{
  if(!self->sublayouts)
    return 0;
  khint_t i = kh_get_fgLayoutMap(self->sublayouts, name);
  return (i < kh_end(self->sublayouts) && kh_exist(self->sublayouts, i)) ? kh_val(self->sublayouts, i) : 0;
}

void fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  fgSkinElement_Init(&self->layout, type, flags, transform, units, order);
  self->name = fgCopyText(name, __FILE__, __LINE__);
  self->id = 0;
  memset(&self->children, 0, sizeof(fgVector));
  memset(&self->userdata, 0, sizeof(fgVector));
  self->userid = 0;
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
  const cXMLValue* userid = cur->GetAttribute("userid");
  if(userid)
    self->userid = userid->Integer;

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
void fgLayout_LoadLayoutNode(fgLayout* self, const cXMLNode* root, std::istream& s, const char* path)
{
  fgStyle_ParseAttributesXML(&self->style, root, 0, &self->base, 0, 0, 0); // This will load and apply skins to the layout base

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const cXMLNode* node = root->GetNode(i);
    if(!stricmp(node->GetName(), "layout")) // If it's a layout we need to load this as a sub-layout
    {
      const char* name = node->GetAttributeString("name");
      if(!name)
        fgLog("Sublayout failed to load because it had no name attribute.");
      else
        fgLayout_LoadLayoutNode(self->AddLayout(name), root, s, path);
    }
    else
    {
      fgTransform transform = { 0 };
      short type = fgStyle_NodeEvalTransform(node, transform);
      fgFlag rmflags = 0;
      fgFlag flags = fgSkinBase_ParseFlagsFromString(node->GetAttributeString("flags"), &rmflags);
      flags = (flags | fgGetTypeFlags(node->GetName()))&(~rmflags);
      FG_UINT index = fgLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, (int)node->GetAttributeInt("order"));
      fgClassLayout* layout = fgLayout_GetChild(self, index);
      fgClassLayout_LoadAttributesXML(layout, node, flags, &self->base, path);
      fgClassLayout_LoadLayoutXML(layout, node, self, path);
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
  
  fgLayout_LoadLayoutNode(self, root, s, path);
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

void fgLayout_SaveElementXML(fgLayout* self, fgClassLayout& e, cXMLNode* parent, const char* path)
{
  cXMLNode* node = parent->AddNode(e.layout.type);
  fgSkinBase_WriteElementAttributesXML(node, e.layout, &self->base);
  if(e.id)
    node->AddAttribute("id")->String = e.id;
  if(e.name)
    node->AddAttribute("name")->String = e.name;
  if(e.name)
    fgSkinBase_WriteInt(node->AddAttribute("userid")->String, e.userid);
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
  fgSkinBase_WriteStyleAttributesXML(root, self->style, &self->base);
  
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

FG_UINT fgLayout::AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order) { return fgLayout_AddChild(this, type, name, flags, transform, units, order); }
bool fgLayout::RemoveChild(FG_UINT layout) { return fgLayout_RemoveChild(this, layout) != 0; }
fgClassLayout* fgLayout::GetChild(FG_UINT layout) const { return fgLayout_GetChild(this, layout); }
fgLayout* fgLayout::AddLayout(const char* name) { return fgLayout_AddLayout(this, name); }
bool fgLayout::RemoveLayout(const char* name) { return fgLayout_RemoveLayout(this, name) != 0; }
fgLayout* fgLayout::GetLayout(const char* name) const { return fgLayout_GetLayout(this, name); }

void fgLayout::LoadFileUBJSON(const char* file) { fgLayout_LoadFileUBJSON(this, file); }
void fgLayout::LoadUBJSON(const char* data, FG_UINT length) { fgLayout_LoadUBJSON(this, data, length); }
void fgLayout::SaveFileUBJSON(const char* file) { fgLayout_SaveFileUBJSON(this, file); }
bool fgLayout::LoadFileXML(const char* file) { return fgLayout_LoadFileXML(this, file) != 0; }
bool fgLayout::LoadXML(const char* data, FG_UINT length) { return fgLayout_LoadXML(this, data, length) != 0; }
void fgLayout::SaveFileXML(const char* file) { fgLayout_SaveFileXML(this, file); }

void fgClassLayout::AddUserString(const char* key, const char* value) { return fgClassLayout_AddUserString(this, key, value); }
FG_UINT fgClassLayout::AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order) { return fgClassLayout_AddChild(this, type, name, flags, transform, units, order); }
bool fgClassLayout::RemoveChild(FG_UINT child) { return fgClassLayout_RemoveChild(this, child) != 0; }
fgClassLayout* fgClassLayout::GetChild(FG_UINT child) const { return fgClassLayout_GetChild(this, child); }

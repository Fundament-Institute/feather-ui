// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgLayout.h"
#include "bss-util/XML.h"
#include "bss-util/Trie.h"

#include <fstream>
#include <sstream>

using namespace bss;

KHASH_INIT(fgLayoutMap, const char*, fgLayout*, 1, kh_str_hash_funcins, kh_str_hash_insequal);

void fgLayout_Init(fgLayout* self, const char* name, const char* path)
{
  bssFill(*self, 0);
  self->base.name = fgCopyText(name, __FILE__, __LINE__);
  self->base.path = fgCopyText(path, __FILE__, __LINE__);
}
void fgLayout_Destroy(fgLayout* self)
{
  fgSkinBase_Destroy(&self->base);
  reinterpret_cast<fgClassLayoutArray&>(self->children).~ArraySort();
  if(self->sublayouts != 0)
  {
    for(khint_t i = 0; i < kh_end(self->sublayouts); ++i)
    {
      if(kh_exist(self->sublayouts, i))
      {
        fgLayout_Destroy(kh_val(self->sublayouts, i));
        fgfree(kh_val(self->sublayouts, i), __FILE__, __LINE__);
      }
    }
    kh_destroy_fgLayoutMap(self->sublayouts);
  }
}
FG_UINT fgLayout_AddChild(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
{
  return ((fgClassLayoutArray&)self->children).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char fgLayout_RemoveChild(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->children, layout);
}
fgClassLayout* fgLayout_GetChild(const fgLayout* self, FG_UINT layout)
{
  return self->children.p + layout;
}
fgLayout* fgLayout_AddLayout(fgLayout* self, const char* name)
{
  if(!self->sublayouts)
    self->sublayouts = kh_init_fgLayoutMap();

  int r = 0;
  khint_t i = kh_put_fgLayoutMap(self->sublayouts, name, &r);
  if(!r)
  {
    fgLog(FGLOG_WARNING, "Sublayout %s already exists.", name);
    return kh_val(self->sublayouts, i);
  }
  kh_val(self->sublayouts, i) = fgmalloc<fgLayout>(1, __FILE__, __LINE__);
  fgLayout_Init(kh_val(self->sublayouts, i), name, self->base.path);
  kh_val(self->sublayouts, i)->base.parent = &self->base;
  kh_key(self->sublayouts, i) = kh_val(self->sublayouts, i)->base.name;
  return kh_val(self->sublayouts, i);
}
char fgLayout_RemoveLayout(fgLayout* self, const char* name)
{
  if(!self->sublayouts)
    return 0;
  khint_t i = kh_get_fgLayoutMap(self->sublayouts, name);
  if(i >= kh_end(self->sublayouts) || !kh_exist(self->sublayouts, i))
  {
    fgLog(FGLOG_WARNING, "Attempted to remove nonexistent sublayout %s", name);
    return 0;
  }
  fgLayout_Destroy(kh_val(self->sublayouts, i));
  fgfree(kh_val(self->sublayouts, i), __FILE__, __LINE__);
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

void fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
{
  fgSkinElement_Init(&self->element, type, flags, transform, units, order);
  self->name = fgCopyText(name, __FILE__, __LINE__);
  self->id = 0;
  memset(&self->children, 0, sizeof(fgVector));
  memset(&self->userdata, 0, sizeof(fgVector));
  self->userid = 0;
}
void fgClassLayout_Destroy(fgClassLayout* self)
{
  fgSkinElement_Destroy(&self->element);
  if(self->name) fgFreeText(self->name, __FILE__, __LINE__);
  if(self->id) fgFreeText(self->id, __FILE__, __LINE__);
  reinterpret_cast<fgKeyValueArray&>(self->userdata).~DynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->children).~ArraySort();
}

void fgClassLayout_AddUserString(fgClassLayout* self, const char* key, const char* value)
{
  size_t i = ((fgKeyValueArray&)(self->userdata)).AddConstruct();
  self->userdata.p[i].key = fgCopyText(key, __FILE__, __LINE__);
  self->userdata.p[i].value = fgCopyText(value, __FILE__, __LINE__);
}

FG_UINT fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, int order)
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
void fgClassLayout_LoadAttributesXML(fgClassLayout* self, const XMLNode* cur, int flags, fgSkinBase* root)
{
  const XMLValue* userid = cur->GetAttribute("userid");
  if(userid)
    self->userid = userid->Integer;

  self->element.skin = fgStyle_ParseAttributesXML(&self->element.style, cur, flags, root, &self->id, (fgKeyValueArray*)&self->userdata);
}

void fgClassLayout_LoadLayoutXML(fgClassLayout* self, const XMLNode* cur, fgLayout* root)
{
  for(size_t i = 0; i < cur->GetNodes(); ++i)
  { 
    const XMLNode* node = cur->GetNode(i);
    fgTransform transform = { 0 };
    int type = fgStyle_NodeEvalTransform(node, transform, 0);
    fgFlag rmflags = 0;
    fgFlag flags = fgSkinBase_ParseFlagsFromString(node->GetAttributeString("flags"), &rmflags, '|', 0);
    flags = (flags | fgGetTypeFlags(node->GetName()))&(~rmflags);

    fgTransform TF_SEPERATOR = { { 0,0,0,0,0,1.0,0,0 }, 0,{ 0,0,0,0 } };
    if(!STRICMP(node->GetName(), "menuitem") && !node->GetNodes() && !node->GetAttributeString("text")) // An empty menuitem is a special case
      fgClassLayout_AddChild(self, "Element", "Submenu$seperator", FGELEMENT_IGNORE, &TF_SEPERATOR, 0, (int)node->GetAttributeInt("order"));
    else
    {
      FG_UINT index = fgClassLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, (int)node->GetAttributeInt("order"));
      fgClassLayout* layout = fgClassLayout_GetChild(self, index);
      fgClassLayout_LoadAttributesXML(layout, node, flags, &root->base);
      fgClassLayout_LoadLayoutXML(layout, node, root);
    }
  }
}
void fgLayout_LoadLayoutNode(fgLayout* self, fgSkinBase* parent, const XMLNode* root, std::istream& s)
{
  self->base.parent = parent;
  size_t nodes = root->GetNodes(); // We have to load any skins in the layout BEFORE we parse the attributes, otherwise the skin won't get applied
  for(size_t i = 0; i < nodes; ++i)
    if(!STRICMP(root->GetNode(i)->GetName(), "skin"))
      fgSkinBase_ParseNodeXML(&self->base, root->GetNode(i));

  self->skin = fgStyle_ParseAttributesXML(&self->base.style, root, 0, &self->base, 0, 0); // This will load and apply skins to the layout base

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const XMLNode* node = root->GetNode(i);
    if(!STRICMP(node->GetName(), "skin")) // We already loaded the skins so skip them
      continue;
    if(!STRICMP(node->GetName(), "layout")) // If it's a layout we need to load this as a sub-layout
    {
      const char* name = node->GetAttributeString("name");
      if(!name)
        fgLog(FGLOG_ERROR, "Sublayout failed to load because it had no name attribute.");
      else
        fgLayout_LoadLayoutNode(self->AddLayout(name), &self->base, node, s);
    }
    else
    {
      fgTransform transform = { 0 };
      fgMsgType type = fgStyle_NodeEvalTransform(node, transform, 0);
      fgFlag rmflags = 0;
      fgFlag flags = fgSkinBase_ParseFlagsFromString(node->GetAttributeString("flags"), &rmflags, '|', 0);
      flags = (flags | fgGetTypeFlags(node->GetName()))&(~rmflags);
      FG_UINT index = fgLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, (int)node->GetAttributeInt("order"));
      fgClassLayout* layout = fgLayout_GetChild(self, index);
      fgClassLayout_LoadAttributesXML(layout, node, flags, &self->base);
      fgClassLayout_LoadLayoutXML(layout, node, self);
    }
  }
}

bool fgLayout_LoadStreamXML(fgLayout* self, std::istream& s, const char* path)
{
  XMLFile xml(s);
  const XMLNode* root = xml.GetNode("fg:Layout");
  if(!root)
  {
    fgLog(FGLOG_ERROR, "No root fg:Layout node found.");
    return false;
  }

  fgSkinBase_SetPath(&self->base, path);
  fgLayout_LoadLayoutNode(self, 0, root, s);
  return true;
}
char fgLayout_LoadFileXML(fgLayout* self, const char* file)
{
  Str path(file);
  fgTrimFileFromPath(path.UnsafeString());
  std::ifstream fs(file, std::ios_base::in|std::ios_base::binary);
  return fgLayout_LoadStreamXML(self, fs, path.c_str());
}

char fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length, const char* path)
{
  std::stringstream ss(std::string(data, length));
  return fgLayout_LoadStreamXML(self, ss, path);
}

void fgLayout_SaveElementXML(fgLayout* self, fgClassLayout& e, XMLNode* parent, const char* path)
{
  XMLNode* node = parent->AddNode(e.element.type);
  fgSkinBase_WriteElementAttributesXML(node, e.element, &self->base);
  if(e.id)
    node->AddAttribute("id")->String = e.id;
  if(e.name)
    node->AddAttribute("name")->String = e.name;
  if(e.userid)
    fgStyle_WriteInt(node->AddAttribute("userid")->String, e.userid);
  for(size_t i = 0; i < e.userdata.l; ++i)
    node->AddAttribute(e.userdata.p[i].key)->String = e.userdata.p[i].value;

  for(size_t i = 0; i < e.children.l; ++i)
    fgLayout_SaveElementXML(self, e.children.p[i], node, path);
}
void fgLayout_SaveSubLayoutXML(fgLayout* self, XMLNode* node, const char* name, const char* path)
{
  if(name)
    node->AddAttribute("name")->String = name;
  if(self->skin)
    node->AddAttribute("skin")->String = self->skin->base.name;

  fgSkinBase_WriteStyleAttributesXML(node, self->base.style, &self->base, self->base.type);

  // Write all children nodes
  for(size_t i = 0; i < self->children.l; ++i)
    fgLayout_SaveElementXML(self, self->children.p[i], node, path);

  // Write all sublayouts
  if(self->sublayouts)
  {
    for(khiter_t i = 0; i < kh_end(self->sublayouts); ++i)
    {
      if(kh_exist(self->sublayouts, i))
        fgLayout_SaveSubLayoutXML(kh_val(self->sublayouts, i), node->AddNode("Layout"), kh_key(self->sublayouts, i), path);
    }
  }
}

void fgLayout_SaveStreamXML(fgLayout* self, std::ostream& s, const char* path)
{
  XMLFile xml;
  xml.SetName("xml");
  xml.AddAttribute("version")->String = "1.0";
  xml.AddAttribute("encoding")->String = "UTF-8";
  
  // Write root element
  XMLNode* root = xml.AddNode("fg:Layout");
  root->AddAttribute("xmlns:xsi")->String = "http://www.w3.org/2001/XMLSchema-instance";
  root->AddAttribute("xmlns:fg")->String = "featherGUI";
  root->AddAttribute("xsi:schemaLocation")->String = "featherGUI feather.xsd";
  fgLayout_SaveSubLayoutXML(self, root, 0, path);
  fgSkinBase_WriteXML(root, &self->base, 0);
  xml.Write(s);
}
void fgLayout_SaveFileXML(fgLayout* self, const char* file)
{
  Str path(file);
  fgTrimFileFromPath(path.UnsafeString());
  std::ofstream fs(file, std::ios_base::out | std::ios_base::binary);
  return fgLayout_SaveStreamXML(self, fs, path.c_str());
}
void fgLayout_IterateLayouts(fgLayout* self, void* p, void(*f)(void*, fgLayout*, const char*))
{
  if(!self->sublayouts)
    return;
  for(khiter_t i = 0; i < kh_end(self->sublayouts); ++i)
  {
    if(kh_exist(self->sublayouts, i))
      f(p, kh_val(self->sublayouts, i), kh_key(self->sublayouts, i));
  }
}

void fgLayout_InitCopy(fgLayout* self, const fgLayout* from, fgSkinBase* parent)
{
  fgSkinBase_InitCopy(&self->base, &from->base);
  fgSkinBase_ResolveCopy(&self->base, parent);
  self->skin = !from->skin ? 0 : self->base.GetSkin(from->skin->base.name);
  self->children.s = from->children.s;
  self->children.l = from->children.l;
  self->children.p = bss::bssMalloc<fgClassLayout>(self->children.s); // do NOT use fgmalloc, this is not freed via fgfree
  for(size_t i = 0; i < from->children.l; ++i)
    fgClassLayout_InitCopy(&self->children.p[i], &from->children.p[i], &self->base);
  self->sublayouts = fgCopyHashPointer<kh_fgLayoutMap_s, const char*, kh_init_fgLayoutMap, kh_put_fgLayoutMap>(from->sublayouts, 
    [self](kh_fgLayoutMap_s* target, khint_t index, kh_fgLayoutMap_s* from, khint_t i) {
      kh_val(target, index) = fgmalloc<fgLayout>(1, __FILE__, __LINE__);
      fgLayout_InitCopy(kh_val(target, index), kh_val(from, i), &self->base);
      kh_key(target, index) = kh_val(target, index)->base.name;
  });
}

void fgClassLayout_InitCopy(fgClassLayout* self, const fgClassLayout* from, fgSkinBase* parent)
{
  fgSkinElement_InitCopy(&self->element, &from->element);
  self->name = fgCopyText(from->name, __FILE__, __LINE__);
  self->id = fgCopyText(from->id, __FILE__, __LINE__);
  self->children.s = from->children.s;
  self->children.l = from->children.l;
  self->children.p = bss::bssMalloc<fgClassLayout>(self->children.s); // do NOT use fgmalloc, this is not freed via fgfree
  for(size_t i = 0; i < from->children.l; ++i)
    fgClassLayout_InitCopy(&self->children.p[i], &from->children.p[i], parent);
  new ((fgKeyValueArray*)&self->userdata) fgKeyValueArray((fgKeyValueArray&)from->userdata);
  self->userid = from->userid;
}
FG_UINT fgLayout::AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, int order) { return fgLayout_AddChild(this, type, name, flags, transform, units, order); }
bool fgLayout::RemoveChild(FG_UINT layout) { return fgLayout_RemoveChild(this, layout) != 0; }
fgClassLayout* fgLayout::GetChild(FG_UINT layout) const { return fgLayout_GetChild(this, layout); }
fgLayout* fgLayout::AddLayout(const char* name) { return fgLayout_AddLayout(this, name); }
bool fgLayout::RemoveLayout(const char* name) { return fgLayout_RemoveLayout(this, name) != 0; }
fgLayout* fgLayout::GetLayout(const char* name) const { return fgLayout_GetLayout(this, name); }
void fgLayout::IterateLayouts(void* p, void(*f)(void*, fgLayout*, const char*)) { fgLayout_IterateLayouts(this, p, f); }

//void fgLayout::LoadFileUBJSON(const char* file) { fgLayout_LoadFileUBJSON(this, file); }
//void fgLayout::LoadUBJSON(const char* data, FG_UINT length) { fgLayout_LoadUBJSON(this, data, length); }
//void fgLayout::SaveFileUBJSON(const char* file) { fgLayout_SaveFileUBJSON(this, file); }
bool fgLayout::LoadFileXML(const char* file) { return fgLayout_LoadFileXML(this, file) != 0; }
bool fgLayout::LoadXML(const char* data, FG_UINT length, const char* path) { return fgLayout_LoadXML(this, data, length, path) != 0; }
void fgLayout::SaveFileXML(const char* file) { fgLayout_SaveFileXML(this, file); }

void fgClassLayout::AddUserString(const char* key, const char* value) { return fgClassLayout_AddUserString(this, key, value); }
FG_UINT fgClassLayout::AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units, int order) { return fgClassLayout_AddChild(this, type, name, flags, transform, units, order); }
bool fgClassLayout::RemoveChild(FG_UINT child) { return fgClassLayout_RemoveChild(this, child) != 0; }
fgClassLayout* fgClassLayout::GetChild(FG_UINT child) const { return fgClassLayout_GetChild(this, child); }

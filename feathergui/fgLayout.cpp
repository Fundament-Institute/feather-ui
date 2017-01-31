// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgWindow.h"
#include "fgText.h"
#include "fgResource.h"
#include "feathercpp.h"
#include "fgBox.h"
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
  return DynArrayRemove((fgFontArray&)self->children, child);
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
  fgStyle_LoadAttributesXML(&self->layout.style, cur, flags, root, path, &self->id, (fgKeyValueArray*)&self->userdata);
}

void fgClassLayout_LoadLayoutXML(fgClassLayout* self, const cXMLNode* cur, fgLayout* root, const char* path)
{
  for(size_t i = 0; i < cur->GetNodes(); ++i)
  { 
    const cXMLNode* node = cur->GetNode(i);
    fgTransform transform = { 0 };
    int type = fgStyle_NodeEvalTransform(node, transform);
    int flags = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), 0);

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
      fgSkinBase_LoadNodeXML(&self->base, xml.GetNode(i));

  const cXMLNode* root = xml.GetNode("fg:Layout");
  if(!root)
    return false;

  fgStyle_LoadAttributesXML(&self->style, root, 0, &self->base, 0, 0, 0); // This will load and apply skins to the layout base

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const cXMLNode* node = root->GetNode(i);
    fgTransform transform = { 0 };
    short type = fgStyle_NodeEvalTransform(node, transform);
    int flags = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), 0);
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

void fgLayout_SaveFileXML(fgLayout* self, const char* file)
{

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
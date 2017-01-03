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

void FG_FASTCALL fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void FG_FASTCALL fgLayout_Destroy(fgLayout* self)
{
  fgSkinBase_Destroy(&self->base);
  fgStyle_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cArraySort();
}
FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgClassLayoutArray&)self->layout).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}

void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  fgStyleLayout_Init(&self->style, type, name, flags, transform, units, order);
  memset(&self->children, 0, sizeof(fgVector));
  memset(&self->userdata, 0, sizeof(fgVector));
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  reinterpret_cast<fgKeyValueArray&>(self->userdata).~cDynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cArraySort();
}

void FG_FASTCALL fgClassLayout_AddUserData(fgClassLayout* self, const char* key, const char* value)
{
  size_t i = ((fgKeyValueArray&)(self->userdata)).AddConstruct();
  self->userdata.p[i].key = fgCopyText(key, __FILE__, __LINE__);
  self->userdata.p[i].value = fgCopyText(value, __FILE__, __LINE__);
}

FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgClassLayoutArray&)self->children).Insert(fgClassLayoutConstruct(type, name, flags, transform, units, order));
}
char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child)
{
  return self->children.p + child;
}
void FG_FASTCALL fgLayout_LoadFileUBJSON(fgLayout* self, const char* file)
{
}
void FG_FASTCALL fgLayout_LoadUBJSON(fgLayout* self, const char* data, FG_UINT length)
{

}

void FG_FASTCALL fgClassLayout_LoadAttributesXML(fgClassLayout* self, const cXMLNode* cur, int flags, fgSkinBase* root, const char* path)
{
  fgStyle_LoadAttributesXML(&self->style.style, cur, flags, root, path, &self->style.id, (fgKeyValueArray*)&self->userdata);
}

void FG_FASTCALL fgClassLayout_LoadLayoutXML(fgClassLayout* self, const cXMLNode* cur, fgLayout* root, const char* path)
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
bool FG_FASTCALL fgLayout_LoadStreamXML(fgLayout* self, std::istream& s, const char* path)
{
  cXML xml(s);
  size_t nodes = xml.GetNodes();
  for(size_t i = 0; i < nodes; ++i) // Load any skins contained in the layout first
    if(!STRICMP(xml.GetNode(i)->GetName(), "fg:skin"))
      fgSkins_LoadNodeXML(&self->base, xml.GetNode(i));

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
char FG_FASTCALL fgLayout_LoadFileXML(fgLayout* self, const char* file)
{
  cStr path(file);
  path.ReplaceChar('\\', '/');
  char* dir = strrchr(path.UnsafeString(), '/');
  if(dir) // If we find a /, we chop off the rest of the string AFTER it, so it becomes a valid directory path.
    dir[1] = 0;
  std::ifstream fs(file, std::ios_base::in);
  return fgLayout_LoadStreamXML(self, fs, path.c_str());
}

char FG_FASTCALL fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length)
{
  std::stringstream ss(std::string(data, length));
  return fgLayout_LoadStreamXML(self, ss, 0);
}

void FG_FASTCALL fgLayout_SaveFileXML(fgLayout* self, const char* file)
{

}
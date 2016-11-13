// Copyright ©2016 Black Sphere Studios
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

KHASH_INIT(fgFunctionMap, const char*, fgListener, 1, kh_str_hash_func, kh_str_hash_equal);

using namespace bss_util;

void FG_FASTCALL fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void FG_FASTCALL fgLayout_Destroy(fgLayout* self)
{
  fgSkinBase_Destroy(&self->base);
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
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cArraySort();
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

__inline struct __kh_fgFunctionMap_t* fgFunctionMap_init()
{
  return kh_init_fgFunctionMap();
}
void fgFunctionMap_destroy(struct __kh_fgFunctionMap_t* h)
{
  for(khiter_t i = 0; i != kh_end(h); ++i)
    if(kh_exist(h, i))
      free(kh_val(h, i));

  kh_destroy_fgFunctionMap(h);
}

int FG_FASTCALL fgLayout_RegisterFunction(fgListener fn, const char* name)
{
  int r;
  khint_t iter = kh_put_fgFunctionMap(fgroot_instance->functionhash, fgCopyText(name), &r);
  kh_val(fgroot_instance->functionhash, iter) = fn;
  return r;
}
void FG_FASTCALL fgLayout_ApplyFunction(fgElement* root, const char* name)
{
  const char* p = (const char*)root->GetUserdata(name);
  if(p != 0)
  {
    khint_t iter = kh_get_fgFunctionMap(fgroot_instance->functionhash, p);
    if(iter != kh_end(fgroot_instance->functionhash))
      fgElement_AddListener(root, fgroot_instance->backend.fgMessageMap(name + 2), kh_val(fgroot_instance->functionhash, iter));
  }
}

void FG_FASTCALL fgLayout_ApplyFunctions(fgElement* root)
{
  fgLayout_ApplyFunction(root, "onconstruct");
  fgLayout_ApplyFunction(root, "ondestroy");
  fgLayout_ApplyFunction(root, "onmove");
  fgLayout_ApplyFunction(root, "onsetalpha");
  fgLayout_ApplyFunction(root, "onsetarea");
  fgLayout_ApplyFunction(root, "onsettransform");
  fgLayout_ApplyFunction(root, "onsetflag");
  fgLayout_ApplyFunction(root, "onsetflags");
  fgLayout_ApplyFunction(root, "onsetmargin");
  fgLayout_ApplyFunction(root, "onsetpadding");
  fgLayout_ApplyFunction(root, "onsetparent");
  fgLayout_ApplyFunction(root, "onaddchild");
  fgLayout_ApplyFunction(root, "onremovechild");
  fgLayout_ApplyFunction(root, "onlayoutchange");
  fgLayout_ApplyFunction(root, "onlayoutfunction");
  fgLayout_ApplyFunction(root, "onlayoutload");
  fgLayout_ApplyFunction(root, "ondrag");
  fgLayout_ApplyFunction(root, "ondragging");
  fgLayout_ApplyFunction(root, "ondrop");
  fgLayout_ApplyFunction(root, "ondraw");
  fgLayout_ApplyFunction(root, "oninject");
  fgLayout_ApplyFunction(root, "onclone");
  fgLayout_ApplyFunction(root, "onsetskin");
  fgLayout_ApplyFunction(root, "ongetskin");
  fgLayout_ApplyFunction(root, "onsetstyle");
  fgLayout_ApplyFunction(root, "ongetstyle");
  fgLayout_ApplyFunction(root, "ongetclassname");
  fgLayout_ApplyFunction(root, "ongetdpi");
  fgLayout_ApplyFunction(root, "onsetdpi");
  fgLayout_ApplyFunction(root, "onsetuserdata");
  fgLayout_ApplyFunction(root, "ongetuserdata");
  fgLayout_ApplyFunction(root, "onmousedown");
  fgLayout_ApplyFunction(root, "onmousedblclick");
  fgLayout_ApplyFunction(root, "onmouseup");
  fgLayout_ApplyFunction(root, "onmouseon");
  fgLayout_ApplyFunction(root, "onmouseoff");
  fgLayout_ApplyFunction(root, "onmousemove");
  fgLayout_ApplyFunction(root, "onmousescroll");
  fgLayout_ApplyFunction(root, "onmouseleave");
  fgLayout_ApplyFunction(root, "ontouchbegin");
  fgLayout_ApplyFunction(root, "ontouchend");
  fgLayout_ApplyFunction(root, "ontouchmove");
  fgLayout_ApplyFunction(root, "onkeyup");
  fgLayout_ApplyFunction(root, "onkeydown");
  fgLayout_ApplyFunction(root, "onkeychar");
  fgLayout_ApplyFunction(root, "onjoybuttondown");
  fgLayout_ApplyFunction(root, "onjoybuttonup");
  fgLayout_ApplyFunction(root, "onjoyaxis");
  fgLayout_ApplyFunction(root, "ongotfocus");
  fgLayout_ApplyFunction(root, "onlostfocus");
  fgLayout_ApplyFunction(root, "onsetname");
  fgLayout_ApplyFunction(root, "ongetname");
  fgLayout_ApplyFunction(root, "onneutral");
  fgLayout_ApplyFunction(root, "onhover");
  fgLayout_ApplyFunction(root, "onactive");
  fgLayout_ApplyFunction(root, "onaction");
  fgLayout_ApplyFunction(root, "onsetdim");
  fgLayout_ApplyFunction(root, "ongetdim");
  fgLayout_ApplyFunction(root, "ongetitem");
  fgLayout_ApplyFunction(root, "onadditem");
  fgLayout_ApplyFunction(root, "onremoveitem");
  fgLayout_ApplyFunction(root, "ongetselecteditem");
  fgLayout_ApplyFunction(root, "ongetvalue");
  fgLayout_ApplyFunction(root, "onsetvalue");
  fgLayout_ApplyFunction(root, "onsetresource");
  fgLayout_ApplyFunction(root, "onsetuv");
  fgLayout_ApplyFunction(root, "onsetcolor");
  fgLayout_ApplyFunction(root, "onsetoutline");
  fgLayout_ApplyFunction(root, "onsetfont");
  fgLayout_ApplyFunction(root, "onsetlineheight");
  fgLayout_ApplyFunction(root, "onsetletterspacing");
  fgLayout_ApplyFunction(root, "onsettext");
  fgLayout_ApplyFunction(root, "ongetresource");
  fgLayout_ApplyFunction(root, "ongetuv");
  fgLayout_ApplyFunction(root, "ongetcolor");
  fgLayout_ApplyFunction(root, "ongetoutline");
  fgLayout_ApplyFunction(root, "ongetfont");
  fgLayout_ApplyFunction(root, "ongetlineheight");
  fgLayout_ApplyFunction(root, "ongetletterspacing");
  fgLayout_ApplyFunction(root, "ongettext");

  fgElement* cur = root->root;
  while(cur)
  {
    fgLayout_ApplyFunctions(cur);
    cur = cur->next;
  }
}
void FG_FASTCALL fgLayout_LoadFileUBJSON(fgLayout* self, const char* file)
{
}
void FG_FASTCALL fgLayout_LoadUBJSON(fgLayout* self, const char* data, FG_UINT length)
{

}

void FG_FASTCALL fgClassLayout_LoadAttributesXML(fgStyleLayout* self, const cXMLNode* cur, int flags, fgSkinBase* root, const char* path)
{
  fgStyle_LoadAttributesXML(&self->style, cur, flags, root, path, &self->id);
}

void FG_FASTCALL fgClassLayout_LoadLayoutXML(fgClassLayout* self, const cXMLNode* cur, fgLayout* root, const char* path)
{
  for(size_t i = 0; i < cur->GetNodes(); ++i)
  { 
    const cXMLNode* node = cur->GetNode(i);
    fgTransform transform = { 0 };
    int type = fgStyle_NodeEvalTransform(node, transform);
    int flags = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), 0);

    if(!STRICMP(node->GetName(), "menuitem") && !node->GetNodes() && !node->GetAttributeString("text")) // An empty menuitem is a special case
      AddStyleSubMsg<FG_ADDITEM>(&self->style.style, FGADDITEM_TEXT);
    else
    {
      FG_UINT index = fgClassLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, node->GetAttributeInt("order"));
      fgClassLayout_LoadAttributesXML(&fgClassLayout_GetChild(self, index)->style, node, flags, &root->base, path);
      fgClassLayout_LoadLayoutXML(fgClassLayout_GetChild(self, index), node, root, path);
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

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const cXMLNode* node = root->GetNode(i);
    fgTransform transform = { 0 };
    short type = fgStyle_NodeEvalTransform(node, transform);
    int flags = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), 0);
    FG_UINT index = fgLayout_AddLayout(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, node->GetAttributeInt("order"));
    fgClassLayout_LoadAttributesXML(&fgLayout_GetLayout(self, index)->style, node, flags, &self->base, path);
    fgClassLayout_LoadLayoutXML(fgLayout_GetLayout(self, index), node, self, path);
  }

  return true;
}
bool FG_FASTCALL fgLayout_LoadFileXML(fgLayout* self, const char* file)
{
  cStr path(file);
  path.ReplaceChar('\\', '/');
  char* dir = strrchr(path.UnsafeString(), '/');
  if(dir) // If we find a /, we chop off the rest of the string AFTER it, so it becomes a valid directory path.
    dir[1] = 0;
  return fgLayout_LoadStreamXML(self, std::ifstream(file, std::ios_base::in), path.c_str());
}

bool FG_FASTCALL fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length)
{
  return fgLayout_LoadStreamXML(self, std::stringstream(std::string(data, length)), 0);
}

void FG_FASTCALL fgLayout_SaveFileXML(fgLayout* self, const char* file)
{

}
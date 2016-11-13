// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "fgSkin.h"
#include "fgResource.h"
#include "fgText.h"
#include "feathercpp.h"
#include "bss-util/cXML.h"
#include "bss-util/cTrie.h"
#include "fgButton.h"
#include "fgTextbox.h"
#include "fgCombobox.h"
#include "fgCurve.h"
#include "fgWindow.h"
#include "fgList.h"
#include <fstream>
#include <sstream>

KHASH_INIT(fgSkins, const char*, fgSkin*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgStyleInt, FG_UINT, fgStyle, 1, kh_int_hash_func, kh_int_hash_equal);

static_assert(sizeof(fgStyleLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgStyleArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgClassLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");

using namespace bss_util;

void FG_FASTCALL fgSkin_Init(fgSkin* self)
{
  memset(self, 0, sizeof(fgSkin));
}

template<class HASH, class T, void (FG_FASTCALL *DESTROY)(T*), void(*DEL)(HASH*, khint_t)>
char FG_FASTCALL DestroyHashElement(HASH* self, khiter_t iter)
{
  if(kh_exist(self, iter))
  {
    (*DESTROY)(kh_val(self, iter));
    free(kh_val(self, iter));
    free((char*)kh_key(self, iter));
    (*DEL)(self, iter);
    return 1;
  }
  return 0;
}

template<class HASH, class T, void(FG_FASTCALL *DESTROYELEM)(T*), void(*DEL)(HASH*, khint_t), void(*DESTROYHASH)(HASH*)>
void FG_FASTCALL DestroyHash(HASH* self)
{
  if(self)
  {
    khiter_t cur = kh_begin(self);
    while(cur != kh_end(self)) DestroyHashElement<HASH, T, DESTROYELEM, DEL>(self, cur++);
    (*DESTROYHASH)(self);
  }
}
void FG_FASTCALL fgSkinBase_Destroy(fgSkinBase* self)
{
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  DestroyHash<kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins, &kh_destroy_fgSkins>(self->skinmap);
}

void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  fgStyle_Destroy(&self->style);
  reinterpret_cast<fgStyleLayoutArray&>(self->children).~cArraySort();

  if(self->styles != 0)
  {
    khiter_t cur = kh_begin(self->styles);
    while(cur != kh_end(self->styles))
    {
      if(kh_exist(self->styles, cur))
        fgStyle_Destroy(self->styles->vals + cur);
      ++cur;
    }
    kh_destroy_fgStyleInt(self->styles);
  }

  fgSkinBase_Destroy(&self->base);
}
size_t FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  return ((fgStyleLayoutArray&)self->children).Insert(fgStyleLayoutConstruct(type, name, flags, transform, units, order));
}
char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child)
{
  return self->children.p + child;
}

FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self, const char* name)
{
  if(!self->styles)
    self->styles = kh_init_fgStyleInt();

  size_t len = strlen(name) + 1;
  DYNARRAY(char, tokenize, len);
  MEMCPY(tokenize, len, name, len);
  char* context;
  char* token = STRTOK(tokenize, "+", &context);
  FG_UINT style = 0;
  while(token)
  {
    style |= fgStyle_GetName(token, style != 0); // If this is the first token we're parsing, it's not a flag, otherwise it is a flag.
    token = STRTOK(0, "+", &context);
  }

  int r = 0;
  khiter_t i = kh_put_fgStyleInt(self->styles, style, &r);

  if(r != 0)
    kh_val(self->styles, i).styles = 0;
  return style;
}
char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style)
{
  khiter_t i = kh_get_fgStyleInt(self->styles, style);
  if(i < kh_end(self->styles) && kh_exist(self->styles, i))
  {
    fgStyle_Destroy(self->styles->vals + i);
    kh_del_fgStyleInt(self->styles, i);
    return 1;
  }
  return 0;
}
fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style)
{
  if(!self->styles) return 0;
  khiter_t i = kh_get_fgStyleInt(self->styles, style);
  return (i < kh_end(self->styles) && kh_exist(self->styles, i)) ? (self->styles->vals + i) : 0;
}

fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name)
{
  if(!self || !name)
    return 0;
  fgSkin* r = fgSkinBase_GetSkin(&self->base, name);
  return !r ? fgSkin_GetSkin(self->inherit, name) : r;
}

size_t FG_FASTCALL fgSkinBase_AddResource(fgSkinBase* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgSkinBase_RemoveResource(fgSkinBase* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgSkinBase_GetResource(const fgSkinBase* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
size_t FG_FASTCALL fgSkinBase_AddFont(fgSkinBase* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgSkinBase_RemoveFont(fgSkinBase* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgSkinBase_GetFont(const fgSkinBase* self, FG_UINT font)
{
  return DynGet<fgFontArray>(self->fonts, font).ref;
}
fgSkin* FG_FASTCALL fgSkinBase_AddSkin(fgSkinBase* self, const char* name)
{
  if(!self->skinmap)
    self->skinmap = kh_init_fgSkins();

  if(!name) return 0;
  int r = 0;
  khiter_t iter = kh_put(fgSkins, self->skinmap, name, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    kh_val(self->skinmap, iter) = bss_util::bssmalloc<fgSkin>(1);
    fgSkin_Init(kh_val(self->skinmap, iter));
    kh_key(self->skinmap, iter) = fgCopyText(name);
  }

  return kh_val(self->skinmap, iter);
}
char FG_FASTCALL fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return DestroyHashElement<struct __kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins>(self->skinmap, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* FG_FASTCALL fgSkinBase_GetSkin(const fgSkinBase* self, const char* name)
{
  if(!self || !name || !self->skinmap)
    return 0;
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? kh_val(self->skinmap, iter) : 0;
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order)
{
  self->type = fgCopyText(type);
  self->name = fgCopyText(name);
  self->id = 0;
  self->transform = *transform;
  self->units = units;
  self->flags = flags;
  self->order = order;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  if(self->type) free(self->type);
  if(self->name) free(self->name);
  if(self->id) free(self->id);
  fgStyle_Destroy(&self->style);
}

int FG_FASTCALL fgStyle_LoadUnit(const char* str, size_t len)
{
  int flags = FGUNIT_DP;
  if(!len) len = strlen(str);
  if(len > 2)
  {
    if(str[len - 2] == 'e' && str[len - 1] == 'm')
      flags = FGUNIT_EM;
    else if(str[len - 2] == 's' && str[len - 1] == 'p')
      flags = FGUNIT_SP;
    else if(str[len - 2] == 'p' && str[len - 1] == 'x')
      flags = FGUNIT_PX;
  }

  return flags;
}

int FG_FASTCALL fgStyle_LoadCoord(const char* attribute, size_t len, Coord& coord)
{
  coord = Coord { 0, 0 };
  if(!attribute) return 0;
  if(!len) len = strlen(attribute);
  const char* rel = strchr(attribute, ':');
  size_t first = len;
  if(rel)
  {
    coord.rel = (FREL)atof(rel + 1);
    if(attribute[len - 1] == '%') // Check if this is a percentage and scale it accordingly.
      coord.rel *= 0.01f;
    first = attribute - rel;
  }
  else if(attribute[len - 1] == '%')
  {
    coord.rel = (FREL)(atof(attribute) * 0.01);
    return 0;
  }

  coord.abs = (FABS)atof(attribute);
  return fgStyle_LoadUnit(attribute, first);
}
int FG_FASTCALL fgStyle_LoadAbsRect(const char* attribute, AbsRect& r)
{
  size_t len = strlen(attribute) + 1;
  DYNARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  const char* s2 = STRTOK(0, ", ", &context);
  const char* s3 = STRTOK(0, ", ", &context);

  r.left = (FABS)atof(s0);
  r.top = (FABS)atof(s1);
  r.right = (FABS)atof(s2);
  r.bottom = (FABS)atof(s3);
  return (fgStyle_LoadUnit(s0, 0) << FGUNIT_LEFT) |
    (fgStyle_LoadUnit(s1, 0) << FGUNIT_TOP) |
    (fgStyle_LoadUnit(s2, 0) << FGUNIT_RIGHT) |
    (fgStyle_LoadUnit(s3, 0) << FGUNIT_BOTTOM);
}

int FG_FASTCALL fgStyle_LoadCRect(const char* attribute, CRect& r)
{
  size_t len = strlen(attribute) + 1;
  DYNARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  const char* s2 = STRTOK(0, ", ", &context);
  const char* s3 = STRTOK(0, ", ", &context);

  return (fgStyle_LoadCoord(attribute, 0, r.left) << FGUNIT_LEFT) |
    (fgStyle_LoadCoord(s1, 0, r.top) << FGUNIT_TOP) |
    (fgStyle_LoadCoord(s2, 0, r.right) << FGUNIT_RIGHT) |
    (fgStyle_LoadCoord(s3, 0, r.bottom) << FGUNIT_BOTTOM);
}

int FG_FASTCALL fgStyle_LoadCVec(const char* attribute, CVec& v)
{
  size_t len = strlen(attribute) + 1;
  DYNARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  return (fgStyle_LoadCoord(s0, 0, v.x) << FGUNIT_X) | (fgStyle_LoadCoord(s1, 0, v.y) << FGUNIT_Y);
}

int FG_FASTCALL fgStyle_NodeEvalTransform(const cXMLNode* node, fgTransform& t)
{
  static cTrie<uint16_t, true> attr(9, "area", "center", "rotation", "left", "top", "right", "bottom", "width", "height");
  int flags = 0;
  bool hastransform = false;

  for(size_t i = 0; i < node->GetAttributes(); ++i)
  {
    const cXMLValue* a = node->GetAttribute(i);
    switch(attr[a->Name.c_str()])
    {
    case 0:
      flags &= (~(FGUNIT_LEFT_MASK | FGUNIT_TOP_MASK | FGUNIT_RIGHT_MASK | FGUNIT_BOTTOM_MASK));
      flags |= fgStyle_LoadCRect(a->String, t.area);
      break;
    case 1:
      flags &= (~(FGUNIT_X_MASK | FGUNIT_Y_MASK));
      flags |= fgStyle_LoadCVec(a->String, t.center);
      break;
    case 2:
      t.rotation = (FABS)a->Float;
      break;
    case 3:
      flags = (flags&(FGUNIT_LEFT_MASK)) | fgStyle_LoadCoord(a->String, 0, t.area.left);
      break;
    case 4:
      flags = (flags&(FGUNIT_TOP_MASK)) | fgStyle_LoadCoord(a->String, 0, t.area.top);
      break;
    case 5:
      flags = (flags&(FGUNIT_RIGHT_MASK)) | fgStyle_LoadCoord(a->String, 0, t.area.right);
      break;
    case 6:
      flags = (flags&(FGUNIT_BOTTOM_MASK)) | fgStyle_LoadCoord(a->String, 0, t.area.bottom);
      break;
    case 7:
      flags = (flags&(FGUNIT_RIGHT_MASK)) | fgStyle_LoadUnit(a->String, 0) | FGUNIT_RIGHT_WIDTH;
      t.area.right.abs = (FABS)a->Float;
      break;
    case 8:
      flags = (flags&(FGUNIT_BOTTOM_MASK)) | fgStyle_LoadUnit(a->String, 0) | FGUNIT_BOTTOM_HEIGHT;
      t.area.bottom.abs = (FABS)a->Float;
      break;
    default:
      continue;
    }
    hastransform = true;
  }

  if(!hastransform) return -1;
  return flags;
}

void FG_FASTCALL fgStyle_LoadAttributesXML(fgStyle* self, const cXMLNode* cur, int flags, fgSkinBase* root, const char* path, char** id)
{
  static cTrie<uint16_t, true> t(35, "id", "min-width", "min-height", "max-width", "max-height", "skin", "alpha", "margin", "padding", "text",
    "placeholder", "color", "placecolor", "cursorcolor", "selectcolor", "hovercolor", "dragcolor", "edgecolor", "font", "lineheight",
    "letterspacing", "value", "uv", "resource", "outline", "area", "center", "rotation", "left", "top", "right", "bottom", "width", "height", "range");
  static cTrie<uint16_t, true> tvalue(5, "checkbox", "curve", "progressbar", "radiobutton", "slider");
  static cTrie<uint16_t, true> tenum(5, "true", "false", "none", "checked", "indeterminate");

  int minflags = 0x8000000;
  AbsVec mindim = { -1, -1 };
  int maxflags = 0x8000000;
  AbsVec maxdim = { -1, -1 };

  for(size_t i = 0; i < cur->GetAttributes(); ++i)
  {
    const cXMLValue* attr = cur->GetAttribute(i);
    const uint16_t ID = t[attr->Name.c_str()];

    switch(ID)
    {
    case 0:
      if(id)
        *id = fgCopyText(attr->String);
      break;
    case 1:
      mindim.x = (FABS)attr->Float;
      minflags &= (~0x8000000);
      minflags |= (fgStyle_LoadUnit(attr->String, attr->String.length() + 1) << FGUNIT_X);
      break;
    case 2:
      mindim.y = (FABS)attr->Float;
      minflags &= (~0x8000000);
      minflags |= (fgStyle_LoadUnit(attr->String, attr->String.length() + 1) << FGUNIT_Y);
      break;
    case 3:
      maxdim.x = (FABS)attr->Float;
      maxflags &= (~0x8000000);
      maxflags |= (fgStyle_LoadUnit(attr->String, attr->String.length() + 1) << FGUNIT_X);
      break;
    case 4:
      maxdim.y = (FABS)attr->Float;
      maxflags &= (~0x8000000);
      maxflags |= (fgStyle_LoadUnit(attr->String, attr->String.length() + 1) << FGUNIT_Y);
      break;
    case 5: // skin
    {
      fgSkin* skin = fgSkinBase_GetSkin(root, attr->String);
      if(!skin) // Attempt loading it as an absolute XML path
        skin = fgSkins_LoadFileXML(root, attr->String);
      if(!skin) // relative XML path
        skin = fgSkins_LoadFileXML(root, cStr(path) + attr->String);
      if(!skin) // absolute UBJSON path
        skin = fgSkins_LoadFileUBJSON(root, attr->String);
      if(!skin) // relative UBJSON path
        skin = fgSkins_LoadFileUBJSON(root, cStr(path) + attr->String);
      if(skin)
        AddStyleMsg<FG_SETSKIN, void*>(self, skin);
    }
    break;
    case 6:
      AddStyleMsg<FG_SETALPHA, FABS>(self, (FABS)attr->Float);
      break;
    case 7:
    {
      AbsRect r;
      int f = fgStyle_LoadAbsRect(attr->String, r);
      AddStyleSubMsgArg<FG_SETMARGIN, AbsRect>(self, f, &r);
      break;
    }
    case 8:
    {
      AbsRect r;
      int f = fgStyle_LoadAbsRect(attr->String, r);
      AddStyleSubMsgArg<FG_SETPADDING, AbsRect>(self, f, &r);
      break;
    }
    case 9: // text
    case 10: // placeholder
    {
      FG_Msg msg = { 0 };
      msg.type = FG_SETTEXT;
      msg.subtype = ((ID == 9) ? FGSETTEXT_UTF8 : FGSETTEXT_PLACEHOLDER_UTF8);
      fgStyle_AddStyleMsg(self, &msg, attr->String.c_str(), attr->String.length() + 1, 0, 0);
      break;
    }
    case 11:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_MAIN, attr->Integer);
      break;
    case 12:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_PLACEHOLDER, attr->Integer);
      break;
    case 13:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_CURSOR, attr->Integer);
      break;
    case 14:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_SELECT, attr->Integer);
      break;
    case 15:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_HOVER, attr->Integer);
      break;
    case 16:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_DRAG, attr->Integer);
      break;
    case 17:
      AddStyleSubMsg<FG_SETCOLOR, ptrdiff_t>(self, FGSETCOLOR_EDGE, attr->Integer);
      break;
    case 18: // font
    {
      const char* split = strchr(attr->String.c_str(), ';');
      FG_UINT index = fgSkinBase_AddFont(root, fgroot_instance->backend.fgCreateFont(flags, cStr(attr->String.c_str(), split - attr->String.c_str()).c_str(), atoi(split + 1), 96));
      if(split != 0)
        AddStyleMsg<FG_SETFONT, void*>(self, fgSkinBase_GetFont(root, index));
      break;
    }
    case 19:
      AddStyleMsg<FG_SETLINEHEIGHT, FABS>(self, (FABS)attr->Float);
      break;
    case 20:
      AddStyleMsg<FG_SETLETTERSPACING, FABS>(self, (FABS)attr->Float);
      break;
    case 21: // Value's type changes depending on what type we are
      switch(tvalue[cur->GetName()])
      {
      case 0: // Checkbox
        switch(tenum[attr->String])
        {
        case 0:
        case 3: // Both true and checked map to checked
          AddStyleSubMsg<FG_SETVALUE, ptrdiff_t>(self, FGVALUE_INT64, FGCHECKED_CHECKED);
          break;
        case 4:
          AddStyleSubMsg<FG_SETVALUE, ptrdiff_t>(self, FGVALUE_INT64, FGCHECKED_INDETERMINATE);
          break;
        default: // Everything else maps to not checked
          AddStyleSubMsg<FG_SETVALUE, ptrdiff_t>(self, FGVALUE_INT64, FGCHECKED_NONE);
          break;
        }
      case 1: // Curve
        break; // Not implemented
      case 2: // Progressbar
      case 4: // Slider
        AddStyleSubMsg<FG_SETVALUE, FABS>(self, FGVALUE_FLOAT, (FABS)attr->Float);
        break;
      case 3: // Radiobutton
        switch(tenum[attr->String])
        {
        case 0:
        case 3: // true, checked, and indeterminate all map to true
        case 4:
          AddStyleSubMsg<FG_SETVALUE, ptrdiff_t>(self, FGVALUE_INT64, 1);
          break;
        default: // Everything else maps to not checked
          AddStyleSubMsg<FG_SETVALUE, ptrdiff_t>(self, FGVALUE_INT64, 0);
          break;
        }
      }
      break;
    case 22: // uv
    {
      CRect uv;
      int f = fgStyle_LoadCRect(attr->String, uv);
      AddStyleSubMsgArg<FG_SETUV, CRect>(self, f, &uv);
      break;
    }
    case 23: // resource
    {
      FG_UINT res = fgSkinBase_AddResource(root, fgCreateResourceFile(flags, attr->String));
      AddStyleMsg<FG_SETRESOURCE, void*>(self, fgSkinBase_GetResource(root, res));
      break;
    }
    case 24: // outline
      AddStyleSubMsg<FG_SETOUTLINE, FABS>(self, fgStyle_LoadUnit(attr->String.c_str(), attr->String.length() + 1), (FABS)attr->Float);
      break;
    case 25: // area
    case 26: // center
    case 27: // rotation
    case 28: // left
    case 29: // top
    case 30: // right
    case 31: // bottom
    case 32: // width
    case 33: // height
      break; // These are processed before we get here, so ignore them.
    case 34: // range
      AddStyleSubMsg<FG_SETVALUE, ptrdiff_t, size_t>(self, FGVALUE_INT64, attr->Integer, 1);
      break;
    default: // Otherwise, unrecognized attributes are set as custom userdata
    {
      FG_Msg msg = { 0 };
      msg.type = FG_SETUSERDATA;
      fgStyle_AddStyleMsg(self, &msg, attr->String.c_str(), attr->String.length() + 1, attr->Name.c_str(), attr->Name.length() + 1);
    }
    break;
    }
  }

  if(!(minflags & 0x8000000))
    AddStyleSubMsg<FG_SETDIM, FABS, FABS>(self, minflags | FGDIM_MIN, mindim.x, mindim.y);
  if(!(maxflags & 0x8000000))
    AddStyleSubMsg<FG_SETDIM, FABS, FABS>(self, maxflags | FGDIM_MAX, maxdim.x, maxdim.y);
}


fgFlag fgSkinBase_GetFlagsFromString(const char* s, fgFlag* remove)
{
  static cTrie<uint16_t, true> t(50, "BACKGROUND", "NOCLIP", "IGNORE", "HIDDEN", "EXPANDX", "EXPANDY", "DISABLE", "SNAPX", "SNAPY", "HIDEH",
    "HIDEV", "SHOWH", "SHOWV", "TILEX", "TILEY", "DISTRIBUTEX", "DISTRIBUTEY", "FIXEDSIZE", "SINGLESELECT", "MULTISELECT",
    "DRAGGABLE", "HIDEH", "HIDEV", "SHOWH", "SHOWV", "CHARWRAP", "WORDWRAP", "ELLIPSES", "RTL", "RIGHTALIGN", "CENTER", "SUBPIXEL", "ACTION",
    "SINGLELINE", "NOFOCUS", "ROUNDRECT", "CIRCLE", "LINE", "QUADRATIC", "CUBIC", "BSPLINE", "MINIMIZABLE", "MAXIMIZABLE", "RESIZABLE",
    "NOTITLEBAR", "NOBORDER", "EXPAND", "SNAP", "TILE", "DISTRIBUTE");

  fgFlag out = 0;
  const char* n;
  while(s)
  {
    n = strchr(s, '|');
    bool del = s[0] == '-';
    if(del)
      ++s;
    uint16_t i = !n ? t[s] : t.Get(s, n - s);
    fgFlag f = 0;
    if(i < 50)
    {
      switch(i)
      {
      default:
        f = (1 << i);
        break;
      case 19: f = FGLIST_MULTISELECT; break; // We can't rely on the default method here because MULTISELECT is actually two seperate flags
      case 20: f = FGLIST_DRAGGABLE; break;
      case 21: f = FGSCROLLBAR_HIDEH; break;
      case 22: f = FGSCROLLBAR_HIDEV; break;
      case 23: f = FGSCROLLBAR_SHOWH; break;
      case 24: f = FGSCROLLBAR_SHOWV; break;
      case 25: f = FGTEXT_CHARWRAP; break;
      case 26: f = FGTEXT_WORDWRAP; break;
      case 27: f = FGTEXT_ELLIPSES; break;
      case 28: f = FGTEXT_RTL; break;
      case 29: f = FGTEXT_RIGHTALIGN;  break;
      case 30: f = FGTEXT_CENTER;  break;
      case 31: f = FGTEXT_SUBPIXEL;  break;
      case 32: f = FGTEXTBOX_ACTION;  break;
      case 33: f = FGTEXTBOX_SINGLELINE;  break;
      case 34: f = FGBUTTON_NOFOCUS;  break;
      case 35: f = FGRESOURCE_ROUNDRECT;  break;
      case 36: f = FGRESOURCE_CIRCLE;  break;
      case 37: f = FGCURVE_LINE;  break;
      case 38: f = FGCURVE_QUADRATIC;  break;
      case 39: f = FGCURVE_CUBIC;  break;
      case 40: f = FGCURVE_BSPLINE;  break;
      case 41: f = FGWINDOW_MINIMIZABLE;  break;
      case 42: f = FGWINDOW_MAXIMIZABLE;  break;
      case 43: f = FGWINDOW_RESIZABLE;  break;
      case 44: f = FGWINDOW_NOTITLEBAR;  break;
      case 45: f = FGWINDOW_NOBORDER;  break;
      case 46: f = FGELEMENT_EXPAND;  break;
      case 47: f = FGELEMENT_SNAP;  break;
      case 48: f = FGBOX_TILE;  break;
      case 49: f = FGBOX_DISTRIBUTEX | FGBOX_DISTRIBUTEY;  break;
      }
    }

    if(del && remove != 0)
      *remove |= f;
    else
      out |= f;
    s = !n? 0 : n + 1;
  }
  
  return out;
}

void FG_FASTCALL fgSkins_LoadSubNodeXML(fgSkin* self, const cXMLNode* cur)
{
  int rootflags = fgSkinBase_GetFlagsFromString(cur->GetAttributeString("flags"), 0);
  fgStyle_LoadAttributesXML(&self->style, cur, rootflags, &self->base, 0, 0);
  //self->inherit = 

  for(size_t i = 0; i < cur->GetNodes(); ++i)
  {
    const cXMLNode* node = cur->GetNode(i);
    if(!STRICMP(node->GetName(), "skin"))
      fgSkins_LoadNodeXML(&self->base, node);
    else if(!STRICMP(node->GetName(), "style"))
    {
      FG_UINT style = fgSkin_AddStyle(self, node->GetAttributeString("name"));
      // Styles parse flags differently - the attributes use the parent flags for resource/font loading, then creates add and remove flag messages
      fgStyle_LoadAttributesXML(fgSkin_GetStyle(self, style), node, rootflags, &self->base, 0, 0);
      fgFlag remove = 0;
      fgFlag add = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), &remove);
      if(remove)
        AddStyleMsg<FG_SETFLAG, ptrdiff_t, size_t>(fgSkin_GetStyle(self, style), remove, 0);
      if(add)
        AddStyleMsg<FG_SETFLAG, ptrdiff_t, size_t>(fgSkin_GetStyle(self, style), add, 1);
    }
    else
    {
      fgTransform transform = { 0 };
      int type = fgStyle_NodeEvalTransform(node, transform);
      fgFlag flags = fgSkinBase_GetFlagsFromString(node->GetAttributeString("flags"), 0);
      FG_UINT index = fgSkin_AddChild(self, node->GetName(), node->GetAttributeString("name"), flags, &transform, type, node->GetAttributeInt("order"));
      fgStyle_LoadAttributesXML(&fgSkin_GetChild(self, index)->style, node, flags, &self->base, 0, 0);
    }
  }
}

fgSkin* FG_FASTCALL fgSkins_LoadNodeXML(fgSkinBase* self, const cXMLNode* root)
{
  const char* id = root->GetAttributeString("id");
  if(!id)
    id = root->GetAttributeString("name");
  if(!id)
     return 0;
  fgSkin* s = fgSkinBase_AddSkin(self, id);
  fgSkins_LoadSubNodeXML(s, root);
  return s;
}

fgSkin* FG_FASTCALL fgSkins_LoadStreamXML(fgSkinBase* self, std::istream& s)
{
  cXML xml(s);
  size_t index = 0;
  const cXMLNode* root;
  fgSkin* ret = 0;

  while(root = xml.GetNode(index++))
  {
    if(!STRICMP(root->GetName(), "fg:skin"))
      ret = fgSkins_LoadNodeXML(self, root);
  }

  return ret;
}

fgSkin* FG_FASTCALL fgSkins_LoadFileXML(fgSkinBase* self, const char* file)
{
  return fgSkins_LoadStreamXML(self, std::ifstream(file, std::ios_base::in));
}
fgSkin* FG_FASTCALL fgSkins_LoadXML(fgSkinBase* self, const char* data, FG_UINT length)
{
  return fgSkins_LoadStreamXML(self, std::stringstream(std::string(data, length)));
}

fgSkin* FG_FASTCALL fgSkins_LoadFileUBJSON(fgSkinBase* self, const char* file)
{
  return 0;
}
fgSkin* FG_FASTCALL fgSkins_LoadUBJSON(fgSkinBase* self, const void* data, FG_UINT length)
{
  return 0;
}
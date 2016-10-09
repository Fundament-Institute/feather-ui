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
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cArraySort();
}
FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font)
{
  return ((fgFontArray&)self->fonts)[font].ref;
}
FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  return ((fgClassLayoutArray&)self->layout).Insert(fgClassLayoutConstruct(type, name, flags, transform, order));
}
char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}

void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  fgStyleLayout_Init(&self->style, type, name, flags, transform, order);
  memset(&self->children, 0, sizeof(fgVector));
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cArraySort();
}

FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  return ((fgClassLayoutArray&)self->children).Insert(fgClassLayoutConstruct(type, name, flags, transform, order));
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
      fgElement_AddListener(root, fgMessageMap(name + 2), kh_val(fgroot_instance->functionhash, iter));
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
  fgLayout_ApplyFunction(root, "onnuetral");
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

inline char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgElement* child, FABS dimx)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  if(area->right.abs > d)
  {
    selfarea->right.abs = selfarea->left.abs + area->right.abs;
    return FGMOVE_RESIZEX;
  }
  return 0;
}

inline char FG_FASTCALL fgLayout_ExpandY(CRect* selfarea, fgElement* child, FABS dimy)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->bottom.abs - selfarea->top.abs;
  if(area->bottom.abs > d)
  {
    selfarea->bottom.abs = selfarea->top.abs + area->bottom.abs;
    return FGMOVE_RESIZEY;
  }
  return 0;
}

size_t FG_FASTCALL fgDefaultLayout(fgElement* self, const FG_Msg* msg, CRect* area, AbsRect* parent)
{
  fgElement* child = (fgElement*)msg->other;
  assert(!child || !(child->flags & FGELEMENT_BACKGROUND));

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTREORDER:
  case FGELEMENT_LAYOUTRESIZE:
    break;
  case FGELEMENT_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGELEMENT_LAYOUTREMOVE;
    size_t dim = fgDefaultLayout(self, &m, area, parent);
    m.subtype = FGELEMENT_LAYOUTADD;
    dim |= fgDefaultLayout(self, &m, area, parent);
    return dim;
  }
  case FGELEMENT_LAYOUTADD:
  {
    char dim = 0;
    dim |= fgLayout_ExpandX(area, child, parent->right - parent->left);
    dim |= fgLayout_ExpandY(area, child, parent->bottom - parent->top);
    return dim;
  }
  case FGELEMENT_LAYOUTREMOVE:
  {
    CRect* childarea = &child->transform.area;
    FABS dx = self->transform.area.right.abs - self->transform.area.left.abs;
    FABS dy = self->transform.area.bottom.abs - self->transform.area.top.abs;
    if((childarea->right.rel == childarea->left.rel &&
      childarea->right.abs >= dx) ||
      (childarea->bottom.rel == childarea->top.rel &&
        childarea->bottom.abs >= dy))
    {
      FG_Msg m = *msg;
      m.subtype = FGELEMENT_LAYOUTRESET;
      m.other = child;
      return fgDefaultLayout(self, &m, area, parent);
    }
  }
  case FGELEMENT_LAYOUTRESET:
  {
    fgElement* hold = self->root;
    area->right.abs = area->left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
    area->bottom.abs = area->top.abs;
    char dim = 0;

    while(hold) // O(n) by necessity
    {
      if(hold != child)
      {
        dim |= fgLayout_ExpandX(area, hold, parent->right - parent->left);
        dim |= fgLayout_ExpandY(area, hold, parent->bottom - parent->top);
      }
      hold = hold->next;
    }

    return dim;
  }
  }

  return 0;
}

char fgTileLayoutSame(FABS posx, FABS posy, fgElement* cur)
{
  return (posx > 0 || posy > 0) && posx == cur->transform.area.left.abs && posy == cur->transform.area.left.abs;
}

fgElement* fgLayout_GetNext(fgElement* cur)
{
  while((cur = cur->next) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

fgElement* fgLayout_GetPrev(fgElement* cur)
{
  while((cur = cur->prev) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

AbsVec fgTileLayoutReorder(fgElement* prev, fgElement* cur, fgElement* skip, char axis, float max, float pitch, AbsVec expand) // axis: 0 means x-axis first, 1 means y-axis first
{
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  AbsVec pos = { 0,0 };
  if(prev)
  {
    assert(!(prev->flags&FGELEMENT_BACKGROUND));
    pos.x = axis ? prev->transform.area.bottom.abs : prev->transform.area.right.abs;
    pos.y = axis ? prev->transform.area.left.abs : prev->transform.area.top.abs;
  }
  char rowbump = 0; // Used to keep track if we've gone down at least one row - this is required for our same position optimization to be valid.

  while(cur) // O(n) complexity, but should break after re-ordering becomes unnecessary.
  {
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND)) // we check for cur background flags so we can pass in the root
    {
      cur = cur->next;
      continue;
    }
    if(fgTileLayoutSame(axis ? pos.y : pos.x, axis ? pos.x : pos.y, cur) && rowbump)
      break;
    AbsVec dim = { (cur->transform.area.right.abs - cur->transform.area.left.abs), (cur->transform.area.bottom.abs - cur->transform.area.top.abs) };
    if(axis) { FABS t = dim.x; dim.x = dim.y; dim.y = t; }

    if(pos.x + dim.x > max)
    {
      pos.y += pitch;
      pos.x = 0;
      pitch = 0;
      rowbump = 1;
    }

    if(!axis)
      MoveCRect(pos.x, pos.y, &cur->transform.area);
    else
      MoveCRect(pos.y, pos.x, &cur->transform.area);
    pos.x += dim.x;
    if(dim.y > pitch) pitch = dim.y;
    dim.y += pos.y;
    if(pos.x > expand.x) expand.x = pos.x;
    if(dim.y > expand.y) expand.y = dim.y;
    cur = cur->next;
  }
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  return expand;
}

FABS FG_FASTCALL fgTileLayoutGetPitch(fgElement* cur, char axis)
{
  if(!cur) return 0.0f;
  FABS start = (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs);
  FABS dim;
  FABS pitch = 0.0;
  while(cur != 0 && (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs) == start)
  {
    dim = (axis ? cur->transform.area.right.abs : cur->transform.area.bottom.abs) - start;
    if(dim > pitch)
      pitch = dim;
    cur = fgLayout_GetPrev(cur); // run backwards until we go up a level
  }
  return pitch;
}

size_t FG_FASTCALL fgTileLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, CRect* area)
{
  AbsVec curdim = AbsVec { area->right.abs - area->left.abs, area->bottom.abs - area->top.abs };
  fgElement* child = (fgElement*)msg->other;
  fgElement* prev;
  char axis = ((flags & FGBOX_TILE) && (flags & FGBOX_DISTRIBUTEY)) || ((flags & FGBOX_TILEY) && !(flags & FGBOX_TILEX));
  FABS max = INFINITY;
  if((flags & FGBOX_TILE) == FGBOX_TILE)
  {
    AbsRect out;
    ResolveRect(self, &out);
    max = !axis ? ((self->flags & FGELEMENT_EXPANDX) ? self->maxdim.x : (out.right - out.left)) : ((self->flags & FGELEMENT_EXPANDY) ? self->maxdim.y : (out.bottom - out.top));
    if(max < 0) max = INFINITY;
  }

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTRESET:
    curdim = fgTileLayoutReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
    break;
  case FGELEMENT_LAYOUTRESIZE:
    if((((msg->otheraux & FGMOVE_RESIZEX) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->otheraux & FGMOVE_RESIZEY) && !(self->flags&FGELEMENT_EXPANDY))) && (flags & FGBOX_TILE))
      curdim = fgTileLayoutReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
    else
      return 0;
    break;
  case FGELEMENT_LAYOUTREORDER:
  {
    fgElement* old = (fgElement*)msg->other2;
    fgElement* cur = old;
    assert(!(old->flags&FGELEMENT_BACKGROUND));
    assert(!(cur->flags&FGELEMENT_BACKGROUND));
    while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
    child = !cur ? child : old;

    prev = fgLayout_GetPrev(child); // walk backwards until we hit the previous non-background element
    curdim = fgTileLayoutReorder(prev, child, 0, axis, max, (flags & FGBOX_TILE) ? fgTileLayoutGetPitch(prev, axis) : 0.0f, curdim);
  }
  break;
  case FGELEMENT_LAYOUTMOVE:
    if(!(msg->otheraux&FGMOVE_RESIZE)) // Only reorder if the child was resized. Note that we need to resize even if it's the opposite axis to account for expansion
      return 0;
    if(((self->flags&FGELEMENT_EXPANDX) && !(flags&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(flags&FGBOX_TILEY)))
    { // If we are expanding along an axis that isn't tiled, we have to recalculate the whole thing.
      curdim = fgTileLayoutReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
      break;
    }

    prev = fgLayout_GetPrev(child);
    if(prev)
      curdim = AbsVec { prev->transform.area.right.abs, prev->transform.area.bottom.abs };
    else
      curdim = AbsVec { 0,0 };

    prev = fgLayout_GetPrev(child);
    curdim = fgTileLayoutReorder(prev, child, 0, axis, max, (flags & FGBOX_TILE) ? fgTileLayoutGetPitch(prev, axis) : 0.0f, curdim);
    break;
  case FGELEMENT_LAYOUTADD:
    prev = fgLayout_GetPrev(child);
    curdim = fgTileLayoutReorder(prev, child, 0, axis, max, (flags & FGBOX_TILE) ? fgTileLayoutGetPitch(prev, axis) : 0.0f, curdim);
    break;
  case FGELEMENT_LAYOUTREMOVE:
    if(((self->flags&FGELEMENT_EXPANDX) && !(flags&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(flags&FGBOX_TILEY)))
    { // If we are expanding along an axis that isn't tiled, we have to recalculate the whole thing.
      curdim = fgTileLayoutReorder(0, self->root, child, axis, max, 0.0f, AbsVec { 0,0 });
      break;
    }

    prev = fgLayout_GetPrev(child);
    if(prev)
      curdim = AbsVec { prev->transform.area.right.abs, prev->transform.area.bottom.abs };
    else
      curdim = AbsVec { 0,0 };

    curdim = fgTileLayoutReorder(prev, fgLayout_GetNext(child), 0, axis, max, (flags & FGBOX_TILE) ? fgTileLayoutGetPitch(prev, axis) : 0.0f, curdim);
    break;
  }

  char dim = 0; // We always expand the area regardless of the EXPAND flags because the function that calls this should use the information appropriately.
  if(area->right.abs - area->left.abs != curdim.x)
  {
    area->right.abs = area->left.abs + curdim.x;
    dim |= FGMOVE_RESIZEX;
  }
  if(area->bottom.abs - area->top.abs != curdim.y)
  {
    area->bottom.abs = area->top.abs + curdim.y;
    dim |= FGMOVE_RESIZEY;
  }
  return dim;
}

size_t FG_FASTCALL fgDistributeLayout(fgElement* self, const FG_Msg* msg, fgFlag axis)
{
  /*switch(msg->type)
  {
  case FGELEMENT_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGELEMENT_LAYOUTRESIZE:
  break;
  case FGELEMENT_LAYOUTREORDER:
  //child = child->order<msg->other->order?child:msg->other; // Get lowest child
  fgDistributeLayoutReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTADD:
  fgDistributeLayoutReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTREMOVE:
  fgDistributeLayoutReorder(child->prev, child->next, axis, num);
  break;
  }*/

  return 0;
}

fgFlag fgLayout_GetFlagsFromString(const char* s)
{
  if(!s) return 0;

}
void FG_FASTCALL fgLayout_LoadFileUBJSON(fgLayout* self, const char* file)
{
}
void FG_FASTCALL fgLayout_LoadUBJSON(fgLayout* self, const char* data, FG_UINT length)
{

}

void FG_FASTCALL fgLayout_NodeEvalTransform(const cXMLNode* node, fgTransform& t)
{

}
/*void* FG_FASTCALL fgClassLayout_LoadFont(fgClassLayout* self, const cXMLValue* attribute)
{

}
uint64_t FG_FASTCALL fgClassLayout_LoadHex(fgClassLayout* self, const cXMLValue* attribute)
{

}
Coord FG_FASTCALL fgClassLayout_LoadCoord(fgClassLayout* self, const cXMLValue* attribute)
{

}
AbsRect FG_FASTCALL fgClassLayout_LoadAbsRect(fgClassLayout* self, const cXMLValue* attribute)
{

}
CRect FG_FASTCALL fgClassLayout_LoadCRect(fgClassLayout* self, const cXMLValue* attribute)
{

}
CVec FG_FASTCALL fgClassLayout_LoadCVec(fgClassLayout* self, const cXMLValue* attribute)
{

}
FABS FG_FASTCALL fgClassLayout_LoadUnit(fgClassLayout* self, const cXMLValue* attribute)
{

}*/

void FG_FASTCALL fgClassLayout_LoadAttributesXML(fgClassLayout* self, const cXMLNode* cur)
{
  static cTrie<uint16_t> t(19, "id", "min-width", "min-height", "max-width", "max-height", "skin", "alpha", "margin", "padding", "text", "color", "font", "lineheight", "letterspacing", "value", "uv", "resource", "edgecolor", "outline");

  switch(t[cur->GetName()])
  {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  case 10:
  case 11:
  case 12:
  case 13:
  case 14:
  case 15:
  case 16:
  case 17:
  case 18:
    break;
  }
}
void FG_FASTCALL fgClassLayout_LoadLayoutXML(fgClassLayout* self, const cXMLNode* cur, fgLayout* root)
{
  for(size_t i = 0; i < cur->GetNodes(); ++i)
  {
    const cXMLNode* node = cur->GetNode(i);
    fgTransform transform = { 0 };
    fgLayout_NodeEvalTransform(node, transform);
    FG_UINT index = fgClassLayout_AddChild(self, node->GetName(), node->GetAttributeString("name"), fgLayout_GetFlagsFromString(node->GetAttributeString("flags")), &transform, node->GetAttributeInt("order"));
    fgClassLayout_LoadAttributesXML(fgClassLayout_GetChild(self, index), node);
    fgClassLayout_LoadLayoutXML(fgClassLayout_GetChild(self, index), node, root);
  }
}
bool FG_FASTCALL fgLayout_LoadStreamXML(fgLayout* self, std::istream& s)
{
  cXML xml(s);
  const cXMLNode* root = xml.GetNode("fg:Layout");
  if(!root)
    return false;

  for(size_t i = 0; i < root->GetNodes(); ++i)
  {
    const cXMLNode* node = root->GetNode(i);
    fgTransform transform = { 0 };
    fgLayout_NodeEvalTransform(node, transform);
    FG_UINT index = fgLayout_AddLayout(self, node->GetName(), node->GetAttributeString("name"), fgLayout_GetFlagsFromString(node->GetAttributeString("flags")), &transform, node->GetAttributeInt("order"));
    fgClassLayout_LoadAttributesXML(fgLayout_GetLayout(self, index), node);
    fgClassLayout_LoadLayoutXML(fgLayout_GetLayout(self, index), node, self);
  }

  return true;
}
void FG_FASTCALL fgLayout_LoadFileXML(fgLayout* self, const char* file)
{
  fgLayout_LoadStreamXML(self, std::ifstream(file, std::ios_base::in));
}

void FG_FASTCALL fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length)
{
  fgLayout_LoadStreamXML(self, std::stringstream(std::string(data, length)));
}

void FG_FASTCALL fgLayout_SaveFileXML(fgLayout* self, const char* file)
{

}
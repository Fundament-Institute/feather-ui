// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgWindow.h"
#include "fgText.h"
#include "fgResource.h"
#include "feathercpp.h"

fgElement* fgLayoutLoadMapping(const char* type, fgElement* parent, fgElement* next, const char* name, fgFlag flags, const fgTransform* transform)
{
  if(!strcmp(type, "fgElement")) { // This could be a hash, but this is low priority as this is never called in a performance critical loop.
    fgElement* r = (fgElement*)malloc(sizeof(fgElement));
    fgElement_Init(r, parent, next, name, flags, transform);
    return r;
  }
  else if(!strcmp(type, "fgControl")) {
    fgControl* r = (fgControl*)malloc(sizeof(fgControl));
    fgControl_Init(r, parent, next, name, flags, transform);
    return (fgElement*)r;
  }
  else if(!strcmp(type, "fgWindow"))
  {
    fgElement* r = fgWindow_Create(0, flags, transform);
    _sendmsg<FG_SETPARENT, void*, void*>(r, parent, next);
    return r;
  }
  else if(!strcmp(type, "fgButton"))
    return fgButton_Create(0, parent, next, name, flags, transform);
  else if(!strcmp(type, "fgText"))
    return fgText_Create(0, 0, 0xFFFFFFFF, parent, next, name, flags, transform);
  else if(!strcmp(type, "fgResource"))
    return fgResource_Create(0, 0, 0xFFFFFFFF, parent, next, name, flags, transform);

  return 0;
}

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

inline char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgElement* child, FABS dimx)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  FABS reld = (selfarea->right.rel - selfarea->left.rel)*dimx;
  FABS right = (area->right.rel - area->left.rel)*(d + reld);
  if(area->right.abs + right > d + reld)
  {
    FABS ndim = area->right.abs + right - reld;
    selfarea->right.abs = selfarea->left.abs + ndim;
    area->right.abs += (area->right.rel - area->left.rel)*(d - ndim);
    return 2;
  }
  return 0;
}

inline char FG_FASTCALL fgLayout_ExpandY(CRect* selfarea, fgElement* child, FABS dimy)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->bottom.abs - selfarea->top.abs;
  FABS reld = (selfarea->bottom.rel - selfarea->top.rel)*dimy;
  FABS bottom = (area->bottom.rel - area->top.rel)*(d + reld);
  if(area->bottom.abs + bottom > d + reld)
  {
    FABS ndim = area->bottom.abs + bottom - reld;
    selfarea->bottom.abs = selfarea->top.abs + ndim;
    area->bottom.abs += (area->bottom.rel - area->top.rel)*(d - ndim); // Equivelent to: area->bottom.rel*(d + reld) - area->bottom.rel*(ndim + reld)
    return 4;
  }
  return 0;
}

size_t FG_FASTCALL fgLayout_Default(fgElement* self, const FG_Msg* msg, CRect* area, AbsRect* parent)
{
  if(!(self->flags & FGELEMENT_EXPAND))
    return 0;

  fgElement* child = (fgElement*)msg->other;

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTREORDER:
  case FGELEMENT_LAYOUTRESIZE:
    break;
  case FGELEMENT_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGELEMENT_LAYOUTREMOVE;
    size_t dim = fgLayout_Default(self, &m, area, parent);
    m.subtype = FGELEMENT_LAYOUTADD;
    dim |= fgLayout_Default(self, &m, area, parent);
    return dim;
  }
  case FGELEMENT_LAYOUTADD:
  {
    if(child->flags & FGELEMENT_BACKGROUND)
      break;
    char dim = 0;
    if(self->flags & FGELEMENT_EXPANDX)
      dim |= fgLayout_ExpandX(area, child, parent->right - parent->left);
    if(self->flags & FGELEMENT_EXPANDY)
      dim |= fgLayout_ExpandY(area, child, parent->bottom - parent->top);
    return dim;
  }
  case FGELEMENT_LAYOUTREMOVE:
  {
    if(child->flags & FGELEMENT_BACKGROUND)
      break;
    CRect* childarea = &child->transform.area;
    FABS dx = self->transform.area.right.abs - self->transform.area.left.abs;
    FABS dy = self->transform.area.bottom.abs - self->transform.area.top.abs;
    if(((self->flags & FGELEMENT_EXPANDX) &&
      childarea->right.rel == childarea->left.rel &&
      childarea->right.abs >= dx) ||
      ((self->flags & FGELEMENT_EXPANDY) &&
        childarea->bottom.rel == childarea->top.rel &&
        childarea->bottom.abs >= dy))
    {
      FG_Msg m = *msg;
      m.subtype = FGELEMENT_LAYOUTRESET;
      m.other = child;
      return fgLayout_Default(self, &m, area, parent);
    }
  }
  case FGELEMENT_LAYOUTRESET:
  {
    fgElement* hold = self->root;
    if(self->flags & FGELEMENT_EXPANDX)
      area->right.abs = area->left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
    if(self->flags & FGELEMENT_EXPANDY)
      area->bottom.abs = area->top.abs;
    char dim = 0;

    while(hold) // O(n) by necessity
    {
      if(hold != child)
      {
        if(self->flags & FGELEMENT_EXPANDX)
          dim |= fgLayout_ExpandX(area, hold, parent->right - parent->left);
        if(self->flags & FGELEMENT_EXPANDY)
          dim |= fgLayout_ExpandY(area, hold, parent->bottom - parent->top);
      }
      hold = hold->next;
    }

    return dim;
  }
  }

  return 0;
}

char fgLayout_TileSame(FABS posx, FABS posy, fgElement* cur)
{
  return (posx > 0 || posy > 0) && posx == cur->transform.area.left.abs && posy == cur->transform.area.left.abs;
}

void fgLayout_TileReorder(fgElement* prev, fgElement* child, char axis, float max, float pitch) // axis: 0 means x-axis first, 1 means y-axis first
{
  AbsVec pos = { 0,0 };
  if(prev)
  {
    pos.x = axis ? prev->transform.area.bottom.abs : prev->transform.area.right.abs;
    pos.y = axis ? prev->transform.area.left.abs : prev->transform.area.top.abs;
  }
  fgElement* cur = child;
  char rowbump = 0; // Used to keep track if we've gone down at least one row - this is required for our same position optimization to be valid.

  while(cur) // O(n) complexity, but should break after re-ordering becomes unnecessary.
  {
    if(!(cur->flags&FGELEMENT_BACKGROUND))
    {
      if(fgLayout_TileSame(axis ? pos.y : pos.x, axis ? pos.x : pos.y, cur) && rowbump)
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
    }
    cur = cur->next;
  }
}

FABS FG_FASTCALL fgLayout_TileGetPitch(fgElement* cur, char axis)
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
    cur = cur->prev; // run backwards until we go up a level
  }
  return pitch;
}

size_t FG_FASTCALL fgLayout_Tile(fgElement* self, const FG_Msg* msg, char axes)
{
  fgElement* child = (fgElement*)msg->other;
  char axis = ((axes & 3) && (axes & 8)) || ((axes & 2) && !(axes & 1));
  FABS max = INFINITY;
  if(axes & 3)
  {
    AbsRect out;
    ResolveRect(self, &out);
    if((axes & 8) && !(self->flags & FGELEMENT_EXPANDY)) max = out.bottom - out.top;
    else if(!(self->flags & FGELEMENT_EXPANDX)) max = out.right - out.left;
  }

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTMOVE:
    break;
  case FGELEMENT_LAYOUTRESIZE:
    if((((msg->otheraux & 0b10) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->otheraux & 0b100) && !(self->flags&FGELEMENT_EXPANDY))) && (axes & 3))
      fgLayout_TileReorder(0, self->root, axis, max, 0.0f);
    break;
  case FGELEMENT_LAYOUTREORDER:
  {
    fgElement* old = (fgElement*)msg->other2;
    fgElement* cur = old;
    while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
    child = !cur ? child : old;
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
  }
  break;
  case FGELEMENT_LAYOUTADD:
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  case FGELEMENT_LAYOUTREMOVE:
    fgLayout_TileReorder(child->prev, child->next, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Distribute(fgElement* self, const FG_Msg* msg, char axis)
{
  /*switch(msg->type)
  {
  case FGELEMENT_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGELEMENT_LAYOUTRESIZE:
  break;
  case FGELEMENT_LAYOUTREORDER:
  //child = child->order<msg->other->order?child:msg->other; // Get lowest child
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTADD:
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTREMOVE:
  fgLayout_DistributeReorder(child->prev, child->next, axis, num);
  break;
  }*/

  return 0;
}
// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgRoot.h"
#include "fgMonitor.h"
#include "fgAll.h"
#include "bss-util/Trie.h"
#include <stdlib.h>
#include <sstream>

struct fgTypeInit {
  fgInitializer init;
  size_t size;
  fgFlag flags;
};

using namespace bss;

KHASH_INIT(fgIDMap, const char*, fgElement*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgIDHash, fgElement*, const char*, 1, kh_ptr_hash_func, kh_int_hash_equal);
KHASH_INIT(fgInitMap, const char*, fgTypeInit, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgCursorMap, unsigned int, void*, 1, kh_int_hash_func, kh_int_hash_equal);

fgRoot* fgroot_instance = 0;

void fgRoot_Init(fgRoot* self, const AbsRect* area, const AbsVec* dpi, const fgBackend* backend)
{
  static fgBackend DEFAULT_BACKEND = {
    FGTEXTFMT_UTF8,
    &fgCreateFontDefault,
    &fgCloneFontDefault,
    &fgDestroyFontDefault,
    &fgDrawFontDefault,
    &fgFontLayoutDefault,
    &fgFontGetDefault,
    &fgFontIndexDefault,
    &fgFontPosDefault,
    &fgCreateAssetFileDefault,
    &fgCreateAssetDefault,
    &fgCloneAssetDefault,
    &fgDestroyAssetDefault,
    &fgDrawAssetDefault,
    &fgAssetSizeDefault,
    &fgDrawLinesDefault,
    &fgCreateDefault,
    &fgFlagMapDefault,
    &fgMessageMapDefault,
    &fgUserDataMapDefault,
    &fgPushClipRectDefault,
    &fgPeekClipRectDefault,
    &fgPopClipRectDefault,
    &fgDragStartDefault,
    &fgSetCursorDefault,
    &fgClipboardCopyDefault,
    &fgClipboardExistsDefault,
    &fgClipboardPasteDefault,
    &fgClipboardFreeDefault,
    &fgDirtyElementDefault,
    &fgProcessMessagesDefault,
    &fgLoadExtensionDefault,
    &fgLogHookDefault,
    &fgTerminateDefault,
  };

  fgStyleStatic::Instance.Init();
  bssFill(*self, 0);
  self->backend = !backend ? DEFAULT_BACKEND : *backend;
  self->fgBehaviorHook = fgBehaviorHookDefault;
  self->dpi = *dpi;
  if(self->dpi.x == 0.0f && self->dpi.y == 0.0f)
    self->dpi = AbsVec_DEFAULTDPI;
  self->cursorblink = 0.53; // 530 ms is the windows default.
  self->tooltipdelay = 0.4; // 400 ms is the windows default for tooltip display time.
  self->lineheight = 30;
  self->fontscale = 1.0f;
  self->radiohash = fgRadioGroup_init();
  self->functionhash = fgFunctionMap_init();
  self->idhash = kh_init_fgIDHash();
  self->idmap = kh_init_fgIDMap();
  self->initmap = kh_init_fgInitMap();
  self->cursormap = kh_init_fgCursorMap();
  self->queue = (_FG_MESSAGEQUEUE*)calloc(1, sizeof(_FG_MESSAGEQUEUE));
  new(self->queue) _FG_MESSAGEQUEUE();
  self->aux = (_FG_MESSAGEQUEUE*)calloc(1, sizeof(_FG_MESSAGEQUEUE));
  new(self->aux) _FG_MESSAGEQUEUE();
  self->inject = fgRoot_DefaultInject;
  fgroot_instance = self;
  fgTransform transform = { area->left, 0, area->top, 0, area->right, 0, area->bottom, 0, 0, 0, 0 };
  fgElement_InternalSetup(*self, 0, 0, 0, 0, &transform, 0, (fgDestroy)&fgRoot_Destroy, (fgMessage)&fgRoot_Message);
  self->gui.element.style = 0;
#ifdef BSS_DEBUG
  self->maxloglevel = FGLOG_DEBUG;
#else
  self->maxloglevel = FGLOG_NOTICE;
#endif

  fgStyle_AddGroupNames(4, "neutral", "hover", "active", "disabled");
  fgStyle_AddGroupNames(3, "unchecked", "checked", "indeterminate");
  fgStyle_AddGroupNames(2, "visible", "hidden");
  
  fgRegisterControl("element", fgElement_Init, sizeof(fgElement), 0);
  fgRegisterControl("control", (fgInitializer)fgControl_Init, sizeof(fgControl), 0);
  fgRegisterControl("resource", (fgInitializer)fgResource_Init, sizeof(fgResource), FGELEMENT_IGNORE);
  fgRegisterControl("text", (fgInitializer)fgText_Init, sizeof(fgText), FGELEMENT_IGNORE);
  fgRegisterControl("box", (fgInitializer)fgBox_Init, sizeof(fgBox), 0);
  fgRegisterControl("scrollbar", (fgInitializer)fgScrollbar_Init, sizeof(fgScrollbar), 0);
  fgRegisterControl("button", (fgInitializer)fgButton_Init, sizeof(fgButton), 0);
  fgRegisterControl("window", (fgInitializer)fgWindow_Init, sizeof(fgWindow), 0);
  fgRegisterControl("checkbox", (fgInitializer)fgCheckbox_Init, sizeof(fgCheckbox), 0);
  fgRegisterControl("radiobutton", (fgInitializer)fgRadiobutton_Init, sizeof(fgRadiobutton), 0);
  fgRegisterControl("progressbar", (fgInitializer)fgProgressbar_Init, sizeof(fgProgressbar), 0);
  fgRegisterControl("slider", (fgInitializer)fgSlider_Init, sizeof(fgSlider), 0);
  fgRegisterControl("textbox", (fgInitializer)fgTextbox_Init, sizeof(fgTextbox), 0);
  fgRegisterControl("treeview", (fgInitializer)fgTreeview_Init, sizeof(fgTreeview), 0);
  fgRegisterControl("treeitem", (fgInitializer)fgTreeItem_Init, sizeof(fgTreeItem), 0);
  fgRegisterControl("list", (fgInitializer)fgList_Init, sizeof(fgList), 0);
  fgRegisterControl("listitem", (fgInitializer)fgListItem_Init, sizeof(fgControl), FGBOX_TILEY | FGELEMENT_EXPANDY);
  fgRegisterControl("curve", (fgInitializer)fgCurve_Init, sizeof(fgCurve), 0);
  fgRegisterControl("dropdown", (fgInitializer)fgDropdown_Init, sizeof(fgDropdown), 0);
  fgRegisterControl("tabcontrol", (fgInitializer)fgTabcontrol_Init, sizeof(fgTabcontrol), 0);
  fgRegisterControl("menu", (fgInitializer)fgMenu_Init, sizeof(fgMenu), FGELEMENT_EXPANDY | FGBOX_TILEX);
  fgRegisterControl("submenu", (fgInitializer)fgSubmenu_Init, sizeof(fgMenu), FGELEMENT_NOCLIP | FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGBOX_TILEY | FGELEMENT_EXPAND);
  fgRegisterControl("contextmenu", (fgInitializer)fgContextMenu_Init, sizeof(fgMenu), FGELEMENT_NOCLIP | FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGBOX_TILEY | FGELEMENT_EXPAND);
  fgRegisterControl("menuitem", (fgInitializer)fgMenuItem_Init, sizeof(fgMenuItem), FGELEMENT_EXPAND | FGELEMENT_NOCLIP);
  fgRegisterControl("grid", (fgInitializer)fgGrid_Init, sizeof(fgGrid), FGBOX_TILEY);
  fgRegisterControl("gridrow", (fgInitializer)fgGridRow_Init, sizeof(fgGridRow), FGBOX_TILEX|FGELEMENT_EXPAND);
  fgRegisterControl("workspace", (fgInitializer)fgWorkspace_Init, sizeof(fgWorkspace), 0);
  fgRegisterControl("toolbar", (fgInitializer)fgToolbar_Init, sizeof(fgToolbar), FGBOX_TILE);
  fgRegisterControl("toolgroup", (fgInitializer)fgToolGroup_Init, sizeof(fgBox), FGBOX_TILE|FGELEMENT_EXPAND);
  fgRegisterControl("combobox", (fgInitializer)fgCombobox_Init, sizeof(fgCombobox), 0);
  fgRegisterControl("debug", (fgInitializer)fgDebug_Init, sizeof(fgDebug), FGELEMENT_BACKGROUND);

  self->fgTooltip = fgCreate("text", &self->gui.element, 0, "Root$Tooltip", FGELEMENT_EXPAND | FGFLAGS_INTERNAL | FGELEMENT_HIDDEN | FGELEMENT_IGNORE | FGELEMENT_BACKGROUND, &fgTransform_EMPTY, 0);
}

void fgRoot_Destroy(fgRoot* self)
{
  if(fgdebug_instance != 0)
    VirtualFreeChild(*fgdebug_instance);
  fgControl_Destroy((fgControl*)self);
  self->queue->~_FG_MESSAGEQUEUE();
  free(self->queue);
  self->aux->~_FG_MESSAGEQUEUE();
  free(self->aux);
  fgRadioGroup_destroy(self->radiohash);
  fgFunctionMap_destroy(self->functionhash);
  kh_destroy_fgIDHash(self->idhash);
  kh_destroy_fgIDMap(self->idmap); // We don't need to clear this because it will have already been emptied.
  for(khiter_t i = 0; i < self->initmap->n_buckets; ++i) // We do have to clear this one, though.
    if(i != kh_end(self->initmap) && kh_exist(self->initmap, i))
      fgFreeText(kh_key(self->initmap, i), __FILE__, __LINE__);
  kh_destroy_fgInitMap(self->initmap);
  for(khiter_t i = 0; i < self->cursormap->n_buckets; ++i)
    if(i != kh_end(self->cursormap) && kh_exist(self->cursormap, i))
      fgfree(kh_val(self->cursormap, i), __FILE__, __LINE__);
  kh_destroy_fgCursorMap(self->cursormap);
  fgStyleStatic::Instance.Clear();
#ifdef BSS_DEBUG
  fgLeakTracker::Tracker.Dump();
#endif
}

void fgRoot_CheckMouseMove(fgRoot* self)
{
  if(self->mouse.state&FGMOUSE_SEND_MOUSEMOVE)
  {
    self->mouse.state &= ~FGMOUSE_SEND_MOUSEMOVE;
    FG_Msg m = { 0 };
    m.type = FG_MOUSEMOVE;
    m.x = self->mouse.x;
    m.y = self->mouse.y;
    m.allbtn = self->mouse.buttons;
    self->inject(self, &m);
  }
}

size_t fgRoot_Message(fgRoot* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEMOVE:
  case FG_KEYCHAR: // If these messages get sent to the root, they have been rejected from everything else.
  case FG_KEYUP:
  case FG_KEYDOWN:
  case FG_MOUSEOFF:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSESCROLL:
  case FG_GOTFOCUS: //Root cannot have focus
  case FG_CLONE:
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"Root";
  case FG_DRAW:
  {
	  fgRoot_CheckMouseMove(self);
	  CRect* rootarea = &self->gui.element.transform.area;
    AbsRect area = { rootarea->left.abs, rootarea->top.abs, rootarea->right.abs, rootarea->bottom.abs };
    fgDrawAuxData data = {
      sizeof(fgDrawAuxData),
      self->dpi,
      { 1,1 },
      { 0,0 }
    };
    FG_Msg m = *msg;
    m.p = &area;
    m.p2 = &data;
    fgControl_Message((fgControl*)self, &m);
    if(self->topmost) // Draw topmost before the drag object
    {
      AbsRect out;
      ResolveRect(self->topmost, &out);
      self->topmost->Draw(&out, &data);
    }
    if(self->dragdraw != 0 && self->dragdraw->parent != *self)
    {
      AbsRect out;
      ResolveRect(self->dragdraw, &out);
      FABS dx = out.right - out.left;
      FABS dy = out.bottom - out.top;
      out.left = (FABS)self->mouse.x;
      out.top = (FABS)self->mouse.y;
      out.right = out.left + dx;
      out.bottom = out.top + dy;
      self->dragdraw->Draw(&out, &data);
    }
    return FG_ACCEPT;
  }
  case FG_GETDPI:
    return (size_t)&self->dpi;
  case FG_SETDPI:
  {
    AbsVec scale = { !self->dpi.x ? 1.0f : (msg->f / self->dpi.x), !self->dpi.y ? 1.0f : (msg->f / self->dpi.y) };
    CRect* rootarea = &self->gui.element.transform.area;
    CRect area = { rootarea->left.abs*scale.x, 0, rootarea->top.abs*scale.y, 0, rootarea->right.abs*scale.x, 0, rootarea->bottom.abs*scale.y, 0 };
    self->dpi.x = msg->f;
    self->dpi.y = msg->f2;
    return self->gui.element.SetArea(area);
  }
    return FG_ACCEPT;
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->f;
    return FG_ACCEPT;
  case FG_GETSTYLE:
    return 0;
  }
  return fgControl_Message((fgControl*)self,msg);
}

char BSS_FORCEINLINE fgStandardApplyClipping(fgFlag flags, const AbsRect* area, char clipping, const fgDrawAuxData* aux)
{
  if(!clipping && !(flags&FGELEMENT_NOCLIP))
  {
    clipping = true;
    fgroot_instance->backend.fgPushClipRect(area, aux);
  }
  else if(clipping && (flags&FGELEMENT_NOCLIP))
  {
    clipping = false;
    fgroot_instance->backend.fgPopClipRect(aux);
  }
  return clipping;
}

char BSS_FORCEINLINE fgStandardDrawElement(AbsRect* padding, fgElement* hold, const AbsRect* area, const fgDrawAuxData* aux, AbsRect& curarea, char clipping)
{
  if(!(hold->flags&FGELEMENT_HIDDEN) && hold != fgroot_instance->topmost)
  {
    ResolveRectCache(hold, &curarea, area, (hold->flags & FGELEMENT_BACKGROUND) ? 0 : padding);
    clipping = fgStandardApplyClipping(hold->flags, area, clipping, aux);

    AbsRect clip = fgroot_instance->backend.fgPeekClipRect(aux);
    char culled = !fgRectIntersect(&curarea, &clip);
    _sendsubmsg<FG_DRAW, void*, const void*>(hold, culled, &curarea, aux);
  }
  return clipping;
}

// This must be its own function due to the way alloca works.
inline char fgDrawSkinElement(fgVector* skinstyle, AbsRect* padding, fgSkinLayout& child, const AbsRect* area, const fgDrawAuxData* aux, AbsRect& curarea, char clipping)
{
  fgElement* element = child.instance;
  if(skinstyle != 0)
  {
    element = (fgElement*)ALLOCA(child.instance->Clone());
    child.instance->Clone(element);
    element->free = 0;
    element->flags |= FGELEMENT_SILENT;
    fgElement_ApplyMessageArray(child.instance, element, skinstyle);
  }
  char r = fgStandardDrawElement(padding, element, area, aux, curarea, clipping);

  AbsRect childarea;
  for(size_t i = 0; i < child.tree.children.l; ++i)
    clipping = fgDrawSkinElement(skinstyle, padding, child.tree.children.p[i], &curarea, aux, childarea, clipping);

  if(skinstyle != 0)
    VirtualFreeChild(element);
  return r;
}

char fgDrawSkinPartial(fgVector* skinstyle, AbsRect* padding, const fgSkin* skin, const AbsRect* area, const fgDrawAuxData* aux, char culled, char foreground, char clipping)
{
  if(skin != 0)
  {
    clipping = fgDrawSkinPartial(skinstyle, padding, skin->base.inherit, area, aux, culled, foreground, clipping);

    AbsRect curarea;
    for(size_t i = 0; i < skin->tree.children.l; ++i)
      if((skin->tree.children.p[i].element.order > 0) == foreground)
        clipping = fgDrawSkinElement(skinstyle, padding, skin->tree.children.p[i], area, aux, curarea, clipping);
  }

  return clipping;
}

char fgDrawSkinInherit(const struct _FG_SKIN* skin, const AbsRect* area, const fgDrawAuxData* aux, char culled, char clipping)
{
  if(skin != 0)
  {
    clipping = fgDrawSkinInherit(skin->base.inherit, area, aux, culled, clipping);

    AbsRect curarea;
    for(size_t i = 0; i < skin->tree.children.l; ++i)
      clipping = fgDrawSkinElement(0, 0, skin->tree.children.p[i], area, aux, curarea, clipping);
  }

  return clipping;
}

void fgDrawSkin(const struct _FG_SKIN* skin, const AbsRect* area, const fgDrawAuxData* aux, char culled)
{
  bool clipping = false;
  clipping = fgDrawSkinInherit(skin, area, aux, culled, clipping);

  if(clipping)
    fgroot_instance->backend.fgPopClipRect(aux);
}

void fgStandardDraw(fgElement* self, const AbsRect* area, const fgDrawAuxData* aux, char culled, fgAuxDrawFunction draw)
{
  fgElement* hold = culled ? self->rootnoclip : self->root;
  AbsRect curarea;
  bool clipping = false;

  clipping = fgDrawSkinPartial(self->skinstyle, &self->padding, self->skin, area, aux, culled, false, clipping);

  if(draw && !culled)
    draw(self, area, aux, hold);

  while(hold)
  {
    clipping = fgStandardDrawElement(&self->padding, hold, area, aux, curarea, clipping);
    hold = culled ? hold->nextnoclip : hold->next;
  }

  clipping = fgDrawSkinPartial(self->skinstyle, &self->padding, self->skin, area, aux, culled, true, clipping);

  if(clipping)
    fgroot_instance->backend.fgPopClipRect(aux);
}

void fgOrderedDraw(fgElement* self, const AbsRect* area, const fgDrawAuxData* aux, char culled, fgElement* skip, fgElement* (*fn)(fgElement*, const AbsRect*, const AbsRect*), void(*draw)(fgElement*, const AbsRect*, const fgDrawAuxData*, fgElement*), fgElement* selected)
{
  if(culled) // If we are culled, there's no point drawing ordered elements, because ordered elements aren't non-clipping, so we let the standard draw take care of it.
    return fgStandardDraw(self, area, aux, culled, draw);

  fgElement* cur = self->root;
  AbsRect curarea;
  bool clipping = false;

  clipping = fgDrawSkinPartial(self->skinstyle, &self->padding, self->skin, area, aux, culled, false, clipping);

  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND)) // Render all background elements before the ordered elements
  {
    clipping = fgStandardDrawElement(&self->padding, cur, area, aux, curarea, clipping);
    cur = cur->next;
  }

  AbsRect out;
  AbsRect clip = fgroot_instance->backend.fgPeekClipRect(aux);
  fgRectIntersection(area, &clip, &out);
  // do binary search on the absolute resolved bottomright coordinates compared to the topleft corner of the render area
  cur = fn(self, &out, area);

  if(draw)
    draw(self, area, aux, cur);

  if(!clipping)
  {
    clipping = true; // always clipping at this stage because ordered elements can't be nonclipping
    fgroot_instance->backend.fgPushClipRect(area, aux);
  }
  char cull = 0;

  while(!cull && cur != 0 && !(cur->flags & FGELEMENT_BACKGROUND)) // Render all ordered elements until they go outside of the culling rect.
  {
    if(cur != fgroot_instance->topmost && cur != selected && !(cur->flags&FGELEMENT_HIDDEN))
    {
      ResolveRectCache(cur, &curarea, area, &self->padding); // always apply padding because these are always foreground elements
      AbsRect clip = fgroot_instance->backend.fgPeekClipRect(aux);
      cull = !fgRectIntersect(&curarea, &clip);
      _sendsubmsg<FG_DRAW, void*, const void*>(cur, cull, &curarea, aux);
      switch(self->flags&(FGBOX_TILE | FGBOX_GROWY))
      {
      case FGBOX_TILEX:
      case FGBOX_TILE | FGBOX_GROWY: // TILEX and TILE growing along the Y axis both get culled once we hit an element that is past the right edge of the cliprect
        cull = curarea.left > clip.right;
        break;
      case FGBOX_TILEY:
      case FGBOX_TILE: // TILE growing along the X axis gets culled once we hit an element that is past the bottom edge of the cliprect
        cull = curarea.top > clip.bottom;
        break;
      }
    }
    cur = cur->next;
  }

  if(selected != 0 && !(selected->flags&FGELEMENT_HIDDEN))
  {
    ResolveRectCache(selected, &curarea, area, &self->padding); // always apply padding because these are always foreground elements
    AbsRect clip = fgroot_instance->backend.fgPeekClipRect(aux);
    cull = !fgRectIntersect(&curarea, &clip);
    _sendsubmsg<FG_DRAW, void*, const void*>(selected, cull, &curarea, aux);
  }

  cur = skip;
  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND)) // Render all background elements after the ordered elements
  {
    clipping = fgStandardDrawElement(&self->padding, cur, area, aux, curarea, clipping);
    cur = cur->next;
  }

  clipping = fgDrawSkinPartial(self->skinstyle, &self->padding, self->skin, area, aux, culled, true, clipping);

  if(clipping)
    fgroot_instance->backend.fgPopClipRect(aux);
}

void fgFixedDraw(fgElement* self, AbsRect* area, size_t dpi, char culled, fgElement** ordered, size_t numordered, AbsVec dim)
{

}

// Recursive event injection function
size_t fgStandardInject(fgElement* self, const FG_Msg* msg, const AbsRect* area)
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // If this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND || !self->parent) ? 0 : &self->parent->padding);

  bool miss = (area != 0 && !MsgHitAbsRect(msg, &curarea)); // If the area is null, the message always hits.
  fgElement* cur = miss ? self->lastnoclip : self->lastinject; // If the event completely misses us, evaluate only nonclipping elements.
  size_t r;
  while(cur) // Try to inject to any children we have
  {
    if(!(cur->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea))) // We have to check FGELEMENT_IGNORE because the noclip list may have render-only elements in it.
      return r; // If the message is NOT rejected, return the result immediately to indicate we accepted the message.
    cur = miss ? cur->prevnoclip : cur->previnject; // Otherwise the child rejected the message.
  }

  // If we get this far either we have no children, the event missed them all, or they all rejected the event...
  return miss ? 0 : (*fgroot_instance->fgBehaviorHook)(self,msg); // So we give the event to ourselves, but only if it didn't miss us (which can happen if we were evaluating nonclipping elements)
}

size_t fgOrderedInject(fgElement* self, const FG_Msg* msg, const AbsRect* area, fgElement* skip, fgElement* (*fn)(fgElement*, const FG_Msg*), fgElement* selected)
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // If this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND || !self->parent) ? 0 : &self->parent->padding);

  size_t r;
  if(area != 0 && !MsgHitAbsRect(msg, &curarea)) // if this misses us, only evaluate nonclipping elements. Don't bother with the ordered array.
  {
    if(selected != 0 && (selected->flags&FGELEMENT_NOCLIP) != 0) // Check our selected item first, but only if it's nonclipping
    {
      if(!(selected->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(selected, msg, &curarea)))
        return r; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
    }

    fgElement* cur = self->lastnoclip;
    while(cur) // Try to inject to any children we have
    {
      if(cur != selected && !(cur->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea))) // We have to check FGELEMENT_IGNORE because the noclip list may have render-only elements in it.
        return r; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
      cur = cur->prevnoclip; // Otherwise the child rejected the message.
    }

    return 0; // We either had no nonclipping children or it missed them all, so reject this.
  }

  if(selected != 0 && !(selected->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(selected, msg, &curarea)))
    return r; // Check our selected first, if it exists.

  fgElement* cur = self->lastinject;
  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND))
  {
    if(cur != selected && !(cur->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)))
      return r;
    cur = cur->previnject; 
  }

  cur = fn(self, msg);
  if(!(cur->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)))
    return r;

  cur = skip;
  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND))
  {
    if(!(cur->flags&FGELEMENT_IGNORE) && (r = _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)))
      return r;
    cur = cur->previnject;
  }

  return (*fgroot_instance->fgBehaviorHook)(self, msg); // So we give the event to ourselves because it couldn't have missed us if we got to this point
}

BSS_FORCEINLINE size_t fgProcessCursor(fgRoot* self, size_t value, fgMsgType type, FG_CURSOR fallback = FGCURSOR_NONE)
{
  static bool FROMCHECK = false;
  unsigned int cursor = (unsigned int)value;
  if(type != FG_MOUSEMOVE && type != FG_DRAGOVER)
  {
    cursor = self->cursor; // If this isn't a mousemove/dragover, we don't change the cursor type, but we do set it anyway in case the OS tries to do weird shit under our noses.
    fallback = FGCURSOR_NONE;
  }
  if(!cursor && fallback != FGCURSOR_NONE)
    cursor = fallback;
  if(cursor != 0)
  {
    self->cursor = cursor;
    void* data = 0;
    if(self->cursormap->n_buckets > 0)
    {
      khiter_t i = kh_get_fgCursorMap(self->cursormap, cursor);
      if(i != kh_end(self->cursormap) && kh_exist(self->cursormap, i))
        data = kh_val(self->cursormap, i);
    }
    if(cursor == FGCURSOR_IBEAM)
      FROMCHECK = true;
    self->backend.fgSetCursor(cursor, data);
  }
  return value;
}

// Before directly injecting messages to topmost/captured windows, we have to manually scale the message DPI if necessary.
size_t _doRootInject(fgRoot* self, fgElement* target, const FG_Msg* msg)
{
  const AbsVec& dpi = target->GetDPI();
  if((dpi.x == 0.0f || dpi.x == self->dpi.x) && (dpi.y == 0.0f || dpi.y == self->dpi.y))
    return _sendmsg<FG_INJECT, const void*, const void*>(target, msg, 0);

  FG_Msg m = *msg;
  m.x *= (self->dpi.x / dpi.x);
  m.y *= (self->dpi.y / dpi.y);
  return _sendmsg<FG_INJECT, const void*, const void*>(target, &m, 0);
}

size_t fgRoot_DefaultInject(fgRoot* self, const FG_Msg* msg)
{
  assert(self != 0);

  CRect* rootarea = &self->gui.element.transform.area;
  fgUpdateMouseState(&self->mouse, msg);
  FG_Msg m;

  if(self->dragtype != FGCLIPBOARD_NONE && (msg->type == FG_MOUSEMOVE || msg->type == FG_MOUSEUP))
  {
    m = *msg;
    m.type = (msg->type == FG_MOUSEMOVE) ? FG_DRAGOVER : FG_DROP;
    msg = &m;
  }

  switch(msg->type)
  {
  case FG_KEYUP:
  case FG_KEYDOWN:
    self->keys[msg->keycode / 32] = (msg->type == FG_KEYDOWN) ? (self->keys[msg->keycode / 32] | (1 << (msg->keycode % 32))) : (self->keys[msg->keycode / 32] & (~(1 << (msg->keycode % 32))));
  case FG_JOYBUTTONDOWN:
  case FG_JOYBUTTONUP:
  case FG_JOYAXIS:
  case FG_KEYCHAR:
  {
    fgElement* cur = !self->fgFocusedWindow ? *self : self->fgFocusedWindow;
    do
    {
      if((*self->fgBehaviorHook)(cur, msg))
        return FG_ACCEPT;
      cur = cur->parent;
    } while(cur);
    return 0;
  }
  case FG_MOUSESCROLL:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
    if(self->dragdraw != 0 && self->dragdraw->parent == *self)
      MoveCRect((FABS)msg->x, (FABS)msg->y, &self->dragdraw->transform.area);

    if(self->fgCaptureWindow)
      if(fgProcessCursor(self, _doRootInject(self, self->fgCaptureWindow, msg), msg->type)) // If it's captured, send the message to the captured window with NULL area.
        return FG_ACCEPT;

    if(self->topmost) // After we attempt sending the message to the captured window, try sending it to the topmost
      if(fgProcessCursor(self, _doRootInject(self, self->topmost, msg), msg->type))
        return FG_ACCEPT;

    if(fgProcessCursor(self, _sendmsg<FG_INJECT, const void*, const void*>(*self, msg, 0), msg->type))
      return FG_ACCEPT;
    if(msg->type != FG_MOUSEMOVE)
      break;
    fgProcessCursor(self, FGCURSOR_ARROW, msg->type);
  case FG_MOUSEOFF:
    if(self->fgLastHover != 0) // If we STILL haven't accepted a mousemove event, send a MOUSEOFF message if lasthover exists
    {
      _sendmsg<FG_MOUSEOFF>(self->fgLastHover);
      self->fgLastHover = 0;
    }
    if(msg->type != FG_MOUSEOFF)
      break;
  case FG_MOUSEON:
    break;
  case FG_DRAGOVER:
  case FG_DROP:
    fgProcessCursor(self, _sendmsg<FG_INJECT, const void*, const void*>(*self, msg, 0), msg->type, FGCURSOR_NO);
    if(msg->type == FG_DROP)
    {
      self->dragtype = FGCLIPBOARD_NONE;
      if(self->dragdraw != 0)
      {
        if(self->dragdraw->parent == *self)
          VirtualFreeChild(self->dragdraw);
        self->dragdraw = 0;
      }
      self->dragdata = 0;
    }
    break;
  }
  return 0;
}

fgElement* fgSetTopmost(fgElement* target)
{
  fgElement* prev = fgroot_instance->topmost;
  fgroot_instance->topmost = target;
  fgroot_instance->backend.fgDirtyElement(target);
  return prev;
}
char fgClearTopmost(fgElement* target)
{
  if(fgroot_instance->topmost == target)
  {
    fgroot_instance->topmost = 0;
    fgroot_instance->backend.fgDirtyElement(target);
    return 1;
  }
  return 0;
}

void fgRoot_Update(fgRoot* self, double delta)
{
  fgDeferAction* cur;
  self->time += delta;

  while((cur = self->updateroot) && (cur->time <= self->time))
  {
    self->updateroot = cur->next; // Remove the node from the update list
    if((*cur->action)(cur->arg)) // If this returns true, we deallocate the node
      fgfree(cur, __FILE__, __LINE__);
  }
}

fgMonitor* fgRoot_GetMonitor(const fgRoot* self, const AbsRect* rect)
{
  fgMonitor* cur = self->monitors;
  float largest = 0;
  fgMonitor* last = 0; // it is possible for all intersections to have an area of zero, meaning the element is not on ANY monitors and is thus not visible.

  while(cur)
  {
    AbsRect area;
    ResolveRect(&cur->element, &area);
    AbsRect monitor = { area.left > rect->left ? area.left : rect->left,
      area.top > rect->top ? area.top : rect->top,
      area.right < rect->right ? area.right : rect->right,
      area.bottom < rect->bottom ? area.bottom : rect->bottom };

    float total = bssmax(monitor.right - monitor.left, 0)*bssmax(monitor.bottom - monitor.top, 0);
    if(total > largest) // This must be GREATER THAN to ensure that a value of "0" is not ever assigned a monitor.
    {
      largest = total;
      last = cur;
    }

    cur = cur->mnext;
  }

  return last;
}

fgDeferAction* fgAllocAction(char (*action)(void*), void* arg, double time)
{
  fgDeferAction* r = fgmalloc<fgDeferAction>(1, __FILE__, __LINE__);
  r->action = action;
  r->arg = arg;
  r->time = time;
  r->next = 0; // We do this so its never ambigious if an action is in the list already or not
  r->prev = 0;
  return r;
}

void fgDeallocAction(fgDeferAction* action)
{
  fgRemoveAction(action);
  fgfree(action, __FILE__, __LINE__);
}

void fgAddAction(fgDeferAction* action)
{
  if((action->next != 0 && action->next->time < action->time) || (action->prev != 0 && action->prev->time > action->time))
    fgRemoveAction(action);
  else if(action->prev != 0 || action == fgroot_instance->updateroot) // If true, you're already in the list and we didn't need to move you
    return;

  fgDeferAction* cur = fgroot_instance->updateroot;
  fgDeferAction* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(action != 0 && !action->prev && action != fgroot_instance->updateroot);
  while(cur != 0 && cur->time < action->time)
  {
    prev = cur;
    cur = cur->next;
  }
  action->next = cur;
  action->prev = prev;
  if(prev) prev->next = action;
  else fgroot_instance->updateroot = action; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev = action; // Cur is null if we are at the end of the list.
}

void fgRemoveAction(fgDeferAction* action)
{
  if(action->prev != 0 || action == fgroot_instance->updateroot)
  {
    if(action->prev != 0) action->prev->next = action->next;
    else fgroot_instance->updateroot = action->next;
    if(action->next != 0) action->next->prev = action->prev;
    action->next = 0; // We do this so its never ambigious if an action is in the list already or not
    action->prev = 0;
  }
}

fgElement* fgGetID(const char* id)
{
  khiter_t i = kh_get_fgIDMap(fgroot_instance->idmap, const_cast<char*>(id));
  if(i != kh_end(fgroot_instance->idmap) && kh_exist(fgroot_instance->idmap, i))
    return kh_val(fgroot_instance->idmap, i);
  return 0;
}

#ifdef BSS_DEBUG
void VERIFY_IDHASH()
{
  for(khiter_t i = 0; i < fgroot_instance->idhash->n_buckets; ++i)
    if(i != kh_end(fgroot_instance->idhash) && kh_exist(fgroot_instance->idhash, i))
      assert(kh_val(fgroot_instance->idhash, i) != (void*)0xcdcdcdcdcdcdcdcd);
}
#endif

void fgAddID(const char* id, fgElement* element)
{
  int r;
  khiter_t i = kh_get_fgIDMap(fgroot_instance->idmap, const_cast<char*>(id));
  if(i != kh_end(fgroot_instance->idmap) && kh_exist(fgroot_instance->idmap, i))  // the key already exists so we need to remove and replace the previous element
  {
    assert(i != kh_end(fgroot_instance->idmap));
    fgLog(FGLOG_WARNING, "Replacing duplicate element ID: %s (%p -> %p)", id, kh_val(fgroot_instance->idmap, i), element);
    kh_del_fgIDHash(fgroot_instance->idhash, kh_get_fgIDHash(fgroot_instance->idhash, kh_val(fgroot_instance->idmap, i)));
  }
  else
    i = kh_put_fgIDMap(fgroot_instance->idmap, fgCopyText(id, __FILE__, __LINE__), &r);

  kh_val(fgroot_instance->idmap, i) = element;
  khiter_t j = kh_put_fgIDHash(fgroot_instance->idhash, element, &r);
  kh_val(fgroot_instance->idhash, j) = kh_key(fgroot_instance->idmap, i);
}
char fgRemoveID(fgElement* element)
{
  khiter_t i = kh_get_fgIDHash(fgroot_instance->idhash, element);
  if(i == kh_end(fgroot_instance->idhash) || !kh_exist(fgroot_instance->idhash, i))
    return false;
  khiter_t j = kh_get_fgIDMap(fgroot_instance->idmap, kh_val(fgroot_instance->idhash, i));
  if(j != kh_end(fgroot_instance->idmap) && kh_exist(fgroot_instance->idmap, j))
    kh_del_fgIDMap(fgroot_instance->idmap, j);
  else
    assert(false);

  fgFreeText(kh_val(fgroot_instance->idhash, i), __FILE__, __LINE__);
  kh_del_fgIDHash(fgroot_instance->idhash, i);
  return true;
}

fgRoot* fgSingleton()
{
  return fgroot_instance;
}

fgElement* fgCreate(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  return fgroot_instance->backend.fgCreate(type, parent, next, name, flags, transform, units);
}

void fgRegisterControl(const char* name, fgInitializer fn, size_t sz, fgFlag flags)
{
  int r;
  khint_t i = kh_put_fgInitMap(fgroot_instance->initmap, const_cast<char*>(name), &r);
  if(r != 0)
    kh_key(fgroot_instance->initmap, i) = fgCopyText(name, __FILE__, __LINE__);
  else
    fgLog(FGLOG_INFO, "Replacing duplicate control name: %s", name);
  kh_val(fgroot_instance->initmap, i).init = fn;
  kh_val(fgroot_instance->initmap, i).size = sz;
  kh_val(fgroot_instance->initmap, i).flags = flags;
}
void fgIterateControls(void* p, void(*fn)(void*, const char*))
{
  for(khiter_t i = 0; i < fgroot_instance->initmap->n_buckets; ++i)
    if(i != kh_end(fgroot_instance->initmap) && kh_exist(fgroot_instance->initmap, i))
      fn(p, kh_key(fgroot_instance->initmap, i));
}

int fgRegisterCursor(int cursor, const void* data, size_t sz)
{
  int r;
  khint_t i = kh_put_fgCursorMap(fgroot_instance->cursormap, cursor, &r);
  if(!r) // if something already existed, delete the data first
    fgfree(kh_val(fgroot_instance->cursormap, i), __FILE__, __LINE__);
  if(data)
  {
    kh_val(fgroot_instance->cursormap, i) = fgmalloc<char>(sz, __FILE__, __LINE__);
    MEMCPY(kh_val(fgroot_instance->cursormap, i), sz, data, sz);
  }
  else
    kh_del_fgCursorMap(fgroot_instance->cursormap, i);
  return r;
}

fgElement* fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  if(!STRICMP(type, "tab"))
  {
    fgElement* e = parent->AddItemText(name);
    if(transform) e->SetTransform(*transform);
    e->SetFlags(flags);
    return e;
  }
  if(!STRICMP(type, "column"))
  {
    fgElement* e = reinterpret_cast<fgGrid*>(parent)->InsertColumn(name, (size_t)~0);
    if(transform) e->SetTransform(*transform);
    e->SetFlags(flags);
    return e;
  }

  khint_t i = kh_get_fgInitMap(fgroot_instance->initmap, const_cast<char*>(type));
  if(i == kh_end(fgroot_instance->initmap) || !kh_exist(fgroot_instance->initmap, i))
  {
    fgLog(FGLOG_ERROR, "Attempted to create nonexistant type: %s", type);
    return 0;
  }
  fgTypeInit& ty = kh_val(fgroot_instance->initmap, i);
  if(FGFLAGS_DEFAULTS&flags)
    flags = (flags&(~FGFLAGS_DEFAULTS)) | ty.flags;

  fgElement* r = reinterpret_cast<fgElement*>(fgmalloc<char>(ty.size, type, 0));
  memset(r, 0xFD, ty.size);
  ty.init(r, parent, next, name, flags, transform, units);
#ifdef BSS_DEBUG
  r->free = &fgfreeblank;
#else
  r->free = &free; // We do this because the compiler can't figure out the inlining weirdness going on here
#endif
  return (fgElement*)r;
}

size_t fgGetTypeSize(const char* type)
{
  khint_t i = kh_get_fgInitMap(fgroot_instance->initmap, const_cast<char*>(type));
  if(i == kh_end(fgroot_instance->initmap) || !kh_exist(fgroot_instance->initmap, i))
  {
    fgLog(FGLOG_ERROR, "Attempted to get size of nonexistant type: %s", type);
    return 0;
  }
  return kh_val(fgroot_instance->initmap, i).size;
}
fgFlag fgGetTypeFlags(const char* type)
{
  if(!STRICMP(type, "tab") || !STRICMP(type, "column"))
    return 0;
  khint_t i = kh_get_fgInitMap(fgroot_instance->initmap, const_cast<char*>(type));
  if(i == kh_end(fgroot_instance->initmap) || !kh_exist(fgroot_instance->initmap, i))
  {
    fgLog(FGLOG_ERROR, "Attempted to get default flags of nonexistant type: %s", type);
    return 0;
  }
  return kh_val(fgroot_instance->initmap, i).flags;
}

void fgSendMessageAsync(fgElement* element, const FG_Msg* msg, unsigned int arg1size, unsigned int arg2size)
{
  uint32_t sz = sizeof(QUEUEDMESSAGE) + arg1size + arg2size;
  size_t r;
  _FG_MESSAGEQUEUE* q = ((std::atomic<_FG_MESSAGEQUEUE*>&)fgroot_instance->queue).load(std::memory_order_relaxed);
  for(;;)
  {
    q->lock.RLock();
    r = q->length.fetch_add(sz, std::memory_order_relaxed);
    size_t rend = r + sz;
    size_t c = q->capacity.load(std::memory_order_relaxed);
    if(rend >= c)
    {
      q->length.fetch_sub(sz, std::memory_order_relaxed);
      if(q->lock.AttemptUpgrade())
      {
        if(rend >= q->capacity.load(std::memory_order_relaxed))
        {
          c = T_FBNEXT(rend);
          q->capacity.store(c, std::memory_order_relaxed);
          q->mem.store(realloc(q->mem, c), std::memory_order_relaxed);
#ifdef BSS_DEBUG
          size_t l = q->length.load(std::memory_order_relaxed);
          memset((char*)q->mem.load(std::memory_order_relaxed) + l, 0xcd, c - l);
#endif
        }
        q->lock.Downgrade();
      }
      q->lock.RUnlock();
    }
    else
      break;
  }

  char* mem = (char*)q->mem.load(std::memory_order_relaxed);
  mem += r;
  QUEUEDMESSAGE* m = (QUEUEDMESSAGE*)mem;
  m->sz = sz;
  m->e = element;
  m->msg = *msg;

  if(arg1size > 0)
  {
    m->msg.p = m + 1;
    MEMCPY(m->msg.p, arg1size, msg->p, arg1size);
  }
  if(arg2size > 0)
  {
    m->msg.p2 = mem + sizeof(QUEUEDMESSAGE) + arg1size;
    MEMCPY(m->msg.p2, arg2size, msg->p2, arg2size);
  }
  q->lock.RUnlock();
}

fgInject fgSetInjectFunc(fgInject inject)
{
  fgInject prev = fgroot_instance->inject;
  fgroot_instance->inject = inject;
  return prev;
}

void fgIterateGroups(void* p, void(*fn)(void*, fgStyleIndex))
{
  fgStyleIndex visit = ~0;

  for(size_t i = 0; i < (sizeof(fgStyleIndex) << 3); ++i)
  {
    if((visit&(1 << i)) && fgStyleStatic::Instance.Masks[i] != (1 << i))
    {
      fn(p, fgStyleStatic::Instance.Masks[i]);
      visit &= ~fgStyleStatic::Instance.Masks[i];
    }
  }
}

size_t fgInjectDPIChange(fgElement* self, const FG_Msg* msg, const AbsRect* area, const AbsVec* dpi, const AbsVec* olddpi)
{
  if((dpi->x == 0 || dpi->x == olddpi->x) && (dpi->y == 0 || dpi->y == olddpi->y))
    return fgStandardInject(self, msg, area);
  FG_Msg m = *msg;
  m.x *= (olddpi->x / dpi->x);
  m.y *= (olddpi->y / dpi->y);
  if(area)
  {
    BSS_ALIGN(16) AbsRect rect = *area;
    BSS_ALIGN(16) float scale[4];
    scale[0] = (olddpi->x / dpi->x);
    scale[1] = (olddpi->y / dpi->y);
    scale[2] = scale[0];
    scale[3] = scale[1];
    (sseVec(rect._array)*sseVec(scale)).Set(rect._array);
    return fgStandardInject(self, &m, &rect);
  }
  return fgStandardInject(self, &m, NULL);
}

void fgRoot_ShowTooltip(fgRoot* self, const char* tooltip)
{
  if(!tooltip)
  {
    fgClearTopmost(self->fgTooltip);
    self->fgTooltip->SetFlag(FGELEMENT_HIDDEN, true);
  }
  else
  {
    self->fgTooltip->SetText(tooltip);
    CRect area = self->fgTooltip->transform.area;
    area.right.abs -= area.left.abs;
    area.bottom.abs -= area.top.abs;
    area.left.abs = self->mouse.x;
    area.top.abs = self->mouse.y + 16;
    area.right.abs += area.left.abs;
    area.bottom.abs += area.top.abs;
    self->fgTooltip->SetArea(area);
    self->fgTooltip->SetFlag(FGELEMENT_HIDDEN, false);
    fgSetTopmost(self->fgTooltip);
  }
}
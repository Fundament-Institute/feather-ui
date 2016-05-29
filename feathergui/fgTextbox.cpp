// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTextbox.h"
#include "feathercpp.h"

void FG_FASTCALL fgTextbox_Init(fgTextbox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  memset(&self->text, 0, sizeof(fgVectorString));
  memset(&self->buf, 0, sizeof(fgVectorUTF32));
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (FN_DESTROY)&fgScrollbar_Destroy, (FN_MESSAGE)&fgScrollbar_Message);
}
void FG_FASTCALL fgTextbox_Destroy(fgTextbox* self)
{
  fgScrollbar_Destroy(&self->window);
}

inline size_t FG_FASTCALL fgTextbox_DeleteSelection(fgTextbox* self)
{
  if(self->start == self->end)
    return 0;
  assert(self->start < self->end);

  memmove(self->text.p + self->start, self->text.p + self->end, (self->text.l - self->end)*sizeof(int));
  self->text.l -= self->end - self->start;
  self->buf.l = 0;
  self->end = self->start;
  return FG_ACCEPT;
}
inline void FG_FASTCALL fgTextbox_Insert(fgTextbox* self, size_t start, const int* s, size_t len)
{
  ((bss_util::cDynArray<int>*)&self->text)->Reserve(self->text.l + len);
  memmove(self->text.p + start + len, self->text.p + start, len * sizeof(int));
  MEMCPY(self->text.p + start, self->text.s - start, s, len);
  self->buf.l = 0;
  self->start += len;
  self->end = self->start;
}

inline void FG_FASTCALL fgTextbox_MoveCursor(fgTextbox* self, int num, bool select, bool word)
{
  // TODO set num to word length if word is true
  if(num < 0)
  {
    self->start += num;
    if(!select) self->end = self->start;
  }
  if(num > 0)
  {
    self->end += num;
    if(!select) self->start = self->end;
  }
}

FG_EXTERN void FG_FASTCALL fgTextbox_Recalc(fgTextbox* self)
{
  if(self->font && (self->window.control.element.flags&FGELEMENT_EXPAND))
  {
    AbsRect area;
    ResolveRect(*self, &area);
    fgFontSize(self->font, !self->text.p ? self->placeholder : self->text.p, self->lineheight, self->letterspacing, &area, self->window.control.element.flags);
    CRect adjust = self->window.control.element.transform.area;
    if(self->window.control.element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left;
    if(self->window.control.element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}

AbsVec FG_FASTCALL fgTextbox_RelativeMouse(fgTextbox* self, const FG_Msg* msg)
{
  AbsRect r;
  ResolveRect(*self, &r);
  return AbsVec { msg->x - r.left, msg->y - r.top };
}

size_t FG_FASTCALL fgTextbox_Message(fgTextbox* self, const FG_Msg* msg)
{
  static const float CURSOR_BLINK = 0.8;
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgScrollbar_Message(&self->window, msg);
    self->validation = 0;
    self->mask = 0;
    self->placeholder = &UNICODE_TERMINATOR;
    self->selector.color = ~0;
    self->placecolor.color = ~0;
    self->start = 0;
    memset(&self->startpos, 0, sizeof(AbsVec));
    self->end = 0;
    memset(&self->endpos, 0, sizeof(AbsVec));
    self->color.color = 0;
    self->font = 0;
    self->lineheight = 0;
    self->letterspacing = 0;
    return FG_ACCEPT;
  case FG_KEYCHAR:
    if(self->validation)
    {
      return 0;
    }
    if(self->inserting && self->start == self->end && self->start < self->text.l && self->text.p[self->start] != 0)
      fgTextbox_MoveCursor(self, 1, true, false);
    
    fgTextbox_DeleteSelection(self);
    fgTextbox_Insert(self, self->start, &msg->keychar, 1);
    return FG_ACCEPT;
  case FG_KEYDOWN:
    switch(msg->keycode)
    {
    case FG_KEY_RIGHT:
    case FG_KEY_LEFT:
      fgTextbox_MoveCursor(self, msg->keycode == FG_KEY_RIGHT ? 1 : -1, msg->IsShiftDown(), msg->IsCtrlDown());
      return FG_ACCEPT;
    case FG_KEY_DOWN:
    case FG_KEY_UP:
      return FG_ACCEPT;
    case FG_KEY_HOME:
      return _sendsubmsg<FG_ACTION, ptrdiff_t>(*self, msg->IsCtrlDown() ? FGTEXTBOX_GOTOSTART : FGTEXTBOX_GOTOLINESTART, msg->IsShiftDown());
    case FG_KEY_END:
      return _sendsubmsg<FG_ACTION, ptrdiff_t>(*self, msg->IsCtrlDown() ? FGTEXTBOX_GOTOEND : FGTEXTBOX_GOTOLINEEND, msg->IsShiftDown());
    case FG_KEY_DELETE:
      if(self->end == self->start)
        fgTextbox_MoveCursor(self, 1, true, msg->IsCtrlDown());
      return fgTextbox_DeleteSelection(self);
    case FG_KEY_BACK: // backspace
      if(self->end == self->start)
        fgTextbox_MoveCursor(self, -1, true, msg->IsCtrlDown());
      return fgTextbox_DeleteSelection(self);
    case FG_KEY_A:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_SELECTALL);
    case FG_KEY_X:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_CUT);
    case FG_KEY_C:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_COPY);
    case FG_KEY_V:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_PASTE);
      break;
    case FG_KEY_INSERT:
      return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_TOGGLEINSERT);
    }
  case FG_ACTION:
    switch(msg->subtype)
    {
      case FGTEXTBOX_SELECTALL:
        self->start = 0;
        self->end = self->text.l;
        return FG_ACCEPT;
      case FGTEXTBOX_CUT:
        fgClipboardCopy(FGCLIPBOARD_TEXT, self->text.p, self->text.l*sizeof(int));
        fgTextbox_DeleteSelection(self);
        return FG_ACCEPT;
      case FGTEXTBOX_COPY:
        fgClipboardCopy(FGCLIPBOARD_TEXT, self->text.p, self->text.l * sizeof(int));
        return FG_ACCEPT;
      case FGTEXTBOX_PASTE:
        if(fgClipboardExists(FGCLIPBOARD_TEXT))
        {
          fgTextbox_DeleteSelection(self);
          size_t sz = 0;
          const int* paste = (const int*)fgClipboardPaste(FGCLIPBOARD_TEXT, &sz);
          fgTextbox_Insert(self, self->start, paste, sz); // TODO: do verification here
          fgClipboardFree(paste);
          return FG_ACCEPT;
        }
        return 0;
      case FGTEXTBOX_GOTOSTART:
        self->start = 0;
        if(!msg->otherint) self->end = 0;
        return FG_ACCEPT;
      case FGTEXTBOX_GOTOEND:
        self->start = self->text.l;
        if(!msg->otherint) self->end = self->text.l;
        return FG_ACCEPT;
      case FGTEXTBOX_GOTOLINESTART:
        while(self->start > 0 && self->text.p[self->start-1] != '\n' && self->text.p[self->start-1] != '\r') --self->start;
        if(!msg->otherint) self->end = self->text.l;
        return FG_ACCEPT;
      case FGTEXTBOX_GOTOLINEEND:
        while(self->start < self->text.l && self->text.p[self->start] != '\n' && self->text.p[self->start] != '\r') ++self->start;
        if(!msg->otherint) self->end = self->text.l;
        return FG_ACCEPT;
      case FGTEXTBOX_TOGGLEINSERT:
        self->inserting = !self->inserting;
        return FG_ACCEPT;
    }
    return 0;
  case FG_SETTEXT:
    ((bss_util::cDynArray<int>*)&self->text)->Clear();
    ((bss_util::cDynArray<char>*)&self->buf)->Clear();
    if(msg->other)
    {
      size_t len = UTF8toUTF32((const char*)msg->other, -1, 0, 0);
      ((bss_util::cDynArray<int>*)&self->text)->Reserve(len);
      self->text.l = UTF8toUTF32((const char*)msg->other, -1, self->text.p, self->text.s);
    }
    fgTextbox_Recalc(self);
    fgDirtyElement(&self->window.control.element.transform);
    return FG_ACCEPT;
  case FG_SETFONT:
    if(self->font) fgDestroyFont(self->font);
    self->font = 0;
    if(msg->other)
    {
      size_t dpi = _sendmsg<FG_GETDPI>(*self);
      unsigned int fontdpi;
      unsigned int fontsize;
      fgFontGet(msg->other, 0, &fontsize, &fontdpi);
      self->font = (dpi == fontdpi) ? fgCloneFont(msg->other) : fgCopyFont(msg->other, fontsize, fontdpi);
    }
    fgTextbox_Recalc(self);
    fgDirtyElement(&self->window.control.element.transform);
    break;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->otherf;
    fgTextbox_Recalc(self);
    fgDirtyElement(&self->window.control.element.transform);
    break;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->otherf;
    fgTextbox_Recalc(self);
    fgDirtyElement(&self->window.control.element.transform);
    break;
  case FG_SETCOLOR:
    if(!msg->otheraux)
      self->color.color = (unsigned int)msg->otherint;
    else
      self->placecolor.color = (unsigned int)msg->otherint;
    fgDirtyElement(&self->window.control.element.transform);
    break;
  case FG_GETTEXT:
    if(!msg->otherint)
    {
      if(!self->buf.l)
      {
        size_t len = UTF32toUTF8(self->text.p, -1, 0, 0);
        ((bss_util::cDynArray<int>*)&self->buf)->Reserve(len);
        self->buf.l = UTF32toUTF8(self->text.p, -1, self->buf.p, self->buf.s);
      }
      return reinterpret_cast<size_t>(self->buf.p);
    }
    return reinterpret_cast<size_t>(self->placeholder);
  case FG_GETFONT:
    return reinterpret_cast<size_t>(self->font);
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return msg->otherint ? self->placecolor.color : self->color.color;
  case FG_MOVE:
    if(!(msg->otheraux & FGMOVE_PROPAGATE) && (msg->otheraux & FGMOVE_RESIZE))
      fgTextbox_Recalc(self);
    break;
  case FG_DRAW:
    if(self->font != 0 && !(msg->subtype & 1))
    {
      // TODO Draw selection rectangles first

      // Draw text
      AbsRect area = *(AbsRect*)msg->other;
      float scale = (!msg->otheraux || !fgroot_instance->dpi) ? 1.0 : (fgroot_instance->dpi / (float)msg->otheraux);
      area.left *= scale;
      area.top *= scale;
      area.right *= scale;
      area.bottom *= scale;
      AbsVec center = ResolveVec(&self->window.control.element.transform.center, &area);
      int* text = self->text.p;
      if(self->mask)
      {
        text = (int*)ALLOCA(self->text.l + 1);
        for(size_t i = 0; i < self->text.l; ++i) text[i] = self->mask;
        text[self->text.l] = 0;
      }

      self->cache = fgDrawFont(self->font,
        !self->text.l ? self->placeholder : text,
        self->lineheight,
        self->letterspacing,
        !self->text.l ? self->placecolor.color : self->color.color,
        &area,
        self->window.control.element.transform.rotation,
        &center,
        self->window.control.element.flags,
        self->cache);

      // Draw cursor
      AbsVec linetop = self->endpos;
      linetop.x += area.left;
      linetop.y += area.top;
      AbsVec linebottom = linetop;
      linebottom.y += self->lineheight;
      fgDrawLine(linetop, linebottom, 0xFF000000); // TODO: Make this blink. This requires ensuring that FG_DRAW is called at least during the blink interval.
    }
    break;
  case FG_MOUSEDOWN:
  {
    self->start = fgFontIndex(self->font, self->text.p, self->lineheight, self->letterspacing, fgTextbox_RelativeMouse(self, msg), self->start, &self->startpos);
    if(!fgroot_instance->GetKey(FG_KEY_SHIFT))
    {
      self->end = self->start;
      self->endpos = self->startpos;
    }
    else if(self->end < self->start)
    {
      bss_util::rswap(self->start, self->end);
      bss_util::rswap(self->startpos, self->endpos);
    }
  }
    break;
  case FG_MOUSEMOVE:
    if(msg->allbtn&FG_MOUSELBUTTON)
    {
      self->start = fgFontIndex(self->font, self->text.p, self->lineheight, self->letterspacing, fgTextbox_RelativeMouse(self, msg), self->start, &self->startpos);
      if(self->end < self->start)
      {
        bss_util::rswap(self->start, self->end);
        bss_util::rswap(self->startpos, self->endpos);
      }
    }
    break;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgTextbox";
  }

  return fgScrollbar_Message(&self->window, msg);
}
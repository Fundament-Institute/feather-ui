// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTextbox.h"
#include "feathercpp.h"
#include "fgCurve.h"

void fgTextbox_Init(fgTextbox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgTextbox_Destroy, (fgMessage)&fgTextbox_Message);
}
void fgTextbox_Destroy(fgTextbox* self)
{
  if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, self->layout);
  if(self->font != 0) fgroot_instance->backend.fgDestroyFont(self->font);
  ((bss_util::cDynArray<int>*)&self->text32)->~cDynArray();
  ((bss_util::cDynArray<wchar_t>*)&self->text16)->~cDynArray();
  ((bss_util::cDynArray<char>*)&self->text8)->~cDynArray();
  ((bss_util::cDynArray<int>*)&self->placeholder32)->~cDynArray();
  ((bss_util::cDynArray<wchar_t>*)&self->placeholder16)->~cDynArray();
  ((bss_util::cDynArray<char>*)&self->placeholder8)->~cDynArray();

  self->scroll->message = (fgMessage)fgScrollbar_Message;
  fgScrollbar_Destroy(&self->scroll);
}

inline void fgTextbox_SetCursorEnd(fgTextbox* self)
{
  self->end = self->start;
  self->endpos = self->startpos;
}

inline size_t fgTextbox_DeleteSelection(fgTextbox* self)
{
  if(self->start == self->end)
    return 0;
  assert(self->text32.p != 0);
  if(self->start > self->end)
  {
    bss_util::rswap(self->start, self->end);
    bss_util::rswap(self->startpos, self->endpos);
  }

  bss_util::RemoveRangeSimple<int>(self->text32.p, self->text32.l, self->start, self->end - self->start);
  self->text32.l -= self->end - self->start;
  self->text32.p[self->text32.l] = 0;
  self->text16.l = 0;
  self->text8.l = 0;
  fgTextbox_SetCursorEnd(self);
  fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
  return FG_ACCEPT;
}

#define FILLMASK(self, text) \
switch(fgroot_instance->backend.BackendTextFormat)\
{\
case FGTEXTFMT_UTF8:\
  text = ALLOCA((self->text8.l + 1) * sizeof(char));\
  for(size_t i = 0; i < self->text8.l; ++i) ((char*)text)[i] = self->mask;\
  ((char*)text)[self->text8.l] = 0;\
  break;\
case FGTEXTFMT_UTF16:\
  text = ALLOCA((self->text16.l + 1) * sizeof(wchar_t));\
  for(size_t i = 0; i < self->text16.l; ++i) ((wchar_t*)text)[i] = self->mask;\
  ((wchar_t*)text)[self->text16.l] = 0;\
  break;\
case FGTEXTFMT_UTF32:\
  text = ALLOCA((self->text32.l + 1) * sizeof(int));\
  for(size_t i = 0; i < self->text32.l; ++i) ((int*)text)[i] = self->mask;\
  ((int*)text)[self->text32.l] = 0;\
  break;\
}

inline void fgTextbox_fixpos(fgTextbox* self, size_t cursor, AbsVec* r)
{
  fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
  if(!v) return;
  void* text = v->p;
  if(self->mask)
    FILLMASK(self, text)

  *r = fgroot_instance->backend.fgFontPos(self->font, text, v->l, self->lineheight, self->letterspacing, &self->areacache, self->scroll->flags, cursor, self->layout);
  AbsRect to = { r->x, r->y, r->x, r->y + self->lineheight*1.125f }; // We don't know what the descender is, so we estimate it as 1/8 the lineheight.
  _sendsubmsg<FG_ACTION, void*>(*self, FGSCROLLBAR_SCROLLTO, &to);
  self->lastx = self->startpos.x;
}
inline size_t fgTextbox_fixindex(fgTextbox* self, AbsVec pos, AbsVec* cursor)
{
  fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
  if(!v) return 0;
  void* text = v->p;
  if(self->mask)
    FILLMASK(self, text)
  size_t r = fgroot_instance->backend.fgFontIndex(self->font, text, v->l, self->lineheight, self->letterspacing, &self->areacache, self->scroll->flags, pos, cursor, self->layout);
  AbsRect to = { cursor->x, cursor->y, cursor->x, cursor->y + self->lineheight*1.125f };
  _sendsubmsg<FG_ACTION, void*>(*self, FGSCROLLBAR_SCROLLTO, &to);
  return r;
}
inline void fgTextbox_Insert(fgTextbox* self, size_t start, const int* s, size_t len)
{
  if(!s[len - 1]) --len; // We cannot insert a null pointer in the middle of our text, so remove it if it exists.
  ((bss_util::cDynArray<int>*)&self->text32)->Reserve(self->text32.l + len + 1);
  assert(self->text32.p != 0);
  bss_util::InsertRangeSimple<int>(self->text32.p, self->text32.l, start, s, len);
  self->text32.l += len;
  self->text32.p[self->text32.l] = 0;
  self->text16.l = 0;
  self->text8.l = 0;
  fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE|FGMOVE_RESIZE);
  self->start += len;
  fgTextbox_fixpos(self, self->start, &self->startpos);
  fgTextbox_SetCursorEnd(self);
}
inline bool fgTextbox_checkspace(fgTextbox* self, ptrdiff_t num, bool space)
{
  if(self->mask != 0) return false;
  int c = self->text32.p[self->start + num];
  return ((!!isspace(c)) == space || c == '\n' || c == '\r');
}
inline void fgTextbox_MoveCursor(fgTextbox* self, int num, bool select, bool word)
{
  if(!self->text32.p) return;
  size_t laststart = self->start;
  if(word) // If we are looking for words, increment num until we hit a whitespace character. If it isn't a newline, include that space in the selection.
  {
    if(num < 0 && self->start > 0)
    {
      while(num + self->start > 0 && !fgTextbox_checkspace(self, num, false)) --num;
      while(num + self->start > 0 && !fgTextbox_checkspace(self, num, true)) --num;
      if(num + self->start > 0) ++num;
    }
    else if(num > 0)
    {
      while(num + self->start < self->text32.l && !fgTextbox_checkspace(self, num, true)) ++num;
      while(num + self->start < self->text32.l && !fgTextbox_checkspace(self, num, false)) ++num;
    }
  }
  if(((int)self->start) + num < 0)
    self->start = 0;
  else if(self->start + num > self->text32.l)
    self->start = self->text32.l;
  else
    self->start += num;
  fgTextbox_fixpos(self, self->start, &self->startpos);
  if(!select)
    fgTextbox_SetCursorEnd(self);
}

AbsVec fgTextbox_RelativeMouse(fgTextbox* self, const FG_Msg* msg)
{
  AbsRect r;
  ResolveRect(*self, &r);
  return AbsVec { msg->x - r.left - self->scroll->padding.left + self->scroll.realpadding.left, msg->y - r.top - self->scroll->padding.top + self->scroll.realpadding.top };
}

size_t fgTextbox_Message(fgTextbox* self, const FG_Msg* msg)
{
  static const float CURSOR_BLINK = 0.8f;
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    memset(&self->text8, 0, sizeof(fgVectorUTF8));
    memset(&self->text16, 0, sizeof(fgVectorUTF16));
    memset(&self->text32, 0, sizeof(fgVectorUTF32));
    memset(&self->placeholder8, 0, sizeof(fgVectorUTF8));
    memset(&self->placeholder16, 0, sizeof(fgVectorUTF16));
    memset(&self->placeholder32, 0, sizeof(fgVectorUTF32));
    self->validation = 0;
    self->formatting = 0;
    self->mask = 0;
    self->selector.color = ~0;
    self->placecolor.color = ~0;
    self->cursorcolor.color = 0xFF000000;
    self->start = 0;
    memset(&self->startpos, 0, sizeof(AbsVec));
    self->end = 0;
    memset(&self->endpos, 0, sizeof(AbsVec));
    self->color.color = 0;
    self->font = 0;
    self->lineheight = 0; // lineheight must be zero'd before a potential transform unit resolution.
    self->letterspacing = 0;
    self->lastx = 0;
    self->inserting = 0;
    memset(&self->areacache, 0, sizeof(AbsRect));
    fgScrollbar_Message(&self->scroll, msg);
    self->lastclick = fgroot_instance->time;
    self->layout = 0;
    return FG_ACCEPT;
  case FG_KEYCHAR:
    if(self->validation)
    { 
      // TODO: do validation here
      return 0;
    }
    if(self->inserting && self->start == self->end && self->start < self->text32.l && self->text32.p[self->start] != 0)
      fgTextbox_MoveCursor(self, 1, true, false);
    
    fgTextbox_DeleteSelection(self);
    fgTextbox_Insert(self, self->start, &msg->keychar, 1);
    self->lastclick = fgroot_instance->time;
    return FG_ACCEPT;
  case FG_KEYDOWN:
    switch(msg->keycode)
    {
    case FG_KEY_RIGHT:
    case FG_KEY_LEFT:
      fgTextbox_MoveCursor(self, msg->keycode == FG_KEY_RIGHT ? 1 : -1, msg->IsShiftDown(), msg->IsCtrlDown());
      self->lastclick = fgroot_instance->time;
      return FG_ACCEPT;
    case FG_KEY_DOWN:
    case FG_KEY_UP:
    {
      AbsVec pos { self->lastx, self->startpos.y + ((msg->keycode == FG_KEY_DOWN) ? self->lineheight*1.5f : -self->lineheight*0.5f) };
      self->start = fgTextbox_fixindex(self, pos, &self->startpos);
      if(!msg->IsShiftDown())
        fgTextbox_SetCursorEnd(self);
      self->lastclick = fgroot_instance->time;
    }
      return FG_ACCEPT;
    case FG_KEY_HOME:
      return _sendsubmsg<FG_ACTION, ptrdiff_t>(*self, msg->IsCtrlDown() ? FGTEXTBOX_GOTOSTART : FGTEXTBOX_GOTOLINESTART, msg->IsShiftDown());
    case FG_KEY_END:
      return _sendsubmsg<FG_ACTION, ptrdiff_t>(*self, msg->IsCtrlDown() ? FGTEXTBOX_GOTOEND : FGTEXTBOX_GOTOLINEEND, msg->IsShiftDown());
    case FG_KEY_DELETE:
      if(self->end == self->start)
        fgTextbox_MoveCursor(self, 1, true, msg->IsCtrlDown());
      fgTextbox_DeleteSelection(self);
      self->lastclick = fgroot_instance->time;
      return FG_ACCEPT;
    case FG_KEY_BACK: // backspace
      if(self->end == self->start)
        fgTextbox_MoveCursor(self, -1, true, msg->IsCtrlDown());
      fgTextbox_DeleteSelection(self);
      self->lastclick = fgroot_instance->time;
      return FG_ACCEPT;
    case FG_KEY_A:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_SELECTALL);
      break;
    case FG_KEY_X:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_CUT);
      break;
    case FG_KEY_C:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_COPY);
      break;
    case FG_KEY_V:
      if(msg->IsCtrlDown())
        return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_PASTE);
      break;
    case FG_KEY_PAGEUP:
    case FG_KEY_PAGEDOWN:
    {
      float height = self->areacache.bottom - self->areacache.top;
      height = floor((height*0.9f) / self->lineheight)*self->lineheight;
      AbsVec pos { self->lastx, self->startpos.y + ((msg->keycode == FG_KEY_PAGEDOWN) ? (height + self->lineheight*1.5f) : (-height - self->lineheight*0.5f)) };
      self->start = fgTextbox_fixindex(self, pos, &self->startpos);
      if(!msg->IsShiftDown())
        fgTextbox_SetCursorEnd(self);
      self->lastclick = fgroot_instance->time;
    }
      return FG_ACCEPT;
    case FG_KEY_INSERT:
      return _sendsubmsg<FG_ACTION>(*self, FGTEXTBOX_TOGGLEINSERT);
    case FG_KEY_RETURN:
      if(!msg->IsShiftDown() && (self->scroll->flags&FGTEXTBOX_ACTION))
        _sendsubmsg<FG_ACTION>(*self, 0);
      else if(!(self->scroll->flags&FGTEXTBOX_SINGLELINE))
        self->scroll->KeyChar('\n', msg->sigkeys); // windows translates this to \r, not \n, so we have to do it ourselves.
      return FG_ACCEPT;
    }
    return 0;
  case FG_ACTION:
    switch(msg->subtype)
    {
      case FGTEXTBOX_SELECTALL:
        self->start = 0;
        fgTextbox_fixpos(self, self->start, &self->startpos);
        self->end = self->text32.l;
        fgTextbox_fixpos(self, self->end, &self->endpos);
        break;
      case FGTEXTBOX_CUT:
        fgroot_instance->backend.fgClipboardCopy(FGCLIPBOARD_TEXT, self->text32.p, self->text32.l*sizeof(int));
        fgTextbox_DeleteSelection(self);
        break;
      case FGTEXTBOX_COPY:
        fgroot_instance->backend.fgClipboardCopy(FGCLIPBOARD_TEXT, self->text32.p, self->text32.l * sizeof(int));
        break;
      case FGTEXTBOX_PASTE:
        if(fgroot_instance->backend.fgClipboardExists(FGCLIPBOARD_TEXT))
        {
          fgTextbox_DeleteSelection(self);
          size_t sz = 0;
          const int* paste = (const int*)fgroot_instance->backend.fgClipboardPaste(FGCLIPBOARD_TEXT, &sz);
          fgTextbox_Insert(self, self->start, paste, sz); // TODO: do verification here
          fgroot_instance->backend.fgClipboardFree(paste);
          break;
        }
        return 0;
      case FGTEXTBOX_GOTOSTART:
        self->start = 0;
        fgTextbox_fixpos(self, self->start, &self->startpos);
        if(!msg->i)
          fgTextbox_SetCursorEnd(self);
        break;
      case FGTEXTBOX_GOTOEND:
        self->start = self->text32.l;
        fgTextbox_fixpos(self, self->start, &self->startpos);
        if(!msg->i)
          fgTextbox_SetCursorEnd(self);
        break;
      case FGTEXTBOX_GOTOLINESTART:
        while(self->start > 0 && self->text32.p[self->start-1] != '\n' && self->text32.p[self->start-1] != '\r') --self->start;
        fgTextbox_fixpos(self, self->start, &self->startpos);
        if(!msg->i)
          fgTextbox_SetCursorEnd(self);
        break;
      case FGTEXTBOX_GOTOLINEEND:
        while(self->start < self->text32.l && self->text32.p[self->start] != '\n' && self->text32.p[self->start] != '\r') ++self->start;
        fgTextbox_fixpos(self, self->start, &self->startpos);
        if(!msg->i)
          fgTextbox_SetCursorEnd(self);
        break;
      case FGTEXTBOX_TOGGLEINSERT:
        self->inserting = !self->inserting;
        break;
      default:
        return fgScrollbar_Message(&self->scroll, msg);
    }
    self->lastclick = fgroot_instance->time;
    return FG_ACCEPT;
  case FG_SETTEXT:
    if(msg->subtype <= FGTEXTFMT_UTF32)
    {
      ((bss_util::cDynArray<int>*)&self->text32)->Clear();
      ((bss_util::cDynArray<wchar_t>*)&self->text16)->Clear();
      ((bss_util::cDynArray<char>*)&self->text8)->Clear();
    }
    else if(msg->subtype <= FGTEXTFMT_PLACEHOLDER_UTF32)
    {
      ((bss_util::cDynArray<int>*)&self->placeholder32)->Clear();
      ((bss_util::cDynArray<wchar_t>*)&self->placeholder16)->Clear();
      ((bss_util::cDynArray<char>*)&self->placeholder8)->Clear();
    }
    if(msg->p)
    {
      switch(msg->subtype)
      {
      case FGTEXTFMT_UTF8:
        ((bss_util::cDynArray<char>*)&self->text8)->operator=(bss_util::cArraySlice<const char>((const char*)msg->p, !msg->u2 ? (strlen((const char*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_UTF16:
        ((bss_util::cDynArray<wchar_t>*)&self->text16)->operator=(bss_util::cArraySlice<const wchar_t>((const wchar_t*)msg->p, !msg->u2 ? (wcslen((const wchar_t*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_UTF32:
      {
        int* txt = (int*)msg->p;
        size_t len = msg->u2;
        if(!len)
          while(txt[len++] != 0);
        ((bss_util::cDynArray<int>*)&self->text32)->operator=(bss_util::cArraySlice<const int>((const int*)msg->p, len));
      }
      break;
      case FGTEXTFMT_PLACEHOLDER_UTF8:
        ((bss_util::cDynArray<char>*)&self->placeholder8)->operator=(bss_util::cArraySlice<const char>((const char*)msg->p, !msg->u2 ? (strlen((const char*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_PLACEHOLDER_UTF16:
        ((bss_util::cDynArray<wchar_t>*)&self->placeholder16)->operator=(bss_util::cArraySlice<const wchar_t>((const wchar_t*)msg->p, !msg->u2 ? (wcslen((const wchar_t*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_PLACEHOLDER_UTF32:
      {
        int* txt = (int*)msg->p;
        size_t len = msg->u2;
        if(!len)
          while(txt[len++] != 0);
        ((bss_util::cDynArray<int>*)&self->placeholder32)->operator=(bss_util::cArraySlice<const int>((const int*)msg->p, len));
      }
      break;
      case FGTEXTFMT_MASK:
        self->mask = msg->i;
        break;
      }
    }
    fgText_Conversion(FGTEXTFMT_UTF32, &self->text8, &self->text16, &self->text32); // the textbox requires a UTF32 format be available at all times
    fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETFONT:
    if(self->font) fgroot_instance->backend.fgDestroyFont(self->font);
    self->font = 0;
    if(msg->p)
    {
      fgFontDesc desc;
      fgroot_instance->backend.fgFontGet(msg->p, &desc);
      fgIntVec dpi = self->scroll->GetDPI();
      bool identical = (dpi.x == desc.dpi.x && dpi.y == desc.dpi.y);
      desc.dpi = dpi;
      self->font = fgroot_instance->backend.fgCloneFont(msg->p, identical ? 0 : &desc);
    }
    fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->f;
    fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->f;
    fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_SETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_MAIN: self->color.color = (unsigned int)msg->i; break;
    case FGSETCOLOR_PLACEHOLDER: self->placecolor.color = (unsigned int)msg->i; break;
    case FGSETCOLOR_CURSOR: self->cursorcolor.color = (unsigned int)msg->i; break;
    case FGSETCOLOR_SELECT: self->selector.color = (unsigned int)msg->i; break;
    }
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_GETTEXT:
    if(msg->subtype <= FGTEXTFMT_UTF32)
    {
      fgVector* v = fgText_Conversion(msg->subtype, &self->text8, &self->text16, &self->text32);
      return !v ? 0 : reinterpret_cast<size_t>(v->p);
    }
    else if(msg->subtype <= FGTEXTFMT_PLACEHOLDER_UTF32)
    {
      fgVector* v = fgText_Conversion(msg->subtype - FGTEXTFMT_PLACEHOLDER_UTF8, &self->placeholder8, &self->placeholder16, &self->placeholder32);
      return !v ? 0 : reinterpret_cast<size_t>(v->p);
    }
    return 0;
  case FG_GETFONT:
    return reinterpret_cast<size_t>(self->font);
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    switch(msg->u2)
    {
    default:
    case 0: return self->color.color;
    case 1: return self->placecolor.color;
    case 2: return self->cursorcolor.color;
    case 3: return self->selector.color;
    }
    assert(false);
  case FG_MOVE:
    if(!(msg->u2 & FGMOVE_PROPAGATE) && (msg->u2 & (FGMOVE_RESIZE | FGMOVE_PADDING | FGMOVE_MARGIN)))
      fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, self, FGMOVE_PROPAGATE | FGMOVE_RESIZE);
    break;
  case FG_DRAW:
    fgScrollbar_Message(&self->scroll, msg); // Render everything else first
    
    if(self->font != 0 && !(msg->subtype & 1))
    {
      // Draw selection rectangles
      AbsRect area = *(AbsRect*)msg->p;
      fgDrawAuxData* data = (fgDrawAuxData*)msg->p2;

      AbsRect cliparea = area;
      cliparea.left += self->scroll.realpadding.left;
      cliparea.top += self->scroll.realpadding.top;
      cliparea.right -= self->scroll.realpadding.right + bssmax(self->scroll.barcache.y, 0);
      cliparea.bottom -= self->scroll.realpadding.bottom + bssmax(self->scroll.barcache.x, 0);
      if(!(self->scroll->flags&FGELEMENT_NOCLIP))
        fgroot_instance->backend.fgPushClipRect(&cliparea, data);

      GetInnerRect(*self, &area, &area);
      AbsVec begin;
      AbsVec end;
      if(self->start < self->end)
      {
        begin = self->startpos;
        end = self->endpos;
      }
      else {
        begin = self->endpos;
        end = self->startpos;
      }


      CRect uv = CRect{ 0,0,0,0,0,0,0,0 };
      AbsVec center = AbsVec{ 0,0 };
      if(begin.y == end.y)
      {
        AbsRect srect = { area.left + begin.x, area.top + begin.y, area.left + end.x, area.top + begin.y + self->lineheight };
        fgroot_instance->backend.fgDrawAsset(0, &uv, self->selector.color, 0, 0, &srect, 0, &center, FGRESOURCE_RECT, data);
      }
      else
      {
        AbsRect srect = AbsRect{ area.left + begin.x, area.top + begin.y, area.right, area.top + begin.y + self->lineheight };
        fgroot_instance->backend.fgDrawAsset(0, &uv, self->selector.color, 0, 0, &srect, 0, &center, FGRESOURCE_RECT, data);
        if(begin.y + self->lineheight + 0.5 < end.y)
        {
          srect = AbsRect{ area.left, area.top + begin.y + self->lineheight, area.right, area.top + end.y };
          fgroot_instance->backend.fgDrawAsset(0, &uv, self->selector.color, 0, 0, &srect, 0, &center, FGRESOURCE_RECT, data);
        }
        srect = AbsRect{ area.left, area.top + end.y, area.left + end.x, area.top + end.y + self->lineheight };
        fgroot_instance->backend.fgDrawAsset(0, &uv, self->selector.color, 0, 0, &srect, 0, &center, FGRESOURCE_RECT, data);
      }

      // Draw text
      AbsVec scale = { (!data->dpi.x || !fgroot_instance->dpi.x) ? 1.0f : (fgroot_instance->dpi.x / (float)data->dpi.x), (!data->dpi.y || !fgroot_instance->dpi.y) ? 1.0f : (fgroot_instance->dpi.y / (float)data->dpi.y) };
      area.left *= scale.x;
      area.top *= scale.y;
      area.right *= scale.x;
      area.bottom *= scale.y;
      //assert(self->areacache.right - self->areacache.left == area.right - area.left);
      //assert(self->areacache.bottom - self->areacache.top == area.bottom - area.top);
      fgSnapAbsRect(area, self->scroll->flags);
      center = ResolveVec(&self->scroll.control.element.transform.center, &area);

      void* text = 0;
      size_t len = 0;
      if(!self->text32.l)
      {
        fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->placeholder8, &self->placeholder16, &self->placeholder32);
        text = v->p;
        len = v->l;
      }
      else
      {
        fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
        text = v->p;
        len = v->l;
        if(self->mask)
          FILLMASK(self, text)
      }

      fgroot_instance->backend.fgDrawFont(self->font,
        text,
        len,
        self->lineheight,
        self->letterspacing,
        !self->text32.l ? self->placecolor.color : self->color.color,
        &area,
        self->scroll.control.element.transform.rotation,
        &center,
        self->scroll.control.element.flags,
        data,
        (self->mask != 0 || !self->text32.l) ? 0 : self->layout);

      // Draw cursor
      if(fgFocusedWindow == *self && bss_util::bssfmod(fgroot_instance->time - self->lastclick, fgroot_instance->cursorblink * 2) < fgroot_instance->cursorblink)
      {
        AbsVec snappos = { roundf(self->startpos.x), roundf(self->startpos.y) };
        AbsVec lines[2] = { snappos, { snappos.x, snappos.y + self->lineheight } };
        AbsVec scale = { 1.0f, 1.0f };
        fgroot_instance->backend.fgDrawLines(lines, 2, self->cursorcolor.color, &area.topleft, &scale, self->scroll.control.element.transform.rotation, &center, data); // TODO: This requires ensuring that FG_DRAW is called at least during the blink interval.
      }

      if(!(self->scroll->flags&FGELEMENT_NOCLIP))
        fgroot_instance->backend.fgPopClipRect(data);
    }
    return FG_ACCEPT;
  case FG_MOUSEDBLCLICK:
    if(self->text32.l > 0 && msg->button == FG_MOUSELBUTTON && !fgroot_instance->GetKey(FG_KEY_SHIFT) && !fgroot_instance->GetKey(FG_KEY_CONTROL))
    {
      self->end = fgTextbox_fixindex(self, fgTextbox_RelativeMouse(self, msg), &self->endpos);
      self->start = 0;
      if(isspace(self->text32.p[self->end]) && self->end > 0) --self->end;
      while(self->end > 0 && !fgTextbox_checkspace(self, self->end, true)) --self->end;
      if(self->end > 0) ++self->end;
      self->start = self->end;
      while(self->start < self->text32.l && !fgTextbox_checkspace(self, 0, true)) ++self->start; // note: checkspace works off of self->start

      fgTextbox_fixpos(self, self->start, &self->startpos);
      fgTextbox_fixpos(self, self->end, &self->endpos);
      self->lastclick = fgroot_instance->time;
      self->lastx = self->startpos.x;
      return FG_ACCEPT;
    }
    break;
  case FG_MOUSEDOWN:
  {
    self->start = fgTextbox_fixindex(self, fgTextbox_RelativeMouse(self, msg), &self->startpos);
    if(!fgroot_instance->GetKey(FG_KEY_SHIFT))
      fgTextbox_SetCursorEnd(self);
    self->lastclick = fgroot_instance->time;
    self->lastx = self->startpos.x;
  }
    break;
  case FG_MOUSEMOVE:
    if(msg->allbtn&FG_MOUSELBUTTON)
    {
      self->start = fgTextbox_fixindex(self, fgTextbox_RelativeMouse(self, msg), &self->startpos);
      self->lastx = self->startpos.x;
    }
    fgScrollbar_Message(&self->scroll, msg);
    return FGCURSOR_IBEAM;
  case FG_MOUSEON:
    return FG_ACCEPT;
  case FG_MOUSEOFF:
    return FG_ACCEPT;
  case FG_LAYOUTFUNCTION:
    if(msg->p != 0 && self->font != 0)
    {
      FG_Msg* m = (FG_Msg*)msg->p;
      AbsVec* dim = (AbsVec*)msg->p2;
      if(m->subtype == FGELEMENT_LAYOUTMOVE)
      {
        ResolveInnerRect(*self, &self->areacache);
        fgIntVec dpi = self->scroll->GetDPI(); // GetDPI can return 0 if we have no parent, which can happen when a layout is being set up or destroyed.
        AbsVec scale = { !dpi.x ? (FABS)1.0 : (fgroot_instance->dpi.x / (FABS)dpi.x), !dpi.y ? (FABS)1.0 : (fgroot_instance->dpi.y / (FABS)dpi.y) };
        assert(isfinite(scale.x) && isfinite(scale.y));
        self->areacache.left *= scale.x;
        self->areacache.top *= scale.y;
        self->areacache.right *= scale.x;
        self->areacache.bottom *= scale.y;
        AbsRect r = self->areacache;
        if(self->scroll->flags&FGELEMENT_EXPANDX) // If maxdim is -1, this will translate into a -1 maxdim for the text and properly deal with all resizing cases.
          r.right = r.left + self->scroll->maxdim.x;
        if(self->scroll->flags&FGELEMENT_EXPANDY)
          r.bottom = r.top + self->scroll->maxdim.y;

        fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
        if(v)
          self->layout = fgroot_instance->backend.fgFontLayout(self->font, v->p, v->l, self->lineheight, self->letterspacing, &r, self->scroll->flags, self->layout);
        dim->x = r.right - r.left;
        dim->y = r.bottom - r.top;
        assert(!isnan(self->scroll.realsize.x) && !isnan(self->scroll.realsize.y));
        fgTextbox_fixpos(self, self->start, &self->startpos);
        fgTextbox_fixpos(self, self->end, &self->endpos);
      }
    }
    return 0;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Textbox";
  }

  return fgScrollbar_Message(&self->scroll, msg);
}
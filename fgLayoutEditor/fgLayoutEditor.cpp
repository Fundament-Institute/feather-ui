// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"
#include <fstream>

#ifdef BSS_PLATFORM_WIN32
#include "bss-util/win32_includes.h"
#include <ShellAPI.h>
#include <Commdlg.h>
#endif

#undef GetClassName // Fucking windows i swear to god

using namespace bss;

fgLayoutEditor* fgLayoutEditor::Instance = 0;

size_t fgLayoutEditor::_inject(fgRoot* self, const FG_Msg* msg)
{
  if(msg->type == FG_KEYDOWN)
  {
    if(msg->keycode == FG_KEY_F11)
    {
      if(fgDebug_Get() != 0 && !(fgDebug_Get()->tabs.control.element.flags&FGELEMENT_HIDDEN))
        fgDebug_Hide();
      else
        fgDebug_Show(0, 1);
    }
    if(msg->IsCtrlDown())
    {
      switch(msg->keycode)
      {
      case FG_KEY_Z:
        Instance->ProcessEvent(EVENT_UNDO);
        return FG_ACCEPT;
      case FG_KEY_Y:
        Instance->ProcessEvent(EVENT_REDO);
        return FG_ACCEPT;
      case FG_KEY_S:
        Instance->ProcessEvent(EVENT_SAVE);
        return FG_ACCEPT;
      case FG_KEY_N:
        Instance->ProcessEvent(EVENT_NEW);
        return FG_ACCEPT;
      }
    }
  }
  return fgRoot_DefaultInject(self, msg);
}

fgLayoutEditor::fgLayoutEditor(fgLayout* layout, EditorSettings& settings) : EditorBase(layout), _settings(settings), _sizing(HIT_NONE),
_needsave(false), displaylayout(0), _workspaceroot(0), _toolbar(0), _resizebox(layout->base.GetSkin("Editor$Resizebox")),
_curelement(0), _historypos(-1), _insert(0), _insertdest(0)
{
  fgLayoutEditor::Instance = this;
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuAction>("menu_action", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuRecent>("menu_recent", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::ToolAction>("tool_action", this);
  _window = fgSingleton()->gui->LayoutLoad(layout);
  _dialog = layout->GetLayout("dialog");
  fgElement_AddDelegateListener<EditorBase, &EditorBase::WindowOnDestroy>(_window, FG_DESTROY, this);
  _layout.Init(this);
  _skin.Init(this);

  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  _recentmenu = fgGetID("Editor$RecentMenu");
  _toolbar = fgGetID("Editor$Toolbar");
  _toolbarcontrols = fgGetID("Editor$Toolbarcontrols");
  _menu = fgGetID("Editor$Menu");

  if(_workspace)
  {
    _workspaceroot = fgCreate("Element", *_workspace, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    _workspaceroot->message = (fgMessage)WorkspaceRootMessage;
  }

  if(_toolbarcontrols)
  {
    _toolbarcontrols = _toolbarcontrols->AddItem(0);
    std::function<void(const char*)> fn = [this](const char* s) {
      fgElement* e = fgCreate("Radiobutton", _toolbarcontrols, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
      Str path = Str("../media/editor/controls/") + s + ".png";
      fgAsset asset = fgSingleton()->backend.fgCreateAssetFile(0, path.c_str(), 0);
      e->userdata = (void*)s;
      if(asset)
        e->SetAsset(asset);
      else
        e->SetText(s);
      e->SetTooltip(s);
      fgElement_AddDelegateListener<fgLayoutEditor, &fgLayoutEditor::ControlAction>(e, FG_ACTION, this);
    };
    fgIterateControls(&fn, &bss::Delegate<void, const char*>::stublambda);
  }

  bssFill(_toolbarbuttons, 0);
  bssFill(_menubuttons, 0);
  if(_toolbar && _menu)
  {
    for(FG_UINT i = 1; i < EVENT_NUM; ++i)
    {
      _toolbarbuttons[i] = FindUserID(_toolbar, i);
      _menubuttons[i] = FindUserID(_menu, i);
    }
  }
  fgSetInjectFunc(_inject);
  FillRecentList();
  NewFile([this]() { OpenLayout(); });
}

fgLayoutEditor::~fgLayoutEditor()
{}

void fgLayoutEditor::Destroy()
{
  _layout.Clear();
  _skin.Clear();
}
fgElement* fgLayoutEditor::LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout)
{
  fgElement* element = fgCreate(layout->element.type, parent, next, layout->name, layout->element.flags, (layout->element.units == (fgMsgType)~0) ? 0 : &layout->element.transform, layout->element.units | FGUNIT_SNAP);
  assert(element != 0);
  if(!element)
    return 0;
  Instance->_layout.Link(element, layout);
  element->userid = layout->userid;
  fgElement_StyleToMessageArray(&layout->element.style, 0, &element->layoutstyle);
  if(element->layoutstyle)
    fgElement_ApplyMessageArray(0, element, element->layoutstyle);

  for(FG_UINT i = 0; i < layout->children.l; ++i)
    LoadLayout(element, 0, layout->children.p + i);

  return element;
}

uint8_t fgLayoutEditor::HitElement(fgElement* target, const FG_Msg* m)
{
  static const int MARGIN = 5;
  if(!target)
    return HIT_NONE;
  AbsRect area;
  ResolveRect(target, &area);
  AbsRect NCarea = { area.left - MARGIN, area.top - MARGIN, area.right + MARGIN, area.bottom + MARGIN };
  if(m->x > NCarea.right || m->x < NCarea.left || m->y > NCarea.bottom || m->y < NCarea.top)
    return HIT_NONE;
  uint8_t hit = 0;
  if(m->x <= area.left)
    hit |= HIT_LEFT;
  if(m->y <= area.top)
    hit |= HIT_TOP;
  if(m->x >= area.right)
    hit |= HIT_RIGHT;
  if(m->y >= area.bottom)
    hit |= HIT_BOTTOM;
  return hit;
}

size_t fgLayoutEditor::WorkspaceRootMessage(fgElement* self, const FG_Msg* m)
{
  switch(m->type)
  {
  case FG_INJECT:
    if(!fgSingleton()->GetKey(FG_KEY_MENU))
    {
      FG_Msg* inner = reinterpret_cast<FG_Msg*>(m->p);
      FG_Msg msg = *inner;
      msg.type = FG_DEBUGMESSAGE;
      FG_Msg mwrap = *m;
      mwrap.p = &msg;
      fgElement* result = (fgElement*)fgElement_Message(self, &mwrap);
      if(inner->button == FG_MOUSERBUTTON)
      {
        if(inner->type == FG_MOUSEDOWN)
        {
          fgElement* treeitem;
          while(!(treeitem = Instance->_layout.GetTreeItem(Instance->_layout.GetClassLayout(result))) && result != self && result)
            result = result->parent;
          if(result == self || !treeitem)
            Instance->SetCurElement(0);
          else
          {
            treeitem->GotFocus();
            Instance->SetCurElement(result);
          }
        }
        else if(inner->type == FG_MOUSEUP && Instance->_curelement)
          fgElement_TriggerContextMenu(Instance->_layout._contextmenulayout, inner);
        return FG_ACCEPT;
      }

      uint8_t hit = HitElement(Instance->_curelement, inner);
      if(hit != HIT_NONE)
        Instance->hoverelement = 0;

      if(Instance->_insert)
      {
        if(!Instance->_insertdest)
        {
          if(inner->type == FG_MOUSEDOWN)
          {
            Instance->_insertdest = result;
            Instance->_insertend = Instance->_insertbegin = AbsVec{ inner->x, inner->y };
            return FGCURSOR_CROSS;
          }
        }
        else
        {
          Instance->_insertend = AbsVec{ inner->x, inner->y };
          if(inner->type == FG_MOUSEUP)
          {
            if(Instance->_insertbegin.x != Instance->_insertend.x || Instance->_insertbegin.y != Instance->_insertend.y)
            {
              AbsRect out;
              fgGenAbsRect(&out, &Instance->_insertbegin, &Instance->_insertend);
              AbsRect relative;
              ResolveInnerRect(Instance->_insertdest, &relative);
              out.left -= relative.left;
              out.right -= relative.left;
              out.top -= relative.top;
              out.bottom -= relative.top;
              fgTransform tf = { { out.left, 0, out.top, 0, out.right, 0, out.bottom, 0 }, 0,{ 0, 0, 0, 0 } };

              Instance->_layout.InsertElement(Instance->_layout.GetTreeItem(Instance->_layout.GetClassLayout(Instance->_insertdest)), Instance->_insert, false, &tf);
              Instance->_insertdest = 0;
              return FGCURSOR_CROSS;
            } // If these are the same drop into noninsert handler
            else
              Instance->_insertdest = 0;
          }
          else
            return FGCURSOR_CROSS;
        }
      }
      switch(inner->type)
      {
      case FG_MOUSEUP:
        if(fgSingleton()->fgCaptureWindow == self)
          fgSingleton()->fgCaptureWindow = 0;
        if(Instance->_sizing != HIT_NONE)
        {
          hit = Instance->_sizing;
          Instance->_sizing = HIT_NONE;
          switch(hit)
          {
          case HIT_TOP | HIT_LEFT:
          case HIT_BOTTOM | HIT_RIGHT:
            return FGCURSOR_RESIZENWSE;
          case HIT_TOP | HIT_RIGHT:
          case HIT_BOTTOM | HIT_LEFT:
            return FGCURSOR_RESIZENESW;
          case HIT_TOP:
          case HIT_BOTTOM:
            return FGCURSOR_RESIZENS;
          case HIT_LEFT:
          case HIT_RIGHT:
            return FGCURSOR_RESIZEWE;
          }
          return FG_ACCEPT;
        }
        else
          break;
      case FG_MOUSEMOVE:
        if(Instance->_sizing == HIT_NONE && hit == 0 && inner->allbtn&FG_MOUSELBUTTON)
        {
          Instance->AddUndo();
          Instance->_sizing = 0;
        }
        if(Instance->_sizing != HIT_NONE)
        {
          hit = Instance->_sizing;
          CRect area = Instance->_curelement->transform.area;
          AbsVec change = { Instance->_anchor.x + (inner->x - Instance->_lastmouse.x), Instance->_anchor.y + (inner->y - Instance->_lastmouse.y) };
          if(!hit)
          {
            AbsVec dim = { area.right.abs - area.left.abs, area.bottom.abs - area.top.abs };
            area.left.abs = change.x;
            area.right.abs = change.x + dim.x;
            area.top.abs = change.y;
            area.bottom.abs = change.y + dim.y;
          }
          if(hit&HIT_RIGHT)
            area.right.abs = change.x;
          else if(hit&HIT_LEFT)
            area.left.abs = change.x;
          if(hit&HIT_BOTTOM)
            area.bottom.abs = change.y;
          else if(hit&HIT_TOP)
            area.top.abs = change.y;
          Instance->_curelement->SetArea(area);
          Instance->_layout.GetClassLayout(Instance->_curelement)->element.transform.area = area;
          Instance->SetNeedSave(true);
        }
        switch(hit)
        {
        case HIT_TOP | HIT_LEFT:
        case HIT_BOTTOM | HIT_RIGHT:
          return FGCURSOR_RESIZENWSE;
        case HIT_TOP | HIT_RIGHT:
        case HIT_BOTTOM | HIT_LEFT:
          return FGCURSOR_RESIZENESW;
        case HIT_TOP:
        case HIT_BOTTOM:
          return FGCURSOR_RESIZENS;
        case HIT_LEFT:
        case HIT_RIGHT:
          return FGCURSOR_RESIZEWE;
        case 0:
          if(Instance->_sizing == 0)
            return FGCURSOR_RESIZEALL;
        }
        break;
      case FG_MOUSEDOWN:
        if(hit != HIT_NONE)
        {
          if(hit != 0)
          {
            Instance->AddUndo();
            Instance->_sizing = hit;
          }
          Instance->_lastmouse = { inner->x, inner->y };
          CRect& area = Instance->_curelement->transform.area;
          Instance->_anchor.x = (hit&HIT_RIGHT) ? area.right.abs : area.left.abs;
          Instance->_anchor.y = (hit&HIT_BOTTOM) ? area.bottom.abs : area.top.abs;
          fgSingleton()->fgCaptureWindow = self;
        }
        return FG_ACCEPT;
      }
      if(inner->type == FG_MOUSEUP || (hit == HIT_NONE && inner->type == FG_MOUSEDOWN))
      {
        fgElement* treeitem;
        while(!(treeitem = Instance->_layout.GetTreeItem(Instance->_layout.GetClassLayout(result))) && result != self && result)
          result = result->parent;
        if(result == self || !treeitem)
          Instance->SetCurElement(0);
        else
        {
          treeitem->GotFocus();
          Instance->SetCurElement(result);
        }
        return FG_ACCEPT;
      }
      if(hit == HIT_NONE && Instance->_layout.GetTreeItem(Instance->_layout.GetClassLayout(result)))
        Instance->hoverelement = result;
      return 0;
    }
    break;
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = (fgLayout*)m->p;
    if(!layout)
      return 0;

    fgElement_StyleToMessageArray(&layout->base.style, 0, &self->layoutstyle);
    if(self->layoutstyle)
      fgElement_ApplyMessageArray(0, self, self->layoutstyle);
    if(layout->skin)
      self->SetSkin(layout->skin);

    Instance->_layout.ClearLinks();
    fgElement* last = 0;
    for(FG_UINT i = 0; i < layout->children.l; ++i)
      last = LoadLayout(self, 0, layout->children.p + i);
    return (size_t)last;
  }
  case FG_DRAW:
    fgElement_Message(self, m);
    if(Instance->hoverelement)
    {
      AbsRect out;
      ResolveRect(Instance->hoverelement, &out);
      AbsVec lines[5] = { out.topleft, {out.right, out.top}, out.bottomright, {out.left, out.bottom}, out.topleft };
      AbsVec scale = { 1,1 };
      fgSingleton()->backend.fgDrawLines(lines, 5, 0xFF008800, &AbsVec_EMPTY, &scale, 0, &AbsVec_EMPTY, (const fgDrawAuxData*)m->p2);
    }
    if(Instance->_insertdest)
    {
      AbsVec lines[5] = { Instance->_insertbegin,{ Instance->_insertend.x, Instance->_insertbegin.y }, Instance->_insertend,{ Instance->_insertbegin.x, Instance->_insertend.y }, Instance->_insertbegin };
      AbsVec scale = { 1,1 };
      fgSingleton()->backend.fgDrawLines(lines, 5, 0xFF008800, &AbsVec_EMPTY, &scale, 0, &AbsVec_EMPTY, (const fgDrawAuxData*)m->p2);
    }
    if(Instance->_curelement && Instance->_resizebox)
    {
      AbsRect out;
      ResolveRect(Instance->_curelement, &out);
      fgDrawSkin(Instance->_resizebox, &out, (const fgDrawAuxData*)m->p2, 0);
    }
    return FG_ACCEPT;
  case FG_GETSKIN:
    if(!m->p) // If msg->p is none we simply return our self->skin, whatever it is
      return reinterpret_cast<size_t>(self->skin);
    if(self->skin != 0) // Otherwise we are performing a skin lookup for a child
    {
      const char* name = m->e->GetName();
      if(name)
      {
        fgSkin* skin = self->skin->base.GetSkin(name);
        if(skin != 0)
          return reinterpret_cast<size_t>(skin);
      }
      name = m->e->GetClassName();

      fgSkin* skin = self->skin->base.GetSkin(name);
      if(skin != 0)
        return reinterpret_cast<size_t>(skin);
    }
    return 0; // Terminate skin lookup
  case FG_KEYDOWN:
    switch(m->keycode)
    {
      case FG_KEY_DELETE:
        Instance->ProcessEvent(EVENT_DELETE);
        return FG_ACCEPT;
      case FG_KEY_X:
        Instance->ProcessEvent(EVENT_CUT);
        return FG_ACCEPT;
      case FG_KEY_C:
        Instance->ProcessEvent(EVENT_COPY);
        return FG_ACCEPT;
      case FG_KEY_V:
        Instance->ProcessEvent(EVENT_PASTE);
        return FG_ACCEPT;
    }
    break;
  }

  return fgElement_Message(self, m);
}
void fgLayoutEditor::ProcessEvent(int id)
{
  switch(id)
  {
  case EVENT_NEW:
    NewFile([this]() { OpenLayout(); });
    break;
  case EVENT_OPEN:
    CheckSave([this]() {
      LoadFile(FileDialog(true, 0, 0, L"XML Files (*.xml)\0*.xml\0", 0, ".xml").c_str());
    });
    break;
  case EVENT_SAVE:
    SaveFile();
    break;
  case EVENT_SAVEAS:
    _path = "";
    SaveFile();
    break;
  //case EVENT_CLOSE:
  //  break;
  case EVENT_EXIT:
    CheckSave([this]() {
      if(Instance->_window)
        VirtualFreeChild(Instance->_window);
    });
    break;
  case EVENT_CUT:
  case  EVENT_COPY:
  case  EVENT_PASTE:
    break;
  case EVENT_DELETE:
    if(_curelement)
      _layout.RemoveElement(_layout.GetSelected());
    if(_skin.GetSelected())
      _skin.RemoveElement(_skin.GetSelected());
    break;
  case EVENT_UNDO:
    if(_history.size() > 0 && _historypos < (ptrdiff_t)(_history.size() - 1))
    {
      if(_historypos < 0)
      {
        fgLayout* layout = bssMalloc<fgLayout>(1);
        fgLayout_InitCopy(layout, &curlayout, 0);
        _history.push_back(layout);
        _historypos = 0;
      }
      ApplyLayoutCopy(_history[_history.size() - (++_historypos) - 1]);
      EnableEvent(EVENT_UNDO, _history.size() > 0 && _historypos < (ptrdiff_t)(_history.size() - 1));
      EnableEvent(EVENT_REDO, _historypos > 0);
    }
    break;
  case EVENT_REDO:
    if(_historypos > 0)
    {
      ApplyLayoutCopy(_history[_history.size() - (--_historypos) - 1]);
      EnableEvent(EVENT_UNDO, _history.size() > 0 && _historypos < (ptrdiff_t)(_history.size() - 1));
      EnableEvent(EVENT_REDO, _historypos > 0);
    }
    break;
  case EVENT_HELP_DOCS:
#ifdef BSS_PLATFORM_WIN32
    ShellExecuteW(0, 0, L"http://www.blackspherestudios.com/feathergui/docs/", 0, 0, SW_SHOW);
#endif
    break;
  case EVENT_HELP_GITHUB:
#ifdef BSS_PLATFORM_WIN32
    ShellExecuteW(0, 0, L"https://github.com/Black-Sphere-Studios/feathergui/", 0, 0, SW_SHOW);
#endif
    break;
  case EVENT_HELP_ABOUT:
    ShowDialog("About", "FeatherGUI Layout Editor\nVersion v0.1.0\n\n(c)2017 Black Sphere Studios\n\nThis is a free, open-source layout editor for FeatherGUI, maintained as a core part of the FeatherGUI library ecosystem. Please send all feedback and bugs to the FeatherGUI Github: https://github.com/Black-Sphere-Studios/feathergui/", [](char) {}, "Close");
    break;
  }
}
void fgLayoutEditor::MenuAction(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e)
    ProcessEvent(m->e->userid);
}
void fgLayoutEditor::MenuRecent(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e && m->e->userdata)
    LoadFile((const char*)m->e->userdata);
}
void fgLayoutEditor::ToolAction(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  ProcessEvent(e->userid);
}
void fgLayoutEditor::ControlAction(struct _FG_ELEMENT* e, const FG_Msg*)
{
  if(_insert != (const char*)e->userdata)
    _insert = (const char*)e->userdata;
  else
  {
    e->SetValue(0);
    _insert = 0;
  }
}

void fgLayoutEditor::Close(std::function<void()> fn)
{
  CheckSave([=]() {
    ClearLayoutVector(_history);
    DisplayLayout(0);
    _layout.Clear();
    _skin.Clear();
    fn();
  });
}
void fgLayoutEditor::OpenLayout()
{
  Close([this]() {
    _layout.OpenLayout(&curlayout);
    fgSkinBase_IterateSkins(&curlayout.base, &_skin, [](void* p, fgSkin* s, const char*) {((SkinTab*)p)->OpenSkin(s); });
    DisplayLayout(&curlayout);
    if(fgElement* tab = fgGetID("Editor$tablayout"))
      tab->Action();
    SetNeedSave(false);
  });
}
void fgLayoutEditor::DisplayLayout(fgLayout* layout)
{
  if(displaylayout == layout)
    return;
  if(_workspaceroot)
  {
    fgElement_Clear(_workspaceroot);
    SetCurElement(0);
    hoverelement = 0;
    _workspaceroot->SetSkin(0);
    if(layout)
      _workspaceroot->LayoutLoad(layout);
  }
  displaylayout = layout;
}
void fgLayoutEditor::LoadFile(const char* file)
{
  if(!file || !file[0])
    return;
  NewFile([this, file]() {
    fgLayout_LoadFileXML(&curlayout, file);
    _path = file;
    if(std::find(_settings.recent.begin(), _settings.recent.end(), _path) == _settings.recent.end())
    {
      _settings.recent.insert(_settings.recent.begin(), _path);
      if(_settings.recent.size() > MAX_OPEN_HISTORY)
        _settings.recent.pop_back();
      SaveSettings();
      FillRecentList();
    }
    OpenLayout();
  });
}
void fgLayoutEditor::SaveFile()
{
  if(_path.empty())
    _path = FileDialog(false, 0, "newfile.xml", L"XML Files (*.xml)\0*.xml\0", 0, ".xml");

  fgLayout_SaveFileXML(&curlayout, _path.c_str());
  SetNeedSave(false);
}
void fgLayoutEditor::NewFile(std::function<void()> fn)
{
  Close([this, fn]() {
    _path = "";
    fgLayout_Destroy(&curlayout);
    fgLayout_Init(&curlayout, 0, 0);
    fn();
  });
}
void fgLayoutEditor::CheckSave(std::function<void()> fn)
{
  if(_needsave)
  {
    ShowDialog("Unsaved Changes", "You have unsaved changes, would you like to save them first?", [=](char r) {
      switch(r)
      {
    case 0:
      return;
    case 1:
      SaveFile();
    case 2:
      SetNeedSave(false);
      break;
      }
      fn();
    }, "Cancel", "Yes", "No");
  }
  else
    fn();
}

void fgLayoutEditor::ShowDialog(const char* title, const char* text, std::function<void(char)> fn, const char* button1, const char* button2, const char* button3)
{
  if(!_dialog)
    return;
  fgElement* window = fgSingleton()->gui->LayoutLoad(_dialog);
  window->SetText(title);
  fgGetID("dialog$text")->SetText(text);
  fgElement* b1 = fgGetID("dialog$button1");
  fgElement* b2 = fgGetID("dialog$button2");
  fgElement* b3 = fgGetID("dialog$button3");
  fgElement* overlay = fgGetID("Editor$ModalOverlay");
  // TODO: Fix this memory leak or just implement proper threading so callbacks aren't necessary
  auto ret = new std::function<void(struct _FG_ELEMENT*, const FG_Msg*)>([=](struct _FG_ELEMENT* e, const FG_Msg*) { fn(e->userid); if(e != window) VirtualFreeChild(window); overlay->SetFlag(FGELEMENT_HIDDEN, true); });
  auto ret2 = new std::function<void(struct _FG_ELEMENT*, const FG_Msg*)>([=](struct _FG_ELEMENT* e, const FG_Msg*) { window->GotFocus(); });
  fgElement_AddLambdaListener(b1, FG_ACTION, *ret);
  fgElement_AddLambdaListener(b2, FG_ACTION, *ret);
  fgElement_AddLambdaListener(b3, FG_ACTION, *ret);
  fgElement_AddLambdaListener(window, FG_DESTROY, *ret);
  fgElement_AddLambdaListener(overlay, FG_MOUSEDOWN, *ret2);
  fgElement_AddLambdaListener(overlay, FG_MOUSEUP, *ret2);
  if(button1)
    b1->SetText(button1);
  b1->SetFlag(FGELEMENT_HIDDEN, button1 == 0);
  if(button2)
    b2->SetText(button2);
  b2->SetFlag(FGELEMENT_HIDDEN, button2 == 0);
  if(button3)
    b3->SetText(button3);
  b3->SetFlag(FGELEMENT_HIDDEN, button3 == 0);
  overlay->SetFlag(FGELEMENT_HIDDEN, false);
}

#ifdef BSS_PLATFORM_WIN32 //Windows function
std::wstring GetWString(const char* s)
{
  std::wstring r;
  size_t len = MultiByteToWideChar(CP_UTF8, 0, s, -1, 0, 0);
  r.resize(len);
  MultiByteToWideChar(CP_UTF8, 0, s, -1, const_cast<wchar_t*>(r.data()), (int)len);
  return r;
}
#endif

std::string fgLayoutEditor::FileDialog(bool open, unsigned long flags, const char* file, const wchar_t* filter, const char* initdir, const char* defext)
{
#ifdef BSS_PLATFORM_WIN32 //Windows function
  wchar_t buf[MAX_PATH];
  GetCurrentDirectoryW(MAX_PATH, buf);
  std::wstring curdirsave(buf);
  ZeroMemory(buf, MAX_PATH);
  auto wfile = GetWString(file);
  auto winitdir = GetWString(initdir);
  auto wdefext = GetWString(defext);

  if(file != 0) WCSNCPY(buf, MAX_PATH, wfile.c_str(), bssmin(wfile.size(), MAX_PATH - 1));

  OPENFILENAMEW ofn;
  ZeroMemory(&ofn, sizeof(ofn));

  ofn.lStructSize = sizeof(OPENFILENAMEW);
  ofn.hwndOwner = 0;
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = buf;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = (open ? OFN_EXPLORER | OFN_FILEMUSTEXIST : OFN_EXPLORER) | flags;
  ofn.lpstrDefExt = wdefext.c_str();
  ofn.lpstrInitialDir = winitdir.c_str();

  BOOL res = open ? GetOpenFileNameW(&ofn) : GetSaveFileNameW(&ofn);

  SetCurrentDirectoryW(curdirsave.c_str()); //There is actually a flag that's supposed to do this for us but it doesn't work on XP for file open, so we have to do it manually just to be sure
  if(!res) buf[0] = '\0';
  size_t len = WideCharToMultiByte(CP_UTF8, 0, buf, -1, 0, 0, 0, 0);
  std::string r;
  r.resize(len);
  WideCharToMultiByte(CP_UTF8, 0, buf, -1, const_cast<char*>(r.data()), (int)len, 0, 0);
  return r;
#else // BSS_PLATFORM_POSIX
#endif
}
void fgLayoutEditor::SaveSettings()
{
  std::ofstream settingsout("fgLayoutEditor.toml", std::ios_base::binary | std::ios_base::out);
  bss::Serializer<bss::TOMLEngine> serializer;
  serializer.Serialize(_settings, settingsout, "editor");
}

void r_reapplyskin(fgSkin* skin, fgElement* e)
{
  if(e->skin == skin)
  {
    e->skin = 0; // please never actually do this in any real program ever
    e->SetSkin(skin);
  }

  for(fgElement* cur = e->root; cur != 0; cur = cur->next)
    r_reapplyskin(skin, cur);
}

void fgLayoutEditor::ReapplySkin(fgSkin* skin)
{
  if(_workspaceroot)
    r_reapplyskin(skin, _workspaceroot);
}

void fgLayoutEditor::FillRecentList()
{
  if(_recentmenu)
  {
    fgElement_Clear(_recentmenu);
    if(!_settings.recent.size())
      _recentmenu->parent->SetFlag(FGCONTROL_DISABLE, true);
    else
    {
      _recentmenu->parent->SetFlag(FGCONTROL_DISABLE, false);
      for(const auto& s : _settings.recent)
        _recentmenu->AddItemText(fgTrimPathFromFile(s.c_str()))->userdata = (void*)s.c_str();
    }
  }
}

void fgLayoutEditor::SetNeedSave(bool needsave)
{
  _needsave = needsave;
  Str name("FeatherGUI Layout Editor - ");
  const char* trim = fgTrimPathFromFile(_path.c_str());
  if(!trim[0])
    trim = "untitled";
  name += trim;
  if(_needsave)
    name += "*";
  if(_window && (!_window->GetText() || _window->GetText() != name))
    _window->SetText(name.c_str());
}

void fgLayoutEditor::ClearLayoutVector(std::vector<fgLayout*>& v)
{
  for(size_t i = 0; i < v.size(); ++i)
  {
    fgLayout_Destroy(v[i]);
    free(v[i]);
  }
  v.clear();
}

void fgLayoutEditor::EnableEvent(FG_UINT id, bool enable)
{
  if(_toolbarbuttons[id])
    _toolbarbuttons[id]->SetFlag(FGCONTROL_DISABLE, !enable);
  if(_menubuttons[id])
    _menubuttons[id]->SetFlag(FGCONTROL_DISABLE, !enable);
}
void fgLayoutEditor::AddUndo()
{
  for(ptrdiff_t i = 0; i < _historypos; ++i)
  {
    fgLayout_Destroy(_history.back());
    free(_history.back());
    _history.pop_back();
  }
  if(_historypos < 0)
  {
    fgLayout* layout = bssMalloc<fgLayout>(1);
    fgLayout_InitCopy(layout, &curlayout, 0);
    _history.push_back(layout);
  }
  EnableEvent(EVENT_UNDO, true);
  EnableEvent(EVENT_REDO, false);
  _historypos = -1;
}

void fgLayoutEditor::ApplyLayoutCopy(fgLayout* layout)
{
  DisplayLayout(0);
  _layout.Clear();
  _skin.Clear();
  fgLayout_Destroy(&curlayout);
  fgLayout_InitCopy(&curlayout, layout, 0);
  _layout.OpenLayout(&curlayout);
  fgSkinBase_IterateSkins(&curlayout.base, &_skin, [](void* p, fgSkin* s, const char*) {((SkinTab*)p)->OpenSkin(s); });
  DisplayLayout(&curlayout);
  if(fgElement* tab = fgGetID("Editor$tablayout"))
    tab->Action();
}

fgElement* fgLayoutEditor::FindUserID(fgElement* parent, FG_UINT userid)
{
  for(fgElement* e = parent->root; e != 0; e = e->next)
  {
    if(!(e->flags&FGELEMENT_BACKGROUND) && e->userid == userid)
      return e;
    if(fgElement* child = FindUserID(e, userid))
      return child;
  }
  return 0;
}
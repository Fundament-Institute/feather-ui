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
  }
  return fgRoot_DefaultInject(self, msg);
}

fgLayoutEditor::fgLayoutEditor(fgLayout* layout, EditorSettings& settings) : EditorBase(layout), _settings(settings),
  _needsave(false), displaylayout(0), _curelement(0), _hoverelement(0), _workspaceroot(0)
{
  fgLayoutEditor::Instance = this;
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuFile>("menu_file", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuRecent>("menu_recent", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuEdit>("menu_edit", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuView>("menu_view", this);
  fgRegisterDelegate<fgLayoutEditor, &fgLayoutEditor::MenuHelp>("menu_help", this);
  _window = fgSingleton()->gui->LayoutLoad(layout);
  fgElement_AddDelegateListener<EditorBase, &EditorBase::WindowOnDestroy>(_window, FG_DESTROY, this);
  _layout.Init(this);
  _skin.Init(this);

  _mainwindow = reinterpret_cast<fgWindow*>(fgGetID("Editor$mainwindow"));
  _workspace = reinterpret_cast<fgWorkspace*>(fgGetID("Editor$workspace"));
  if(_workspace)
  {
    _workspaceroot = fgCreate("Element", *_workspace, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    _workspaceroot->message = (fgMessage)WorkspaceRootMessage;
  }
  /*
  { // Setup toolbar
    fgElement* mainbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // New
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Open
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Save
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Save All
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Undo
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Redo
                                                                    // seperator
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Cut
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Copy
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Paste
    fgCreate("Button", mainbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Delete

    fgElement* viewbar = self->toolbar->box->AddItem(0);
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Show Grid
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Show Crosshair
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Hide Rulers
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Hide Cursors
                                                                    // seperator
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Snapping
    fgCreate("Button", viewbar, 0, 0, FGFLAGS_DEFAULTS, 0, 0); // Toggle Wireframe

    fgElement* insertbar = self->toolbar->box->AddItem(0); // Bar with buttons for each element
    fgIterateControls(insertbar, [](void* p, const char* s) { fgElement* e = (fgElement*)p; e->AddItemText(s); });
  }*/

  fgSetInjectFunc(_inject);
}

fgLayoutEditor::~fgLayoutEditor()
{
}

void fgLayoutEditor::Destroy()
{
  _layout.Clear();
  _skin.Clear();
}
fgElement* fgLayoutEditor::LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout)
{
  fgElement* element = fgCreate(layout->element.type, parent, next, layout->name, layout->element.flags, (layout->element.units == -1) ? 0 : &layout->element.transform, layout->element.units | FGUNIT_SNAP);
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

size_t fgLayoutEditor::WorkspaceRootMessage(fgElement* self, const FG_Msg* m)
{
  switch(m->type)
  {
  case FG_INJECT:
    if(!fgSingleton()->GetKey(FG_KEY_MENU))
    {
      FG_Msg msg = *reinterpret_cast<FG_Msg*>(m->p);
      msg.type = FG_DEBUGMESSAGE;
      FG_Msg mwrap = *m;
      mwrap.p = &msg;
      fgElement* result = (fgElement*)fgElement_Message(self, &mwrap);
      if(reinterpret_cast<FG_Msg*>(m->p)->type == FG_MOUSEDOWN)
        Instance->_curelement = result;
      Instance->_hoverelement = result;
      return FG_ACCEPT;
    }
    break;
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = (fgLayout*)m->p;
    if(!layout)
      return 0;
    
    fgElement_StyleToMessageArray(&layout->style, 0, &self->layoutstyle);
    if(self->layoutstyle)
      fgElement_ApplyMessageArray(0, self, self->layoutstyle);

    Instance->_layout.ClearLinks();
    fgElement* last = 0;
    for(FG_UINT i = 0; i < layout->children.l; ++i)
      last = LoadLayout(self, 0, layout->children.p + i);
    return (size_t)last;
  }
  case FG_DRAW:
    fgElement_Message(self, m);
    if(Instance->_hoverelement)
    {
      AbsRect out;
      ResolveRect(Instance->_hoverelement, &out);
      AbsVec lines[5] = { out.topleft, {out.right, out.top}, out.bottomright, {out.left, out.bottom}, out.topleft };
      AbsVec scale = { 1,1 };
      fgSingleton()->backend.fgDrawLines(lines, 5, 0xFF008800, &AbsVec_EMPTY, &scale, 0, &AbsVec_EMPTY, (const fgDrawAuxData*)m->p2);
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
        fgSkin* skin = fgSkin_GetSkin(self->skin, name);
        if(skin != 0)
          return reinterpret_cast<size_t>(skin);
      }
      name = m->e->GetClassName();

      fgSkin* skin = fgSkin_GetSkin(self->skin, name);
      if(skin != 0)
        return reinterpret_cast<size_t>(skin);
    }
    return 0; // Terminate skin lookup
  }

  return fgElement_Message(self, m);
}
void fgLayoutEditor::MenuFile(struct _FG_ELEMENT* e, const FG_Msg* m)
{
  if(m->e)
  {
    switch(m->e->userid)
    {
    case 1:
      NewFile();
      break;
    case 2:
      CheckSave();
      LoadFile(FileDialog(true, 0, 0, L"XML Files (*.xml)\0*.xml\0", 0, ".xml").c_str());
      break;
    case 4:
      SaveFile();
      break;
    case 5:
      _path = "";
      SaveFile();
      break;
    case 6:
      Close();
      break;
    case 8:
      if(Instance->_mainwindow)
        VirtualFreeChild(*Instance->_mainwindow);
    }
  }
}
void fgLayoutEditor::MenuRecent(struct _FG_ELEMENT*, const FG_Msg* m)
{
  if(m->e && m->e->userdata)
  {
    LoadFile((const char*)m->e->userdata);
  }
}
void fgLayoutEditor::MenuEdit(struct _FG_ELEMENT*, const FG_Msg*)
{

}
void fgLayoutEditor::MenuView(struct _FG_ELEMENT*, const FG_Msg*)
{

}

void fgLayoutEditor::MenuHelp(struct _FG_ELEMENT*, const FG_Msg* m)
{
  if(m->e)
  {
    switch(m->e->userid)
    {
    case 2:
#ifdef BSS_PLATFORM_WIN32
      ShellExecute(0, 0, L"http://www.google.com", 0, 0, SW_SHOW);
#endif
      break;
    case 3:
      ShowDialog("FeatherGUI Layout Editor\nVersion v0.1.0\n\n(c)2017 Black Sphere Studios\n\nThis is a free, open-source layout editor for FeatherGUI, maintained as a core part of the FeatherGUI library ecosystem. Please send all feedback and bugs to the FeatherGUI Github: https://github.com/Black-Sphere-Studios/feathergui/", "Close");
      break;
    }
  }
}
bool fgLayoutEditor::Close()
{
  if(!CheckSave())
    return false;
  if(_workspaceroot)
    fgElement_Clear(_workspaceroot);
  _layout.Clear();
  _skin.Clear();
  return true;
}
void fgLayoutEditor::OpenLayout()
{
  if(!Close())
    return;
  _layout.OpenLayout(&curlayout);
  fgSkinBase_IterateSkins(&curlayout.base, &_skin, [](void* p, fgSkin* s, const char*) {((SkinTab*)p)->OpenSkin(s); });
  DisplayLayout(&curlayout);
}
void fgLayoutEditor::DisplayLayout(fgLayout* layout)
{
  if(displaylayout == layout)
    return;
  if(_workspaceroot)
  {
    fgElement_Clear(_workspaceroot);
    _workspaceroot->LayoutLoad(layout);
  }
  displaylayout = layout;
}
void fgLayoutEditor::LoadFile(const char* file)
{
  if(!file || !file[0])
    return;
  NewFile();
  fgLayout_LoadFileXML(&curlayout, file);
  _path = file;
  if(std::find(_settings.recent.begin(), _settings.recent.end(), _path) == _settings.recent.end())
  {
    _settings.recent.insert(_settings.recent.begin(), _path);
    if(_settings.recent.size() > MAX_OPEN_HISTORY)
      _settings.recent.erase(_settings.recent.end());
    SaveSettings();
  }
  OpenLayout();
}
void fgLayoutEditor::SaveFile()
{
  if(_path.empty())
    _path = FileDialog(false, 0, "newfile.xml", L"XML Files (*.xml)\0*.xml\0", 0, ".xml");

  fgLayout_SaveFileXML(&curlayout, _path.c_str());
  _needsave = false;
}
void fgLayoutEditor::NewFile()
{
  _path = "";
  fgLayout_Destroy(&curlayout);
  fgLayout_Init(&curlayout, 0);
}
bool fgLayoutEditor::CheckSave()
{
  if(_needsave)
  {
    switch(ShowDialog("You have unsaved changes, would you like to save them first?", "Cancel", "Yes", "No"))
    {
    case 0:
      return false;
    case 1:
      SaveFile();
    case 2:
      _needsave = false;
      break;
    }
  }

  return true;
}
char fgLayoutEditor::ShowDialog(const char* text, const char* button1, const char* button2, const char* button3)
{
  return 0;
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

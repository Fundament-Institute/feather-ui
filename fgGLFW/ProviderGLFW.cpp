// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "Provider.hpp"

#include "platform.hpp"
#include "ProviderGLFW.hpp"
#include <cfloat>
#include <cstring>

using GLFW::Provider;

int Provider::_lasterr            = 0;
int Provider::_refcount           = 0;
char Provider::_lasterrdesc[1024] = {};
int Provider::_maxjoy             = 0;
Provider* Provider::_singleton    = nullptr;
const float Provider::BASE_DPI    = 96.0f;

FG_Result Provider::Behavior(Window* w, const FG_Msg& msg)
{
  return (*_behavior)(w, const_cast<FG_Msg*>(&msg), _uictx, w->window_id);
}

FG_Window* Provider::CreateWindowImpl(FG_DesktopInterface* self, uintptr_t window_id, FG_Display* display, FG_Vec2* pos,
                                      FG_Vec2* dim, const char* caption, uint64_t flags)
{
  auto backend = static_cast<Provider*>(self);
  _lasterr     = 0;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  Window* window = new Window(backend, reinterpret_cast<GLFWmonitor*>(display), window_id, pos, dim, flags, caption);

  if(!_lasterr)
  {
    window->_next     = backend->_windows;
    backend->_windows = window;
    if(window->_next)
      window->_next->_prev = window;

    return window;
  }

  delete window;
  return nullptr;
}

int Provider::SetWindow(FG_DesktopInterface* self, FG_Window* window, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                        const char* caption, uint64_t flags)
{
  if(!window)
    return ERR_MISSING_PARAMETER;
  auto w = static_cast<Window*>(window);

  auto glwindow = w->GetWindow();
  if(!glwindow)
    return ERR_INVALID_CALL;
  if(caption)
    glfwSetWindowTitle(glwindow, caption);

  if((w->_flags ^ flags) & FG_WindowFlag_Fullscreen) // If we toggled fullscreen we need a different code path
  {
    int posx, posy;
    int dimx, dimy;
    glfwGetWindowPos(glwindow, &posx, &posy);
    glfwGetWindowSize(glwindow, &dimx, &dimy);
    if(flags & FG_WindowFlag_Fullscreen)
      glfwSetWindowMonitor(glwindow, (GLFWmonitor*)display, 0, 0, !dim ? dimx : static_cast<int>(ceilf(dim->x)),
                           !dim ? dimy : static_cast<int>(ceilf(dim->y)), GLFW_DONT_CARE);
    else
      glfwSetWindowMonitor(glwindow, NULL, !pos ? posx : static_cast<int>(ceilf(pos->x)),
                           !pos ? posy : static_cast<int>(ceilf(pos->y)), !dim ? dimx : static_cast<int>(ceilf(dim->x)),
                           !dim ? dimy : static_cast<int>(ceilf(dim->y)), GLFW_DONT_CARE);
  }
  else
  {
    if(pos)
      glfwSetWindowPos(glwindow, static_cast<int>(ceilf(pos->x)), static_cast<int>(ceilf(pos->y)));
    if(dim)
      glfwSetWindowSize(glwindow, static_cast<int>(ceilf(dim->x)), static_cast<int>(ceilf(dim->y)));
  }

  if(flags & FG_WindowFlag_Maximized)
    glfwMaximizeWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  if(flags & FG_WindowFlag_Minimized)
    glfwIconifyWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  w->_flags = flags;
  return ERR_SUCCESS;
}

int Provider::DestroyWindow(FG_DesktopInterface* self, FG_Window* window)
{
  if(!self || !window)
    return ERR_MISSING_PARAMETER;

  _lasterr = 0; // We set this to capture GLFW errors, which are seperate from OpenGL errors (for right now)
  delete static_cast<Window*>(window);
  return _lasterr;
}

int Provider::InvalidateWindow(FG_DesktopInterface* self, FG_Window* window, FG_Rect* optarea)
{
  if(!window)
    return ERR_MISSING_PARAMETER;
  auto w = static_cast<Window*>(window);

  _lasterr = 0; // We set this to capture GLFW errors, which are seperate from OpenGL errors (for right now)
  w->Invalidate(optarea);
  return _lasterr;
}

int Provider::PutClipboard(FG_DesktopInterface* self, FG_Window* window, FG_Clipboard kind, const char* data,
                           uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_CLIPBOARD_FAILURE;
  if(data != 0 && count > 0 && EmptyClipboard())
  {
    if(kind == FG_Clipboard_Text)
    {
      size_t unilen  = MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(count), 0, 0);
      HGLOBAL unimem = GlobalAlloc(GMEM_MOVEABLE, unilen * sizeof(wchar_t));
      if(unimem)
      {
        wchar_t* uni = reinterpret_cast<wchar_t*>(GlobalLock(unimem));
        size_t sz    = MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(count), uni, static_cast<int>(unilen));
        if(sz < unilen) // ensure we have a null terminator
          uni[sz] = 0;
        GlobalUnlock(unimem);
        SetClipboardData(CF_UNICODETEXT, unimem);
      }
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count + 1);
      if(gmem)
      {
        char* mem = reinterpret_cast<char*>(GlobalLock(gmem));
        MEMCPY(mem, count + 1, data, count);
        mem[count] = 0;
        GlobalUnlock(gmem);
        SetClipboardData(CF_TEXT, gmem);
      }
    }
    else
    {
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count);
      if(gmem)
      {
        void* mem = GlobalLock(gmem);
        MEMCPY(mem, count, data, count);
        GlobalUnlock(gmem);
        UINT format = CF_PRIVATEFIRST;
        switch(kind)
        {
        case FG_Clipboard_Wave: format = CF_WAVE; break;
        case FG_Clipboard_Bitmap: format = CF_BITMAP; break;
        }
        SetClipboardData(format, gmem);
      }
    }
  }
  CloseClipboard();
  return ERR_SUCCESS;
#else
  if(kind != FG_Clipboard_Text)
    return ERR_INVALID_KIND;

  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), data);
  return _lasterr;
#endif
}

uint32_t Provider::GetClipboard(FG_DesktopInterface* self, FG_Window* window, FG_Clipboard kind, void* target,
                                uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_SUCCESS;
  UINT format = CF_PRIVATEFIRST;
  switch(kind)
  {
  case FG_Clipboard_Text:
    if(IsClipboardFormatAvailable(CF_UNICODETEXT))
    {
      HANDLE gdata = GetClipboardData(CF_UNICODETEXT);
      SIZE_T len   = 0;
      if(gdata)
      {
        SIZE_T size        = GlobalSize(gdata) / 2;
        const wchar_t* str = (const wchar_t*)GlobalLock(gdata);
        if(str)
        {
          len = WideCharToMultiByte(CP_UTF8, 0, str, static_cast<int>(size), 0, 0, NULL, NULL);

          if(target && count >= len)
            len = WideCharToMultiByte(CP_UTF8, 0, str, static_cast<int>(size), (char*)target, static_cast<int>(count), NULL,
                                      NULL);

          GlobalUnlock(gdata);
        }
      }
      CloseClipboard();
      return (uint32_t)len;
    }
    format = CF_TEXT;
    break;
  case FG_Clipboard_Wave: format = CF_WAVE; break;
  case FG_Clipboard_Bitmap: format = CF_BITMAP; break;
  }
  HANDLE gdata = GetClipboardData(format);
  SIZE_T size  = 0;
  if(gdata)
  {
    size = GlobalSize(gdata);
    if(target && count >= size)
    {
      void* data = GlobalLock(gdata);
      if(data)
      {
        MEMCPY(target, count, data, size);
        GlobalUnlock(gdata);
      }
      else
        size = 0;
    }
  }
  CloseClipboard();
  return (uint32_t)size;
#else
  if(kind != FG_Clipboard_Text)
    return ERR_SUCCESS;

  auto str = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(target)
    strncpy(reinterpret_cast<char*>(target), str, count);
  return strlen(str) + 1;
#endif
}

bool Provider::CheckClipboard(FG_DesktopInterface* self, FG_Window* window, FG_Clipboard kind)
{
#ifdef FG_PLATFORM_WIN32
  switch(kind)
  {
  case FG_Clipboard_Text: return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT);
  case FG_Clipboard_Wave: return IsClipboardFormatAvailable(CF_WAVE);
  case FG_Clipboard_Bitmap: return IsClipboardFormatAvailable(CF_BITMAP);
  case FG_Clipboard_Custom: return IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  case FG_Clipboard_All:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT) |
           IsClipboardFormatAvailable(CF_WAVE) | IsClipboardFormatAvailable(CF_BITMAP) |
           IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  }
  return false;
#else
  if(kind != FG_Clipboard_Text && kind != FG_Clipboard_All)
    return false;

  _lasterr = 0;
  auto p   = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(!p || _lasterr != 0)
    return false;
  return p[0] != 0;
#endif
}

int Provider::ClearClipboard(FG_DesktopInterface* self, FG_Window* window, FG_Clipboard kind)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_CLIPBOARD_FAILURE;
  if(!EmptyClipboard())
    return ERR_CLIPBOARD_FAILURE;
  CloseClipboard();
  return ERR_SUCCESS;
#else
  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), "");
  return _lasterr;
#endif
}

int Provider::ProcessMessages(FG_DesktopInterface* self, FG_Window* window, void* ui_state)
{
  // TODO: handle GLFW errors
  auto backend    = static_cast<Provider*>(self);
  backend->_uictx = ui_state;
  glfwPollEvents();
  if(!window)
    window = backend->_windows;
  if(window)
    static_cast<Window*>(window)->PollJoysticks();

  backend->_uictx = nullptr;
  return backend->_windows != nullptr;
}
int Provider::GetMessageSyncObject(FG_DesktopInterface* self, FG_Window* window)
{
  auto backend = static_cast<Provider*>(self);
  return ERR_NOT_IMPLEMENTED;
}

int Provider::SetCursorImpl(FG_DesktopInterface* self, FG_Window* window, FG_Cursor cursor)
{
  static GLFWcursor* arrow   = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  static GLFWcursor* ibeam   = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
  static GLFWcursor* cross   = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  static GLFWcursor* hand    = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
  static GLFWcursor* hresize = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
  static GLFWcursor* vresize = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);

  auto glwindow = static_cast<Window*>(window)->GetWindow();
  switch(cursor)
  {
  case FG_Cursor_Arrow: glfwSetCursor(glwindow, arrow); return _lasterr;
  case FG_Cursor_IBeam: glfwSetCursor(glwindow, ibeam); return _lasterr;
  case FG_Cursor_Cross: glfwSetCursor(glwindow, cross); return _lasterr;
  case FG_Cursor_Hand: glfwSetCursor(glwindow, hand); return _lasterr;
  case FG_Cursor_ResizeWE: glfwSetCursor(glwindow, hresize); return _lasterr;
  case FG_Cursor_ResizeNS: glfwSetCursor(glwindow, vresize); return _lasterr;
  }

  return ERR_INVALID_CURSOR;
}

int Provider::GetDisplayIndex(FG_DesktopInterface* self, unsigned int index, FG_Display* out)
{
  int count;
  auto monitors = glfwGetMonitors(&count);
  if(index >= static_cast<unsigned int>(count))
    return ERR_INVALID_DISPLAY;
  return GetDisplay(self, reinterpret_cast<uintptr_t>(monitors[index]), out);
}

int Provider::GetDisplay(FG_DesktopInterface* self, uintptr_t handle, FG_Display* out)
{
  if(!handle || !out)
    return ERR_MISSING_PARAMETER;
  out->handle  = handle;
  out->primary = glfwGetPrimaryMonitor() == reinterpret_cast<GLFWmonitor*>(handle);

  _lasterr = 0;
  glfwGetMonitorWorkarea(reinterpret_cast<GLFWmonitor*>(handle), &out->offset.x, &out->offset.y, &out->size.x,
                         &out->size.y);
  if(_lasterr != 0)
    return _lasterr;

  _lasterr = 0;
  glfwGetMonitorContentScale(reinterpret_cast<GLFWmonitor*>(handle), &out->dpi.x, &out->dpi.y);
  if(_lasterr != 0)
    return _lasterr;

  out->dpi.x *= BASE_DPI;
  out->dpi.y *= BASE_DPI;
  out->scale = 1.0f; // GLFW doesn't know what text scaling is
  return ERR_SUCCESS;
}

int Provider::GetDisplayWindow(FG_DesktopInterface* self, FG_Window* window, FG_Display* out)
{
  return GetDisplay(self, reinterpret_cast<uintptr_t>(glfwGetWindowMonitor(static_cast<Window*>(window)->GetWindow())),
                    out);
}

int Provider::CreateSystemControl(FG_DesktopInterface* self, FG_Window* context, const char* id, FG_Rect* area, ...)
{
  auto backend = static_cast<Provider*>(self);
  return ERR_NOT_IMPLEMENTED;
}
int Provider::SetSystemControl(FG_DesktopInterface* self, FG_Window* context, void* control, FG_Rect* area, ...)
{
  return ERR_NOT_IMPLEMENTED;
}
int Provider::DestroySystemControl(FG_DesktopInterface* self, FG_Window* context, void* control)
{
  return ERR_NOT_IMPLEMENTED;
}

int Provider::DestroyImpl(FG_DesktopInterface* self)
{
  if(!self)
    return ERR_INVALID_PARAMETER;
  delete static_cast<Provider*>(self);

  if(--Provider::_refcount == 0)
  {
    Provider::_maxjoy  = 0;
    Provider::_lasterr = 0;
    glfwTerminate();
  }

  return ERR_SUCCESS;
}

extern "C" FG_COMPILER_DLLEXPORT FG_DesktopInterface* fgGLFW(void* log_context, FG_Log log, FG_Behavior behavior)
{
  static_assert(std::is_same<FG_InitDesktop, decltype(&fgGLFW)>::value, "fgOpenGL must match InitBackend function pointer");

#ifdef FG_PLATFORM_WIN32
  typedef BOOL(WINAPI * tGetPolicy)(LPDWORD lpFlags);
  typedef BOOL(WINAPI * tSetPolicy)(DWORD dwFlags);
  const DWORD EXCEPTION_SWALLOWING = 0x1;
  DWORD dwFlags;

  HMODULE kernel32 = LoadLibraryA("kernel32.dll");
  fgassert(kernel32 != 0);
  tGetPolicy pGetPolicy = (tGetPolicy)GetProcAddress(kernel32, "GetProcessUserModeExceptionPolicy");
  tSetPolicy pSetPolicy = (tSetPolicy)GetProcAddress(kernel32, "SetProcessUserModeExceptionPolicy");
  if(pGetPolicy && pSetPolicy && pGetPolicy(&dwFlags))
    pSetPolicy(dwFlags & ~EXCEPTION_SWALLOWING); // Turn off the filter
#endif

  if(++Provider::_refcount == 1)
  {
    glfwSetErrorCallback(&Provider::ErrorCallback);

    if(!glfwInit())
    {
      FG_LogValue logvalues[2] = { { .type = FG_LogType_I32, .i32 = Provider::_lasterr },
                                   { .type = FG_LogType_String, .string = Provider::_lasterrdesc } };
      (*log)(log_context, FG_Level_Error, __FILE__, __LINE__, "glfwInit() failed!", logvalues, 2, &Provider::FreeImpl);
      --Provider::_refcount;
      return nullptr;
    }

    glfwSetJoystickCallback(&Provider::JoystickCallback);
  }

  return new Provider(log_context, log, behavior);
}

#ifdef FG_PLATFORM_WIN32
long long GetRegistryValueW(HKEY__* hKeyRoot, const wchar_t* szKey, const wchar_t* szValue, unsigned char* data,
                            unsigned long sz)
{
  HKEY__* hKey;
  LRESULT e = RegOpenKeyExW(hKeyRoot, szKey, 0, KEY_READ, &hKey);
  if(!hKey)
    return -2;
  LSTATUS r = RegQueryValueExW(hKey, szValue, 0, 0, data, &sz);
  RegCloseKey(hKey);
  if(r == ERROR_SUCCESS)
    return sz;
  return (r == ERROR_MORE_DATA) ? sz : -1;
}
#endif

Provider::Provider(void* log_context, FG_Log log, FG_Behavior behavior) :
  _logctx(log_context), _log(log), _behavior(behavior), _windows(nullptr), _uictx(nullptr)
{
  createWindow         = &CreateWindowImpl;
  setWindow            = &SetWindow;
  destroyWindow        = &DestroyWindow;
  invalidateWindow     = &InvalidateWindow;
  putClipboard         = &PutClipboard;
  getClipboard         = &GetClipboard;
  checkClipboard       = &CheckClipboard;
  clearClipboard       = &ClearClipboard;
  processMessages      = &ProcessMessages;
  getMessageSyncObject = &GetMessageSyncObject;
  setCursor            = &SetCursorImpl;
  getDisplayIndex      = &GetDisplayIndex;
  getDisplay           = &GetDisplay;
  getDisplayWindow     = &GetDisplayWindow;
  createSystemControl  = &CreateSystemControl;
  setSystemControl     = &SetSystemControl;
  destroySystemControl = &DestroySystemControl;
  destroy              = &DestroyImpl;

  this->LOG(FG_Level_Notice, "Initializing fgGLFW...");

#ifdef FG_PLATFORM_WIN32
  #ifdef FG_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
  #endif

  long long sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate",
                           reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      cursorblink = _wtoi(buf.data());
  }
  else
    this->LOG(FG_Level_Warning, "Couldn't get user's cursor blink rate.");

  sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime",
                           reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      tooltipdelay = _wtoi(buf.data());
  }
  else
    this->LOG(FG_Level_Warning, "Couldn't get user's mouse hover time.");

#else
  cursorblink  = 530;
  tooltipdelay = 500;
#endif
  _singleton = this;
}

Provider::~Provider()
{
  while(_windows)
  {
    auto p = _windows->_next;
    delete _windows;
    _windows = p;
  }

  if(_singleton == this)
    _singleton = nullptr;

#ifdef FG_PLATFORM_WIN32
  PostQuitMessage(0);
#endif
}
void Provider::ErrorCallback(int error, const char* description)
{
  _lasterr = error;
#ifdef _WIN32
  strncpy_s(_lasterrdesc, description, 1024);
#else
  strncpy(_lasterrdesc, description, 1024);
#endif

  if(_singleton)
    _singleton->LOG(FG_Level_Error, description);
}

void Provider::JoystickCallback(int id, int connected)
{
  // This only keeps an approximate count of the highest joy ID because internally, GLFW has it's own array of joysticks
  // that it derives the ID from anyway, with a maximum of 16.
  if(connected && id > _maxjoy)
    _maxjoy = id;
  if(!connected && id == _maxjoy)
    --_maxjoy;
}

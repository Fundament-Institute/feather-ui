// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "Provider.hpp"

#include "feather/compiler.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef FG_PLATFORM_WIN32
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include "GLFW/glfw3native.h"
#else
  #define GLFW_EXPOSE_NATIVE_X11
  #include "GLFW/glfw3native.h"
#endif

#include "ProviderGLFW.hpp"
#include <cstring>
#include <cmath>

using namespace GLFW;

// We have to translate all of GLFW's key values into Feather's universal key codes.
uint8_t WindowGL::KeyMap[512] = { 1, 0 };
void WindowGL::FillKeyMap()
{
  // No point in making this threadsafe because GLFW isn't threadsafe.
  if(!WindowGL::KeyMap[0])
    return;
  WindowGL::KeyMap[0]                      = 0;
  WindowGL::KeyMap[GLFW_KEY_SPACE]         = FG_Keys_SPACE;
  WindowGL::KeyMap[GLFW_KEY_APOSTROPHE]    = FG_Keys_APOSTROPHE; // '
  WindowGL::KeyMap[GLFW_KEY_COMMA]         = FG_Keys_COMMA;      // ,
  WindowGL::KeyMap[GLFW_KEY_MINUS]         = FG_Keys_MINUS;
  WindowGL::KeyMap[GLFW_KEY_PERIOD]        = FG_Keys_PERIOD;
  WindowGL::KeyMap[GLFW_KEY_SLASH]         = FG_Keys_SLASH; // /
  WindowGL::KeyMap[GLFW_KEY_0]             = FG_Keys_0;
  WindowGL::KeyMap[GLFW_KEY_1]             = FG_Keys_1;
  WindowGL::KeyMap[GLFW_KEY_2]             = FG_Keys_2;
  WindowGL::KeyMap[GLFW_KEY_3]             = FG_Keys_3;
  WindowGL::KeyMap[GLFW_KEY_4]             = FG_Keys_4;
  WindowGL::KeyMap[GLFW_KEY_5]             = FG_Keys_5;
  WindowGL::KeyMap[GLFW_KEY_6]             = FG_Keys_6;
  WindowGL::KeyMap[GLFW_KEY_7]             = FG_Keys_7;
  WindowGL::KeyMap[GLFW_KEY_8]             = FG_Keys_8;
  WindowGL::KeyMap[GLFW_KEY_9]             = FG_Keys_9;
  WindowGL::KeyMap[GLFW_KEY_SEMICOLON]     = FG_Keys_SEMICOLON; // ;
  WindowGL::KeyMap[GLFW_KEY_EQUAL]         = FG_Keys_PLUS;      // =+
  WindowGL::KeyMap[GLFW_KEY_A]             = FG_Keys_A;
  WindowGL::KeyMap[GLFW_KEY_B]             = FG_Keys_B;
  WindowGL::KeyMap[GLFW_KEY_C]             = FG_Keys_C;
  WindowGL::KeyMap[GLFW_KEY_D]             = FG_Keys_D;
  WindowGL::KeyMap[GLFW_KEY_E]             = FG_Keys_E;
  WindowGL::KeyMap[GLFW_KEY_F]             = FG_Keys_F;
  WindowGL::KeyMap[GLFW_KEY_G]             = FG_Keys_G;
  WindowGL::KeyMap[GLFW_KEY_H]             = FG_Keys_H;
  WindowGL::KeyMap[GLFW_KEY_I]             = FG_Keys_I;
  WindowGL::KeyMap[GLFW_KEY_J]             = FG_Keys_J;
  WindowGL::KeyMap[GLFW_KEY_K]             = FG_Keys_K;
  WindowGL::KeyMap[GLFW_KEY_L]             = FG_Keys_L;
  WindowGL::KeyMap[GLFW_KEY_M]             = FG_Keys_M;
  WindowGL::KeyMap[GLFW_KEY_N]             = FG_Keys_N;
  WindowGL::KeyMap[GLFW_KEY_O]             = FG_Keys_O;
  WindowGL::KeyMap[GLFW_KEY_P]             = FG_Keys_P;
  WindowGL::KeyMap[GLFW_KEY_Q]             = FG_Keys_Q;
  WindowGL::KeyMap[GLFW_KEY_R]             = FG_Keys_R;
  WindowGL::KeyMap[GLFW_KEY_S]             = FG_Keys_S;
  WindowGL::KeyMap[GLFW_KEY_T]             = FG_Keys_T;
  WindowGL::KeyMap[GLFW_KEY_U]             = FG_Keys_U;
  WindowGL::KeyMap[GLFW_KEY_V]             = FG_Keys_V;
  WindowGL::KeyMap[GLFW_KEY_W]             = FG_Keys_W;
  WindowGL::KeyMap[GLFW_KEY_X]             = FG_Keys_X;
  WindowGL::KeyMap[GLFW_KEY_Y]             = FG_Keys_Y;
  WindowGL::KeyMap[GLFW_KEY_Z]             = FG_Keys_Z;
  WindowGL::KeyMap[GLFW_KEY_LEFT_BRACKET]  = FG_Keys_LEFT_BRACKET;  // [
  WindowGL::KeyMap[GLFW_KEY_BACKSLASH]     = FG_Keys_BACKSLASH;     /* \ */
  WindowGL::KeyMap[GLFW_KEY_RIGHT_BRACKET] = FG_Keys_RIGHT_BRACKET; // ]
  WindowGL::KeyMap[GLFW_KEY_GRAVE_ACCENT]  = FG_Keys_GRAVE;         // `
  WindowGL::KeyMap[GLFW_KEY_WORLD_1]       = FG_Keys_OEM_8;         // non-US #1
  WindowGL::KeyMap[GLFW_KEY_WORLD_2]       = FG_Keys_KANJI;         // non-US #2
  WindowGL::KeyMap[GLFW_KEY_ESCAPE]        = FG_Keys_ESCAPE;
  WindowGL::KeyMap[GLFW_KEY_ENTER]         = FG_Keys_RETURN;
  WindowGL::KeyMap[GLFW_KEY_TAB]           = FG_Keys_TAB;
  WindowGL::KeyMap[GLFW_KEY_BACKSPACE]     = FG_Keys_BACK;
  WindowGL::KeyMap[GLFW_KEY_INSERT]        = FG_Keys_INSERT;
  WindowGL::KeyMap[GLFW_KEY_DELETE]        = FG_Keys_DELETE;
  WindowGL::KeyMap[GLFW_KEY_RIGHT]         = FG_Keys_RIGHT;
  WindowGL::KeyMap[GLFW_KEY_LEFT]          = FG_Keys_LEFT;
  WindowGL::KeyMap[GLFW_KEY_DOWN]          = FG_Keys_DOWN;
  WindowGL::KeyMap[GLFW_KEY_UP]            = FG_Keys_UP;
  WindowGL::KeyMap[GLFW_KEY_PAGE_UP]       = FG_Keys_PAGEUP;
  WindowGL::KeyMap[GLFW_KEY_PAGE_DOWN]     = FG_Keys_PAGEDOWN;
  WindowGL::KeyMap[GLFW_KEY_HOME]          = FG_Keys_HOME;
  WindowGL::KeyMap[GLFW_KEY_END]           = FG_Keys_END;
  WindowGL::KeyMap[GLFW_KEY_CAPS_LOCK]     = FG_Keys_CAPITAL;
  WindowGL::KeyMap[GLFW_KEY_SCROLL_LOCK]   = FG_Keys_SCROLL;
  WindowGL::KeyMap[GLFW_KEY_NUM_LOCK]      = FG_Keys_NUMLOCK;
  WindowGL::KeyMap[GLFW_KEY_PRINT_SCREEN]  = FG_Keys_PRINT;
  WindowGL::KeyMap[GLFW_KEY_PAUSE]         = FG_Keys_PAUSE;
  WindowGL::KeyMap[GLFW_KEY_F1]            = FG_Keys_F1;
  WindowGL::KeyMap[GLFW_KEY_F2]            = FG_Keys_F2;
  WindowGL::KeyMap[GLFW_KEY_F3]            = FG_Keys_F3;
  WindowGL::KeyMap[GLFW_KEY_F4]            = FG_Keys_F4;
  WindowGL::KeyMap[GLFW_KEY_F5]            = FG_Keys_F5;
  WindowGL::KeyMap[GLFW_KEY_F6]            = FG_Keys_F6;
  WindowGL::KeyMap[GLFW_KEY_F7]            = FG_Keys_F7;
  WindowGL::KeyMap[GLFW_KEY_F8]            = FG_Keys_F8;
  WindowGL::KeyMap[GLFW_KEY_F9]            = FG_Keys_F9;
  WindowGL::KeyMap[GLFW_KEY_F10]           = FG_Keys_F10;
  WindowGL::KeyMap[GLFW_KEY_F11]           = FG_Keys_F11;
  WindowGL::KeyMap[GLFW_KEY_F12]           = FG_Keys_F12;
  WindowGL::KeyMap[GLFW_KEY_F13]           = FG_Keys_F13;
  WindowGL::KeyMap[GLFW_KEY_F14]           = FG_Keys_F14;
  WindowGL::KeyMap[GLFW_KEY_F15]           = FG_Keys_F15;
  WindowGL::KeyMap[GLFW_KEY_F16]           = FG_Keys_F16;
  WindowGL::KeyMap[GLFW_KEY_F17]           = FG_Keys_F17;
  WindowGL::KeyMap[GLFW_KEY_F18]           = FG_Keys_F18;
  WindowGL::KeyMap[GLFW_KEY_F19]           = FG_Keys_F19;
  WindowGL::KeyMap[GLFW_KEY_F20]           = FG_Keys_F20;
  WindowGL::KeyMap[GLFW_KEY_F21]           = FG_Keys_F21;
  WindowGL::KeyMap[GLFW_KEY_F22]           = FG_Keys_F22;
  WindowGL::KeyMap[GLFW_KEY_F23]           = FG_Keys_F23;
  WindowGL::KeyMap[GLFW_KEY_F24]           = FG_Keys_F24;
  WindowGL::KeyMap[GLFW_KEY_F25]           = FG_Keys_F25;
  WindowGL::KeyMap[GLFW_KEY_KP_0]          = FG_Keys_NUMPAD0;
  WindowGL::KeyMap[GLFW_KEY_KP_1]          = FG_Keys_NUMPAD1;
  WindowGL::KeyMap[GLFW_KEY_KP_2]          = FG_Keys_NUMPAD2;
  WindowGL::KeyMap[GLFW_KEY_KP_3]          = FG_Keys_NUMPAD3;
  WindowGL::KeyMap[GLFW_KEY_KP_4]          = FG_Keys_NUMPAD4;
  WindowGL::KeyMap[GLFW_KEY_KP_5]          = FG_Keys_NUMPAD5;
  WindowGL::KeyMap[GLFW_KEY_KP_6]          = FG_Keys_NUMPAD6;
  WindowGL::KeyMap[GLFW_KEY_KP_7]          = FG_Keys_NUMPAD7;
  WindowGL::KeyMap[GLFW_KEY_KP_8]          = FG_Keys_NUMPAD8;
  WindowGL::KeyMap[GLFW_KEY_KP_9]          = FG_Keys_NUMPAD9;
  WindowGL::KeyMap[GLFW_KEY_KP_DECIMAL]    = FG_Keys_NUMPAD_DECIMAL;
  WindowGL::KeyMap[GLFW_KEY_KP_DIVIDE]     = FG_Keys_NUMPAD_DIVIDE;
  WindowGL::KeyMap[GLFW_KEY_KP_MULTIPLY]   = FG_Keys_NUMPAD_MULTIPLY;
  WindowGL::KeyMap[GLFW_KEY_KP_SUBTRACT]   = FG_Keys_NUMPAD_SUBTRACT;
  WindowGL::KeyMap[GLFW_KEY_KP_ADD]        = FG_Keys_NUMPAD_ADD;
  WindowGL::KeyMap[GLFW_KEY_KP_ENTER]      = FG_Keys_RETURN;
  WindowGL::KeyMap[GLFW_KEY_KP_EQUAL]      = FG_Keys_NUMPAD_EQUAL;
  WindowGL::KeyMap[GLFW_KEY_LEFT_SHIFT]    = FG_Keys_LSHIFT;
  WindowGL::KeyMap[GLFW_KEY_LEFT_CONTROL]  = FG_Keys_LCONTROL;
  WindowGL::KeyMap[GLFW_KEY_LEFT_ALT]      = FG_Keys_LMENU;
  WindowGL::KeyMap[GLFW_KEY_LEFT_SUPER]    = FG_Keys_LSUPER;
  WindowGL::KeyMap[GLFW_KEY_RIGHT_SHIFT]   = FG_Keys_RSHIFT;
  WindowGL::KeyMap[GLFW_KEY_RIGHT_CONTROL] = FG_Keys_RCONTROL;
  WindowGL::KeyMap[GLFW_KEY_RIGHT_ALT]     = FG_Keys_RMENU;
  WindowGL::KeyMap[GLFW_KEY_RIGHT_SUPER]   = FG_Keys_RSUPER;
  WindowGL::KeyMap[GLFW_KEY_MENU]          = FG_Keys_MENU;
}

WindowGL::WindowGL(Provider* backend, GLFWmonitor* display, uintptr_t window_id, FG_Vec2* pos, FG_Vec2* dim, uint64_t flags,
                   const char* caption) :
  _next(nullptr), _prev(nullptr), _joysticks(0), _backend(backend)
{
  FillKeyMap();
  if(flags & FG_WindowFlag_No_Caption)
    caption = "";

  this->window_id = window_id;

  glfwWindowHint(GLFW_DECORATED, !(flags & FG_WindowFlag_No_Border));
  glfwWindowHint(GLFW_AUTO_ICONIFY, flags & FG_WindowFlag_Minimized);
  glfwWindowHint(GLFW_RESIZABLE, flags & FG_WindowFlag_Resizable);
  glfwWindowHint(GLFW_MAXIMIZED, flags & FG_WindowFlag_Maximized);
  glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

  //#if defined(__linux__) || defined(__linux)
  //  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
  //#endif

  _window =
    glfwCreateWindow(!dim ? 0 : static_cast<int>(dim->x), !dim ? 0 : static_cast<int>(dim->y), caption, nullptr, nullptr);

  if(_window)
  {
    glfwSetWindowUserPointer(_window, this);
    glfwSetWindowCloseCallback(_window, CloseCallback);
    glfwSetKeyCallback(_window, KeyCallback);
    glfwSetCharCallback(_window, CharCallback);
    glfwSetMouseButtonCallback(_window, MouseButtonCallback);
    glfwSetCursorPosCallback(_window, MousePosCallback);
    glfwSetCursorEnterCallback(_window, EnterCallback);
    glfwSetScrollCallback(_window, ScrollCallback);
    glfwSetDropCallback(_window, DropCallback);
    glfwSetWindowFocusCallback(_window, FocusCallback);
    glfwSetWindowSizeCallback(_window, SizeCallback);
    glfwSetWindowRefreshCallback(_window, RefreshCallback);
    glfwSetJoystickCallback(JoystickCallback);

#ifdef FG_PLATFORM_WIN32
    // Remove the transparency hack by removing the layered flag that ISN'T NECESSARY >:C
    HWND hWnd     = glfwGetWin32Window(_window);
    DWORD exStyle = GetWindowLongW(hWnd, GWL_EXSTYLE);
    SetWindowLongW(hWnd, GWL_EXSTYLE, exStyle & (~WS_EX_LAYERED));

    this->handle = reinterpret_cast<uintptr_t>(glfwGetWin32Window(_window));
#endif
  }
  else
    backend->LOG(FG_Level_Error, "glfwCreateWindow failed");
}

WindowGL::~WindowGL()
{
  if(!_prev)
    _backend->_windows = _next;
  else
    _prev->_next = _next;

  if(_next)
    _next->_prev = _prev;

  if(_window)
  {
    glfwSetWindowCloseCallback(_window, nullptr);
    glfwDestroyWindow(_window);
  }
}

void WindowGL::DirtyRect(const FG_Rect* area)
{
#ifdef FG_PLATFORM_WIN32
  auto hWnd = glfwGetWin32Window(_window);

  if(!area)
    InvalidateRect(hWnd, nullptr, FALSE);
  else
  {
    RECT rect = { static_cast<LONG>(floorf(area->left)), static_cast<LONG>(floorf(area->top)),
                  static_cast<LONG>(ceilf(area->right)), static_cast<LONG>(ceilf(area->bottom)) };
    InvalidateRect(hWnd, &rect, FALSE);
  }
#else
  // TODO inject draw message into queue
#endif
}

uint8_t WindowGL::GetModKeys(int mods)
{
  uint8_t m = 0;
  if(mods & GLFW_MOD_SHIFT)
    m |= FG_ModKey_Shift;
  if(mods & GLFW_MOD_CONTROL)
    m |= FG_ModKey_Control;
  if(mods & GLFW_MOD_ALT)
    m |= FG_ModKey_Alt;
  if(mods & GLFW_MOD_SUPER)
    m |= FG_ModKey_Super;
  if(mods & GLFW_MOD_CAPS_LOCK)
    m |= FG_ModKey_Capslock;
  if(mods & GLFW_MOD_NUM_LOCK)
    m |= FG_ModKey_Numlock;
  return m;
}

void WindowGL::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  auto self          = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt         = { static_cast<uint16_t>((action == GLFW_RELEASE) ? FG_Event_Kind_KeyUp : FG_Event_Kind_KeyDown) };
  evt.keyUp.key      = KeyMap[key];
  evt.keyUp.scancode = scancode;
  evt.keyUp.modkeys  = GetModKeys(mods);

  if(action == GLFW_REPEAT)
    evt.keyUp.modkeys |= FG_ModKey_Held;

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

void WindowGL::CharCallback(GLFWwindow* window, unsigned int key)
{
  auto self           = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt          = { FG_Event_Kind_KeyChar };
  evt.keyChar.unicode = key;
  evt.keyChar.modkeys = 0;

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

void WindowGL::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  auto self         = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt        = { static_cast<uint16_t>((action == GLFW_PRESS) ? FG_Event_Kind_MouseDown : FG_Event_Kind_MouseUp) };
  evt.mouseDown.all = (1 << button);
  evt.mouseDown.button  = (1 << button);
  evt.mouseDown.modkeys = GetModKeys(mods);
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

void WindowGL::MousePosCallback(GLFWwindow* window, double x, double y)
{
  auto self       = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt      = { FG_Event_Kind_MouseMove };
  evt.mouseMove.x = static_cast<float>(x);
  evt.mouseMove.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

// Hooks the mouse event for a mouse entering or exiting the window
void WindowGL::EnterCallback(GLFWwindow* window, int entered)
{
  auto self  = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt = { static_cast<uint16_t>(entered ? FG_Event_Kind_MouseOn : FG_Event_Kind_MouseOff) };
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

// Hooks mouse wheel scroll (both dimensions)
void WindowGL::ScrollCallback(GLFWwindow* window, double xdelta, double ydelta)
{
  auto self              = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt             = { FG_Event_Kind_MouseScroll };
  evt.mouseScroll.delta  = static_cast<float>(ydelta);
  evt.mouseScroll.hdelta = static_cast<float>(xdelta);
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

// TODO: Should handle drag and drop somehow
void WindowGL::DropCallback(GLFWwindow* window, int count, const char* paths[])
{
  auto self     = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt    = { FG_Event_Kind_Drop };
  evt.drop.kind = FG_Clipboard_None;

  if(count > 0)
  {
    evt.drop.kind   = FG_Clipboard_File;
    evt.drop.target = (void*)paths[0];
    evt.drop.count  = static_cast<uint32_t>(strlen(paths[0]));
  }

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

void WindowGL::FocusCallback(GLFWwindow* window, int focused)
{
  auto self  = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg evt = { static_cast<uint16_t>(focused ? FG_Event_Kind_GotFocus : FG_Event_Kind_LostFocus) };

  self->_backend->Behavior(reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window)), evt);
}

void WindowGL::CloseCallback(GLFWwindow* window)
{
  auto self                = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  FG_Msg msg               = { FG_Event_Kind_SetWindowFlags };
  msg.setWindowFlags.flags = self->_backend->Behavior(self, FG_Msg{ FG_Event_Kind_GetWindowFlags }).getWindowFlags |
                             FG_WindowFlag_Closed;
  self->_backend->Behavior(self, msg);
  FG_Msg action         = FG_Msg{ FG_Event_Kind_Action };
  action.action.subkind = 1; // FG_WINDOW_ONCLOSE
  self->_backend->Behavior(self, action);
}

void WindowGL::SizeCallback(GLFWwindow* window, int width, int height)
{
  FG_Msg msg             = { FG_Event_Kind_SetWindowRect };
  msg.setWindowRect.rect = { 0, 0, static_cast<float>(width), static_cast<float>(height) };

  auto self = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));
  self->_backend->Behavior(self, msg);
}

void WindowGL::RefreshCallback(GLFWwindow* window)
{
  auto self = reinterpret_cast<WindowGL*>(glfwGetWindowUserPointer(window));

#ifdef FG_PLATFORM_WIN32
  // Call beginpaint and Endpaint so that assertions don't blow everything up.
  PAINTSTRUCT ps;
  HWND hWnd = glfwGetWin32Window(window);
  auto hdc  = BeginPaint(hWnd, &ps);
  EndPaint(hWnd, &ps);

  // Assertions can also call our paint method outside of our queue processing, so ignore those
  if(!self->_backend->_uictx)
    return;
#endif

  FG_Msg msg    = { FG_Event_Kind_Draw };
  auto dim      = self->GetSize();
  msg.draw.area = { 0, 0, static_cast<float>(dim.x), static_cast<float>(dim.y) };

  self->_backend->Behavior(self, msg);
}
FG_Vec2i WindowGL::GetSize() const
{
  FG_Vec2i dim;
  glfwGetFramebufferSize(_window, &dim.x, &dim.y);
  return dim;
}

uint8_t WindowGL::ScanJoysticks()
{
  uint8_t count = 0;
#ifdef FG_PLATFORM_WIN32
  HWND hWnd  = glfwGetWin32Window(_window);
  _joysticks = 0;

  for(uint8_t i = 0; i < MAXJOY; ++i)
  {
    if(joySetCapture(hWnd, i, 0, TRUE) == JOYERR_NOERROR)
    {
      ++count;
      _joysticks |= (1 << i);
      JOYCAPSA caps;
      if(joyGetDevCapsA(i, &caps, sizeof(JOYCAPSA)) == JOYERR_NOERROR)
      {
        _joycaps[i].axes      = caps.wNumAxes;
        _joycaps[i].buttons   = caps.wNumButtons;
        _joycaps[i].offset[0] = caps.wXmin;
        _joycaps[i].offset[1] = caps.wYmin;
        _joycaps[i].offset[2] = caps.wZmin;
        _joycaps[i].offset[3] = caps.wRmin;
        _joycaps[i].offset[4] = caps.wUmin;
        _joycaps[i].offset[5] = caps.wVmin;
        _joycaps[i].range[0]  = caps.wXmax - caps.wXmin;
        _joycaps[i].range[1]  = caps.wYmax - caps.wYmin;
        _joycaps[i].range[2]  = caps.wZmax - caps.wZmin;
        _joycaps[i].range[3]  = caps.wRmax - caps.wRmin;
        _joycaps[i].range[4]  = caps.wUmax - caps.wUmin;
        _joycaps[i].range[5]  = caps.wVmax - caps.wVmin;
      }
    }
  }
#endif

  return count;
}

void WindowGL::PollJoysticks()
{
#ifdef FG_PLATFORM_WIN32
  JOYINFOEX info;
  info.dwSize  = sizeof(JOYINFOEX);
  info.dwFlags = JOY_RETURNBUTTONS | JOY_RETURNCENTERED | JOY_RETURNX | JOY_RETURNY | JOY_RETURNZ | JOY_RETURNR |
                 JOY_RETURNU | JOY_RETURNV;

  for(uint16_t i = 0; i < MAXJOY; ++i)
  {
    if(!(_joysticks & (1 << i)))
      continue;

    if(joyGetPosEx(i << 8, &info) == JOYERR_NOERROR)
    {
      if(_allbuttons[i] != info.dwButtons)
      {
        FG_Msg evt     = { 0 };
        uint32_t old   = _allbuttons[i];
        _allbuttons[i] = info.dwButtons;
        uint32_t diff  = (old ^ info.dwButtons);
        uint32_t k;
        for(int j = 0; j < 32; ++j) // go through the bits and generate BUTTONUP or BUTTONDOWN events
        {
          k = (1 << j);
          if((diff & k) != 0)
          {
            evt.kind = ((info.dwButtons & k) != 0) ? FG_Event_Kind_JoyButtonDown : FG_Event_Kind_JoyButtonUp;
            evt.joyButtonDown.button = j;
            evt.joyButtonDown.index  = i;
            _backend->Behavior(this, evt);
          }
        }
      }

      uint8_t numaxes = _joycaps[i].axes;
      if(memcmp(&_alljoyaxis[i][FG_JoyAxis_X], &info.dwXpos, sizeof(unsigned long) * numaxes) != 0)
      {
        unsigned long old[MAXAXIS];
        memcpy(old, &_alljoyaxis[i][FG_JoyAxis_X], sizeof(unsigned long) * numaxes);
        memcpy(&_alljoyaxis[i][FG_JoyAxis_X], &info.dwXpos, sizeof(unsigned long) * numaxes);

        for(int j = 0; j < numaxes; ++j)
        {
          if(old[j] != _alljoyaxis[i][j])
          {
            FG_Msg evt        = { FG_Event_Kind_JoyAxis };
            evt.joyAxis.axis  = j;
            evt.joyAxis.index = i;
            evt.joyAxis.value = TranslateJoyAxis(evt.joyAxis.axis, evt.joyAxis.index);
            _backend->Behavior(this, evt);
          }
        }
      }
    }
    else
      _joysticks &= (~(1 << i)); // If it failed, remove joystick from active list
  }
#endif
}

#ifdef FG_PLATFORM_WIN32
double WindowGL::TranslateJoyAxis(uint8_t axis, uint8_t index) const
{
  uint16_t half = _joycaps[index].range[axis] / 2;
  return (((uint32_t)(_alljoyaxis[index][axis] - _joycaps[index].offset[axis]) - half) / (double)half);
}
#endif

void WindowGL::JoystickCallback(int jid, int e)
{
  // Temporary way of dealing with joysticks because GLFW doesn't give us a window pointer for the window the message was
  // sent to.
  for(WindowGL* w = Provider::_singleton->_windows; w != nullptr; w = w->_next)
  {
    w->ScanJoysticks();
  }
}

int WindowGL::MakeCurrent()
{
  Provider::_lasterr = 0;
  glfwMakeContextCurrent(_window);
  return Provider::_lasterr;
}

int WindowGL::SwapBuffers()
{
  Provider::_lasterr = 0;
  glfwSwapBuffers(_window);
  return Provider::_lasterr;
}

int WindowGL::Invalidate(FG_Rect* area)
{
#ifdef FG_PLATFORM_WIN32
  Provider::_lasterr = 0;
  auto hWnd          = glfwGetWin32Window(_window);
  if(Provider::_lasterr)
    return Provider::_lasterr;

  if(!area)
    return InvalidateRect(hWnd, nullptr, FALSE) ? 0 : -1;
  else
  {
    RECT rect = { static_cast<LONG>(floorf(area->left)), static_cast<LONG>(floorf(area->top)),
                  static_cast<LONG>(ceilf(area->right)), static_cast<LONG>(ceilf(area->bottom)) };
    return InvalidateRect(hWnd, &rect, FALSE) ? 0 : -1;
  }
#else
  if(!area)
    XClearArea(glfwGetX11Display(), glfwGetX11Window(_window), 0, 0, 0, 0, True);
  else
  {
    int rect[4] = { static_cast<int>(floorf(area->left)), static_cast<int>(floorf(area->top)),
                    static_cast<int>(ceilf(area->right)), static_cast<int>(ceilf(area->bottom)) };
    XClearArea(glfwGetX11Display(), glfwGetX11Window(_window), rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1],
               True);
  }
  return 0;
#endif
}

// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"

#ifdef FG_PLATFORM_WIN32
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include "GLFW/glfw3native.h"
#endif

using namespace GL;

uint8_t Window::KeyMap[512] = { 1, 0 };
void Window::FillKeyMap()
{
  // No point in making this threadsafe because GLFW isn't threadsafe.
  if(!Window::KeyMap[0])
    return;
  Window::KeyMap[0]                      = 0;
  Window::KeyMap[GLFW_KEY_SPACE]         = FG_Keys_SPACE;
  Window::KeyMap[GLFW_KEY_APOSTROPHE]    = FG_Keys_APOSTROPHE; /* ' */
  Window::KeyMap[GLFW_KEY_COMMA]         = FG_Keys_COMMA;      /* , */
  Window::KeyMap[GLFW_KEY_MINUS]         = FG_Keys_MINUS;
  Window::KeyMap[GLFW_KEY_PERIOD]        = FG_Keys_PERIOD;
  Window::KeyMap[GLFW_KEY_SLASH]         = FG_Keys_SLASH; /* / */
  Window::KeyMap[GLFW_KEY_0]             = FG_Keys_0;
  Window::KeyMap[GLFW_KEY_1]             = FG_Keys_1;
  Window::KeyMap[GLFW_KEY_2]             = FG_Keys_2;
  Window::KeyMap[GLFW_KEY_3]             = FG_Keys_3;
  Window::KeyMap[GLFW_KEY_4]             = FG_Keys_4;
  Window::KeyMap[GLFW_KEY_5]             = FG_Keys_5;
  Window::KeyMap[GLFW_KEY_6]             = FG_Keys_6;
  Window::KeyMap[GLFW_KEY_7]             = FG_Keys_7;
  Window::KeyMap[GLFW_KEY_8]             = FG_Keys_8;
  Window::KeyMap[GLFW_KEY_9]             = FG_Keys_9;
  Window::KeyMap[GLFW_KEY_SEMICOLON]     = FG_Keys_SEMICOLON; /* ; */
  Window::KeyMap[GLFW_KEY_EQUAL]         = FG_Keys_PLUS;      /* =+ */
  Window::KeyMap[GLFW_KEY_A]             = FG_Keys_A;
  Window::KeyMap[GLFW_KEY_B]             = FG_Keys_B;
  Window::KeyMap[GLFW_KEY_C]             = FG_Keys_C;
  Window::KeyMap[GLFW_KEY_D]             = FG_Keys_D;
  Window::KeyMap[GLFW_KEY_E]             = FG_Keys_E;
  Window::KeyMap[GLFW_KEY_F]             = FG_Keys_F;
  Window::KeyMap[GLFW_KEY_G]             = FG_Keys_G;
  Window::KeyMap[GLFW_KEY_H]             = FG_Keys_H;
  Window::KeyMap[GLFW_KEY_I]             = FG_Keys_I;
  Window::KeyMap[GLFW_KEY_J]             = FG_Keys_J;
  Window::KeyMap[GLFW_KEY_K]             = FG_Keys_K;
  Window::KeyMap[GLFW_KEY_L]             = FG_Keys_L;
  Window::KeyMap[GLFW_KEY_M]             = FG_Keys_M;
  Window::KeyMap[GLFW_KEY_N]             = FG_Keys_N;
  Window::KeyMap[GLFW_KEY_O]             = FG_Keys_O;
  Window::KeyMap[GLFW_KEY_P]             = FG_Keys_P;
  Window::KeyMap[GLFW_KEY_Q]             = FG_Keys_Q;
  Window::KeyMap[GLFW_KEY_R]             = FG_Keys_R;
  Window::KeyMap[GLFW_KEY_S]             = FG_Keys_S;
  Window::KeyMap[GLFW_KEY_T]             = FG_Keys_T;
  Window::KeyMap[GLFW_KEY_U]             = FG_Keys_U;
  Window::KeyMap[GLFW_KEY_V]             = FG_Keys_V;
  Window::KeyMap[GLFW_KEY_W]             = FG_Keys_W;
  Window::KeyMap[GLFW_KEY_X]             = FG_Keys_X;
  Window::KeyMap[GLFW_KEY_Y]             = FG_Keys_Y;
  Window::KeyMap[GLFW_KEY_Z]             = FG_Keys_Z;
  Window::KeyMap[GLFW_KEY_LEFT_BRACKET]  = FG_Keys_LEFT_BRACKET;  /* [ */
  Window::KeyMap[GLFW_KEY_BACKSLASH]     = FG_Keys_BACKSLASH;     /* \ */
  Window::KeyMap[GLFW_KEY_RIGHT_BRACKET] = FG_Keys_RIGHT_BRACKET; /* ] */
  Window::KeyMap[GLFW_KEY_GRAVE_ACCENT]  = FG_Keys_GRAVE;         /* ` */
  Window::KeyMap[GLFW_KEY_WORLD_1]       = FG_Keys_OEM_8;         /* non-US #1 */
  Window::KeyMap[GLFW_KEY_WORLD_2]       = FG_Keys_KANJI;         /* non-US #2 */
  Window::KeyMap[GLFW_KEY_ESCAPE]        = FG_Keys_ESCAPE;
  Window::KeyMap[GLFW_KEY_ENTER]         = FG_Keys_RETURN;
  Window::KeyMap[GLFW_KEY_TAB]           = FG_Keys_TAB;
  Window::KeyMap[GLFW_KEY_BACKSPACE]     = FG_Keys_BACK;
  Window::KeyMap[GLFW_KEY_INSERT]        = FG_Keys_INSERT;
  Window::KeyMap[GLFW_KEY_DELETE]        = FG_Keys_DELETE;
  Window::KeyMap[GLFW_KEY_RIGHT]         = FG_Keys_RIGHT;
  Window::KeyMap[GLFW_KEY_LEFT]          = FG_Keys_LEFT;
  Window::KeyMap[GLFW_KEY_DOWN]          = FG_Keys_DOWN;
  Window::KeyMap[GLFW_KEY_UP]            = FG_Keys_UP;
  Window::KeyMap[GLFW_KEY_PAGE_UP]       = FG_Keys_PAGEUP;
  Window::KeyMap[GLFW_KEY_PAGE_DOWN]     = FG_Keys_PAGEDOWN;
  Window::KeyMap[GLFW_KEY_HOME]          = FG_Keys_HOME;
  Window::KeyMap[GLFW_KEY_END]           = FG_Keys_END;
  Window::KeyMap[GLFW_KEY_CAPS_LOCK]     = FG_Keys_CAPITAL;
  Window::KeyMap[GLFW_KEY_SCROLL_LOCK]   = FG_Keys_SCROLL;
  Window::KeyMap[GLFW_KEY_NUM_LOCK]      = FG_Keys_NUMLOCK;
  Window::KeyMap[GLFW_KEY_PRINT_SCREEN]  = FG_Keys_PRINT;
  Window::KeyMap[GLFW_KEY_PAUSE]         = FG_Keys_PAUSE;
  Window::KeyMap[GLFW_KEY_F1]            = FG_Keys_F1;
  Window::KeyMap[GLFW_KEY_F2]            = FG_Keys_F2;
  Window::KeyMap[GLFW_KEY_F3]            = FG_Keys_F3;
  Window::KeyMap[GLFW_KEY_F4]            = FG_Keys_F4;
  Window::KeyMap[GLFW_KEY_F5]            = FG_Keys_F5;
  Window::KeyMap[GLFW_KEY_F6]            = FG_Keys_F6;
  Window::KeyMap[GLFW_KEY_F7]            = FG_Keys_F7;
  Window::KeyMap[GLFW_KEY_F8]            = FG_Keys_F8;
  Window::KeyMap[GLFW_KEY_F9]            = FG_Keys_F9;
  Window::KeyMap[GLFW_KEY_F10]           = FG_Keys_F10;
  Window::KeyMap[GLFW_KEY_F11]           = FG_Keys_F11;
  Window::KeyMap[GLFW_KEY_F12]           = FG_Keys_F12;
  Window::KeyMap[GLFW_KEY_F13]           = FG_Keys_F13;
  Window::KeyMap[GLFW_KEY_F14]           = FG_Keys_F14;
  Window::KeyMap[GLFW_KEY_F15]           = FG_Keys_F15;
  Window::KeyMap[GLFW_KEY_F16]           = FG_Keys_F16;
  Window::KeyMap[GLFW_KEY_F17]           = FG_Keys_F17;
  Window::KeyMap[GLFW_KEY_F18]           = FG_Keys_F18;
  Window::KeyMap[GLFW_KEY_F19]           = FG_Keys_F19;
  Window::KeyMap[GLFW_KEY_F20]           = FG_Keys_F20;
  Window::KeyMap[GLFW_KEY_F21]           = FG_Keys_F21;
  Window::KeyMap[GLFW_KEY_F22]           = FG_Keys_F22;
  Window::KeyMap[GLFW_KEY_F23]           = FG_Keys_F23;
  Window::KeyMap[GLFW_KEY_F24]           = FG_Keys_F24;
  Window::KeyMap[GLFW_KEY_F25]           = FG_Keys_F25;
  Window::KeyMap[GLFW_KEY_KP_0]          = FG_Keys_NUMPAD0;
  Window::KeyMap[GLFW_KEY_KP_1]          = FG_Keys_NUMPAD1;
  Window::KeyMap[GLFW_KEY_KP_2]          = FG_Keys_NUMPAD2;
  Window::KeyMap[GLFW_KEY_KP_3]          = FG_Keys_NUMPAD3;
  Window::KeyMap[GLFW_KEY_KP_4]          = FG_Keys_NUMPAD4;
  Window::KeyMap[GLFW_KEY_KP_5]          = FG_Keys_NUMPAD5;
  Window::KeyMap[GLFW_KEY_KP_6]          = FG_Keys_NUMPAD6;
  Window::KeyMap[GLFW_KEY_KP_7]          = FG_Keys_NUMPAD7;
  Window::KeyMap[GLFW_KEY_KP_8]          = FG_Keys_NUMPAD8;
  Window::KeyMap[GLFW_KEY_KP_9]          = FG_Keys_NUMPAD9;
  Window::KeyMap[GLFW_KEY_KP_DECIMAL]    = FG_Keys_NUMPAD_DECIMAL;
  Window::KeyMap[GLFW_KEY_KP_DIVIDE]     = FG_Keys_NUMPAD_DIVIDE;
  Window::KeyMap[GLFW_KEY_KP_MULTIPLY]   = FG_Keys_NUMPAD_MULTIPLY;
  Window::KeyMap[GLFW_KEY_KP_SUBTRACT]   = FG_Keys_NUMPAD_SUBTRACT;
  Window::KeyMap[GLFW_KEY_KP_ADD]        = FG_Keys_NUMPAD_ADD;
  Window::KeyMap[GLFW_KEY_KP_ENTER]      = FG_Keys_RETURN;
  Window::KeyMap[GLFW_KEY_KP_EQUAL]      = FG_Keys_NUMPAD_EQUAL;
  Window::KeyMap[GLFW_KEY_LEFT_SHIFT]    = FG_Keys_LSHIFT;
  Window::KeyMap[GLFW_KEY_LEFT_CONTROL]  = FG_Keys_LCONTROL;
  Window::KeyMap[GLFW_KEY_LEFT_ALT]      = FG_Keys_LMENU;
  Window::KeyMap[GLFW_KEY_LEFT_SUPER]    = FG_Keys_LSUPER;
  Window::KeyMap[GLFW_KEY_RIGHT_SHIFT]   = FG_Keys_RSHIFT;
  Window::KeyMap[GLFW_KEY_RIGHT_CONTROL] = FG_Keys_RCONTROL;
  Window::KeyMap[GLFW_KEY_RIGHT_ALT]     = FG_Keys_RMENU;
  Window::KeyMap[GLFW_KEY_RIGHT_SUPER]   = FG_Keys_RSUPER;
  Window::KeyMap[GLFW_KEY_MENU]          = FG_Keys_MENU;
}

Window::Window(Backend* backend, GLFWmonitor* display, FG_MsgReceiver* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags,
               const char* caption, void* context) :
  Context(backend, element, dim), _next(nullptr), _prev(nullptr)
{
  FillKeyMap();
  if(flags & FG_Window_NOCAPTION)
    caption = "";

  glfwWindowHint(GLFW_DECORATED, !(flags & FG_Window_NOBORDER));
  glfwWindowHint(GLFW_AUTO_ICONIFY, flags & FG_Window_MINIMIZED);
  glfwWindowHint(GLFW_RESIZABLE, flags & FG_Window_RESIZABLE);
  glfwWindowHint(GLFW_MAXIMIZED, flags & FG_Window_MAXIMIZED);
  glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GL_TRUE);

  //#if defined(__linux__) || defined(__linux)
  //  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
  //#endif

  _window = glfwCreateWindow(!dim ? 0 : static_cast<int>(dim->x), !dim ? 0 : static_cast<int>(dim->y), caption, NULL, NULL);

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

#ifdef FG_PLATFORM_WIN32
    // Remove the transparency hack by removing the layered flag that ISN'T NECESSARY >:C
    HWND hWnd     = glfwGetWin32Window(_window);
    DWORD exStyle = GetWindowLongW(hWnd, GWL_EXSTYLE);
    SetWindowLongW(hWnd, GWL_EXSTYLE, exStyle & (~WS_EX_LAYERED));
#endif

    glfwMakeContextCurrent(_window);
    if(!gladLoadGL(glfwGetProcAddress))
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "gladLoadGL failed");
    _backend->LogError("gladLoadGL");
    CreateResources();
  }
  else
    (*_backend->_log)(_backend->_root, FG_Level_ERROR, "glfwCreateWindow failed");
}

Window::~Window()
{
  if(!_prev)
    _backend->_windows = _next;
  else
    _prev->_next = _next;

  if(_next)
    _next->_prev = _prev;

  if(_window)
  {
    if(!glfwWindowShouldClose(_window))
      CloseCallback(_window);

    if(_initialized) // We have to call this up here before we destroy the context
      DestroyResources();
    glfwDestroyWindow(_window);
  }
}

void Window::DirtyRect(const FG_Rect* area)
{
#ifdef FG_PLATFORM_WIN32
  auto hWnd = glfwGetWin32Window(_window);

  if(!area)
    InvalidateRect(hWnd, NULL, FALSE);
  else
  {
    RECT rect = { static_cast<LONG>(floorf(area->left)), static_cast<LONG>(floorf(area->top)),
                  static_cast<LONG>(ceilf(area->right)), static_cast<LONG>(ceilf(area->bottom)) };
    InvalidateRect(hWnd, &rect, FALSE);
  }
#else
  Draw(area);
#endif
}

uint8_t Window::GetModKeys(int mods)
{
  uint8_t m = 0;
  if(mods & GLFW_MOD_SHIFT)
    m |= FG_ModKey_SHIFT;
  if(mods & GLFW_MOD_CONTROL)
    m |= FG_ModKey_CONTROL;
  if(mods & GLFW_MOD_ALT)
    m |= FG_ModKey_ALT;
  if(mods & GLFW_MOD_SUPER)
    m |= FG_ModKey_SUPER;
  if(mods & GLFW_MOD_CAPS_LOCK)
    m |= FG_ModKey_CAPSLOCK;
  if(mods & GLFW_MOD_NUM_LOCK)
    m |= FG_ModKey_NUMLOCK;
  return m;
}

void Window::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  auto self          = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt         = { static_cast<uint16_t>((action == GLFW_RELEASE) ? FG_Kind_KEYUP : FG_Kind_KEYDOWN) };
  evt.keyUp.key      = KeyMap[key];
  evt.keyUp.scancode = scancode;
  evt.keyUp.modkeys  = GetModKeys(mods);

  if(action == GLFW_REPEAT)
    evt.keyUp.modkeys |= FG_ModKey_HELD;

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::CharCallback(GLFWwindow* window, unsigned int key)
{
  auto self           = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt          = { FG_Kind_KEYCHAR };
  evt.keyChar.unicode = key;
  evt.keyChar.modkeys = 0;

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  auto self             = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt            = { static_cast<uint16_t>((action == GLFW_PRESS) ? FG_Kind_MOUSEDOWN : FG_Kind_MOUSEUP) };
  evt.mouseDown.all     = (1 << button);
  evt.mouseDown.button  = (1 << button);
  evt.mouseDown.modkeys = GetModKeys(mods);
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::MousePosCallback(GLFWwindow* window, double x, double y)
{
  auto self       = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt      = { FG_Kind_MOUSEMOVE };
  evt.mouseMove.x = static_cast<float>(x);
  evt.mouseMove.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::EnterCallback(GLFWwindow* window, int entered)
{
  auto self  = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt = { entered ? FG_Kind_MOUSEON : FG_Kind_MOUSEOFF };
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::ScrollCallback(GLFWwindow* window, double xdelta, double ydelta)
{
  auto self              = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt             = { FG_Kind_MOUSESCROLL };
  evt.mouseScroll.delta  = static_cast<float>(ydelta);
  evt.mouseScroll.hdelta = static_cast<float>(xdelta);
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  evt.mouseOn.x = static_cast<float>(x);
  evt.mouseOn.y = static_cast<float>(y);

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::DropCallback(GLFWwindow* window, int count, const char* paths[])
{
  auto self     = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt    = { FG_Kind_DROP };
  evt.drop.kind = FG_Clipboard_NONE;

  if(count > 0)
  {
    evt.drop.kind   = FG_Clipboard_FILE;
    evt.drop.target = (void*)paths[0];
    evt.drop.count  = static_cast<uint32_t>(strlen(paths[0]));
  }

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::FocusCallback(GLFWwindow* window, int focused)
{
  auto self  = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt = { static_cast<uint16_t>(focused ? FG_Kind_GOTFOCUS : FG_Kind_LOSTFOCUS) };

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::CloseCallback(GLFWwindow* window)
{
  auto self                = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg msg               = { FG_Kind_SETWINDOWFLAGS };
  msg.setWindowFlags.flags = self->_backend->Behavior(self, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags |
                             FG_Window_CLOSED;
  self->_backend->Behavior(self, msg);
  FG_Msg action         = FG_Msg{ FG_Kind_ACTION };
  action.action.subkind = 1; // FG_WINDOW_ONCLOSE
  self->_backend->Behavior(self, action);
}

void Window::SizeCallback(GLFWwindow* window, int width, int height)
{
  auto self = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  self->SetDim(FG_Vec{ (float)width, (float)height });
}

void Window::RefreshCallback(GLFWwindow* window)
{
  reinterpret_cast<Window*>(glfwGetWindowUserPointer(window))->Draw(nullptr);
}

// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"

#ifdef FG_PLATFORM_WIN32
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include "GLFW/glfw3native.h"
#endif

using namespace GL;

Window::Window(Backend* backend, GLFWmonitor* display, FG_Element* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags,
               const char* caption, void* context) :
  Context(backend, element, dim), _next(nullptr), _prev(nullptr)
{
  if(flags & FG_Window_NOCAPTION)
    caption = "";

  glfwWindowHint(GLFW_DECORATED, !(flags & FG_Window_NOBORDER));
  glfwWindowHint(GLFW_AUTO_ICONIFY, flags & FG_Window_MINIMIZED);
  glfwWindowHint(GLFW_RESIZABLE, flags & FG_Window_RESIZABLE);
  glfwWindowHint(GLFW_MAXIMIZED, flags & FG_Window_MAXIMIZED);
  glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GL_TRUE);

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
  }

#ifdef FG_PLATFORM_WIN32
  // Remove the transparency hack by removing the layered flag that ISN'T NECESSARY >:C
  HWND hWnd     = glfwGetWin32Window(_window);
  DWORD exStyle = GetWindowLongW(hWnd, GWL_EXSTYLE);
  SetWindowLongW(hWnd, GWL_EXSTYLE, exStyle & (~WS_EX_LAYERED));
#endif

  glfwMakeContextCurrent(_window);
  gladLoadGL(glfwGetProcAddress);
  _backend->LogError("gladLoadGL");
  CreateResources();
}

Window::~Window()
{
  if(!_prev)
    _backend->_windows = _next;
  else
    _prev->_next = _next;

  if(_next)
    _next->_prev = _prev;

  if(!glfwWindowShouldClose(_window))
    CloseCallback(_window);

  if(_initialized) // We have to call this up here before we destroy the context
    DestroyResources();
  glfwDestroyWindow(_window);
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
  auto self         = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg evt        = { (action == GLFW_RELEASE) ? FG_Kind_KEYUP : FG_Kind_KEYDOWN };
  evt.keyUp.code    = scancode;
  evt.keyUp.modkeys = GetModKeys(mods);

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
  FG_Msg evt            = { (action == GLFW_PRESS) ? FG_Kind_MOUSEDOWN : FG_Kind_MOUSEUP };
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
  FG_Msg evt = { focused ? FG_Kind_GOTFOCUS : FG_Kind_LOSTFOCUS };

  self->_backend->Behavior(reinterpret_cast<Window*>(glfwGetWindowUserPointer(window)), evt);
}

void Window::CloseCallback(GLFWwindow* window)
{
  auto self                = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  FG_Msg msg               = { FG_Kind_SETWINDOWFLAGS };
  msg.setWindowFlags.flags = self->_backend->Behavior(self, FG_Msg{ FG_Kind_GETWINDOWFLAGS }).getWindowFlags |
                             FG_Window_CLOSED;
  self->_backend->Behavior(self, msg);
  self->_backend->Behavior(self, FG_Msg{ FG_Kind_ACTION, 1 }); // FG_WINDOW_ONCLOSE
}
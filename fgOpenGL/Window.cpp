// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "Backend.h"

using namespace GL;

Window::Window(Backend* backend, GLFWmonitor* display, FG_Element* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags, const char* caption) : _backend(backend), _element(element)
{
  _window = glfwCreateWindow(!dim ? 0 : dim->x, !dim ? 0 : dim->y, caption, NULL, NULL);
  if(_window)
  {
    glfwSetWindowUserPointer(_window, this);
    glfwSetWindowCloseCallback(_window, CloseCallback);
    glfwSetKeyCallback(_window, KeyCallback);
  }

}

Window::~Window()
{
  glfwDestroyWindow(_window);
}

void Window::BeginDraw(const FG_Rect* area)
{
  glfwMakeContextCurrent(_window);

  GLsizei w;
  GLsizei h;
  glfwGetFramebufferSize(_window, &w, &h);
  glViewport(0, 0, w, h);

  glClear(GL_COLOR_BUFFER_BIT);

}
void Window::EndDraw()
{

  glfwSwapBuffers(_window);
}

void Window::Draw(const FG_Rect* area)
{
  FG_Msg msg = { FG_Kind_DRAW };
  msg.draw.data = _window;
  _backend->BeginDraw(_backend, _window, &msg.draw.area);
  _backend->Behavior(this, msg);
  _backend->EndDraw(_backend, _window);
}

void Window::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}
void Window::CloseCallback(GLFWwindow* window)
{
}
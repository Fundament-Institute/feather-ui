// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "../backend.h"
#include "../deps/glfw/include/GLFW/glfw3.h"
#include "khash.h"
#include <stack>
#include <vector>

namespace GL {
  class Backend;
  struct Asset;

  struct Window
  {
    Window(Backend* backend, GLFWmonitor* display, FG_Element* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags, const char* caption);
    ~Window();
    void BeginDraw(const FG_Rect* area);
    void EndDraw();
    size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
    size_t SetChar(int key, unsigned long time);
    size_t SetMouse(FG_Vec& points, FG_Kind type, unsigned char button, size_t wparam, unsigned long time);
    void Draw(const FG_Rect* area);
    inline GLFWwindow* GetWindow() { return _window; }
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CloseCallback(GLFWwindow* window);

    FG_Element* _element;
    uint64_t _flags;

  protected:
    GLFWwindow* _window;
    Backend* _backend;
  };
}

#endif

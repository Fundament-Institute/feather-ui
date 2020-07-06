// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__CONTEXT_H
#define GL__CONTEXT_H

#include "../backend.h"
#include "GLFW/glfw3.h"
#include "compiler.h"
#include "khash.h"
#include <stack>
#include <vector>

namespace GL {
  class Backend;
  struct Asset;

  struct Window
  {
    Window(Backend* backend, GLFWmonitor* display, FG_Element* element, FG_Vec* pos, FG_Vec* dim, uint64_t flags,
           const char* caption);
    ~Window();
    void BeginDraw(const FG_Rect* area);
    void EndDraw();
    size_t SetKey(uint8_t keycode, bool down, bool held, unsigned long time);
    size_t SetChar(int key, unsigned long time);
    size_t SetMouse(FG_Vec& points, FG_Kind type, unsigned char button, size_t wparam, unsigned long time);
    void Draw(const FG_Rect* area);
    inline GLFWwindow* GetWindow() { return _window; }
    static uint8_t GetModKeys(int mods);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CloseCallback(GLFWwindow* window);
    static void CharCallback(GLFWwindow* window, unsigned int key);
    static void MouseButtonCallback(GLFWwindow* window, int action, int button, int mods);
    static void MousePosCallback(GLFWwindow* window, double x, double y);
    static void EnterCallback(GLFWwindow* window, int entered);
    static void ScrollCallback(GLFWwindow* window, double xdelta, double ydelta);
    static void DropCallback(GLFWwindow* window, int count, const char* paths[]);
    static void FocusCallback(GLFWwindow* window, int focused);

    FG_Element* _element;
    uint64_t _flags;
    Window* _next;

  protected:
    GLFWwindow* _window;
    Backend* _backend;
  };
}

#endif

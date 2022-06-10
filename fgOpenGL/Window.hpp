// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__WINDOW_H
#define GL__WINDOW_H

#include "Context.hpp"

namespace GL {
  class Backend;
  struct Asset;

  struct Window : Context
  {
    Window(Backend* backend, GLFWmonitor* display, FG_Element* element, FG_Vec2* pos, FG_Vec2* dim, uint64_t flags,
           const char* caption);
    virtual ~Window();
    static uint8_t GetModKeys(int mods);
    void DirtyRect(const FG_Rect* r);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CloseCallback(GLFWwindow* window);
    static void CharCallback(GLFWwindow* window, unsigned int key);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void MousePosCallback(GLFWwindow* window, double x, double y);
    static void EnterCallback(GLFWwindow* window, int entered);
    static void ScrollCallback(GLFWwindow* window, double xdelta, double ydelta);
    static void DropCallback(GLFWwindow* window, int count, const char* paths[]);
    static void FocusCallback(GLFWwindow* window, int focused);
    static void SizeCallback(GLFWwindow* window, int width, int height);
    static void RefreshCallback(GLFWwindow* window);
    
    uint64_t _flags;
    Window* _next; // GLFW doesn't let us detect when it destroys a window so we have to do it ourselves.
    Window* _prev;

    static uint8_t KeyMap[512];
    static void FillKeyMap();
  };
}

#endif

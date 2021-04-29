// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__SHADER_H
#define GL__SHADER_H

#include "backend.h"
#include "khash.h"
#include <utility>
#include <vector>
#include <string>

struct GLFWwindow;

namespace GL {
  class Backend;

  struct Shader : FG_Shader
  {
    Shader()
    {
      parameters   = nullptr;
      n_parameters = 0;
    }
    Shader(const Shader& copy);
    Shader(Shader&& mov) :
      _pixel(std::move(mov._pixel)), _vertex(std::move(mov._vertex)), _geometry(std::move(mov._geometry))
    {
      n_parameters   = mov.n_parameters;
      parameters     = mov.parameters;
      mov.parameters = nullptr;
    }
    Shader(const char* pixel, const char* vertex, const char* geometry, std::initializer_list<FG_ShaderParameter> data);
    Shader(const char* pixel, const char* vertex, const char* geometry, FG_ShaderParameter* parameters,
           size_t n_parameters);
    ~Shader();

    // Creates the shader in the current context
    unsigned int Create(Backend* backend) const;
    // Destroys the shader from the current context
    void Destroy(Backend* backend, unsigned int shader) const;

    static GLenum GetType(const FG_ShaderParameter& param);
    static void SetUniform(Backend* backend, unsigned int shader, const char* name, GLenum type, float* data);

    Shader& operator=(const Shader& copy);
    Shader& operator=(Shader&& mov) noexcept;

  private:
    static void compile(Backend* backend, GLuint program, const char* src, int type);

    std::string _pixel;
    std::string _vertex;
    std::string _geometry;
  };
}

#endif

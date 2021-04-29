// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "BackendGL.h"
#include "glad/gl.h"
#include <string.h>
#include <memory>

using namespace GL;

Shader::Shader(const Shader& copy) : _pixel(copy._pixel), _vertex(copy._vertex), _geometry(copy._geometry)
{
  n_parameters = copy.n_parameters;
  parameters   = new FG_ShaderParameter[n_parameters];
  memcpy(parameters, copy.parameters, sizeof(FG_ShaderParameter) * n_parameters);
}

Shader::Shader(const char* pixel, const char* vertex, const char* geometry,
               std::initializer_list<FG_ShaderParameter> params) :
  _pixel(!pixel ? "" : pixel), _vertex(!vertex ? "" : vertex), _geometry(!geometry ? "" : geometry)
{
  n_parameters = params.size();
  parameters   = new FG_ShaderParameter[n_parameters];
  auto cur     = parameters;

  for(auto p = params.begin(); p != params.end(); ++p)
    *(cur++) = *p;
}

Shader::Shader(const char* pixel, const char* vertex, const char* geometry, FG_ShaderParameter* params, size_t n_params) :
  _pixel(!pixel ? "" : pixel), _vertex(!vertex ? "" : vertex), _geometry(!geometry ? "" : geometry)
{
  n_parameters = n_params;
  parameters   = new FG_ShaderParameter[n_parameters];
  memcpy(parameters, params, sizeof(FG_ShaderParameter) * n_parameters);
}

Shader::~Shader()
{
  if(parameters)
    delete[] parameters;
}

void Shader::compile(Backend* backend, GLuint program, const char* src, int type)
{
  auto shader = glCreateShader(type);
  backend->LogError("glCreateShader");
  glShaderSource(shader, 1, &src, NULL);
  backend->LogError("glShaderSource");
  glCompileShader(shader);
  backend->LogError("glCompileShader");
  GLint isCompiled = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
  if(isCompiled == GL_FALSE)
  {
    GLint l;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &l);
    std::unique_ptr<char[]> buffer(new char[l]);
    glGetShaderInfoLog(shader, l, &l, buffer.get());
    (backend->_log)(backend->_root, FG_Level_WARNING, "Shader compilation failed: %s", buffer.get());
  }
  glAttachShader(program, shader);
  backend->LogError("glAttachShader");
}

GLuint Shader::Create(Backend* backend) const
{
  auto program = glCreateProgram();
  backend->LogError("glCreateProgram");

  if(!_vertex.empty())
    compile(backend, program, _vertex.c_str(), GL_VERTEX_SHADER);
  if(!_pixel.empty())
    compile(backend, program, _pixel.c_str(), GL_FRAGMENT_SHADER);
  if(!_geometry.empty())
    compile(backend, program, _geometry.c_str(), GL_GEOMETRY_SHADER);

  glLinkProgram(program);
  backend->LogError("glLinkProgram");

  glValidateProgram(program);
  GLint status;
  glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
  if(status == GL_FALSE)
  {
    GLint l = -1;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &l);
    std::unique_ptr<char[]> buffer(new char[l]);
    glGetProgramInfoLog(program, l, &l, buffer.get());
    (backend->_log)(backend->_root, FG_Level_WARNING, "Validation failed: %s", buffer.get());
  }
  return program;
}

void Shader::Destroy(Backend* backend, unsigned int shader) const
{
  glDeleteProgram(shader);
  backend->LogError("glDeleteProgram");
}

void Shader::SetUniform(Backend* backend, unsigned int shader, const char* name, GLenum type, float* data)
{
  if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
  {
    if(name) // Names are optional for textures
    {
      auto loc = glGetUniformLocation(shader, name);
      backend->LogError("glGetUniformLocation");
      if(loc > 0)
        type = GL_TEXTURE0 + loc - 1;
    }

    glActiveTexture(type);
    backend->LogError("glActiveTexture");
    glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<GLuint*>(data));
    backend->LogError("glBindTexture");
  }
  else
  {
    auto loc = glGetUniformLocation(shader, name);
    backend->LogError("glGetUniformLocation");
    switch(type)
    {
    case GL_FLOAT_MAT2: glUniformMatrix2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT2x3: glUniformMatrix2x3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT2x4: glUniformMatrix2x4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3x2: glUniformMatrix3x2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3: glUniformMatrix3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT3x4: glUniformMatrix3x4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4x2: glUniformMatrix4x2fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4x3: glUniformMatrix4x3fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_MAT4: glUniformMatrix4fv(loc, 1, GL_FALSE, data); break;
    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
    case GL_BOOL_VEC2: glUniform2fv(loc, 1, data); break;
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
    case GL_BOOL_VEC3: glUniform3fv(loc, 1, data); break;
    case GL_FLOAT_VEC4:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
    case GL_BOOL_VEC4: glUniform4fv(loc, 1, data); break;
    case GL_DOUBLE: backend->LogError("Cannot assign GL_DOUBLE!"); break;
    default: glUniform1f(loc, data[0]); break;
    }
    backend->LogError("glUniform1f");
  }
}

GLenum Shader::GetType(const FG_ShaderParameter& param)
{
  if(param.multi > 1 && param.type != FG_ShaderType_FLOAT)
    return 0;

  switch(param.type)
  {
  case FG_ShaderType_TEXTURE: return GL_TEXTURE0 + param.length;
  case FG_ShaderType_FLOAT:
    switch(param.multi)
    {
    case 0:
    case 1:
      switch(param.length)
      {
      case 1: return GL_FLOAT;
      case 2: return GL_FLOAT_VEC2;
      case 3: return GL_FLOAT_VEC3;
      case 4: return GL_FLOAT_VEC4;
      }
      return 0;
    case 2:
      switch(param.length)
      {
      case 1: return GL_FLOAT_VEC2;
      case 2: return GL_FLOAT_MAT2;
      case 3: return GL_FLOAT_MAT2x3;
      case 4: return GL_FLOAT_MAT2x4;
      }
      return 0;
    case 3:
      switch(param.length)
      {
      case 1: return GL_FLOAT_VEC3;
      case 2: return GL_FLOAT_MAT3x2;
      case 3: return GL_FLOAT_MAT3;
      case 4: return GL_FLOAT_MAT3x4;
      }
      return 0;
    case 4:
      switch(param.length)
      {
      case 1: return GL_FLOAT_VEC4;
      case 2: return GL_FLOAT_MAT4x2;
      case 3: return GL_FLOAT_MAT4x3;
      case 4: return GL_FLOAT_MAT4;
      }
      return 0;
    }
    return 0;
  case FG_ShaderType_DOUBLE:
    if(param.length == 1)
      return GL_DOUBLE;
    return 0;
  case FG_ShaderType_HALF:
    if(param.length == 1)
      return GL_HALF_FLOAT;
    return 0;
  case FG_ShaderType_INT:
    switch(param.length)
    {
    case 1: return GL_INT;
    case 2: return GL_INT_VEC2;
    case 3: return GL_INT_VEC3;
    case 4: return GL_INT_VEC4;
    }
  case FG_ShaderType_UINT:
    switch(param.length)
    {
    case 1: return GL_UNSIGNED_INT;
    case 2: return GL_UNSIGNED_INT_VEC2;
    case 3: return GL_UNSIGNED_INT_VEC3;
    case 4: return GL_UNSIGNED_INT_VEC4;
    }
  }
  return 0;
}

Shader& Shader::operator=(const Shader& copy)
{
  if(parameters)
    delete[] parameters;
  _pixel       = copy._pixel;
  _vertex      = copy._vertex;
  _geometry    = copy._geometry;
  n_parameters = copy.n_parameters;
  parameters   = new FG_ShaderParameter[n_parameters];
  memcpy(parameters, copy.parameters, sizeof(FG_ShaderParameter) * n_parameters);
  return *this;
}

Shader& Shader::operator=(Shader&& mov) noexcept
{
  if(parameters)
    delete[] parameters;
  _pixel           = std::move(mov._pixel);
  _vertex          = std::move(mov._vertex);
  _geometry        = std::move(mov._geometry);
  n_parameters     = mov.n_parameters;
  parameters       = mov.parameters;
  mov.parameters   = nullptr;
  mov.n_parameters = 0;
  return *this;
}
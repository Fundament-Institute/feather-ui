// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "Backend.h"
#include <string.h>
#include "glad/gl.h"

using namespace GL;

namespace GL {
  KHASH_INIT2(attributes, , const char*, Shader::Data, 1, kh_str_hash_func, kh_str_hash_equal);
}

Shader::Shader(const Shader& copy)
{
  init(copy._pixel, copy._vertex, copy._geometry);

  for(khint_t i = 0; i < kh_end(_layout); ++i)
  {
    if(kh_exist(_layout, i))
      add(kh_value(_layout, i));
  }
}

Shader::Shader(const char* pixel, const char* vertex, const char* geometry, std::initializer_list<Data> data)
{
  init(pixel, vertex, geometry);
  for(auto datum : data)
    add(datum);
}

Shader::~Shader() { destruct(); }

void Shader::destruct()
{
  if(_pixel)
    delete[] _pixel;
  if(_vertex)
    delete[] _vertex;
  if(_geometry)
    delete[] _geometry;
  if(_layout)
    kh_destroy_attributes(_layout);
}

void Shader::init(const char* pixel, const char* vertex, const char* geometry)
{
  auto loader = [](char** target, const char* src) {
    if(src)
    {
      size_t len = strlen(src) + 1; // include null terminator
      *target    = new char[len];
      strncpy(*target, src, len);
    }
    else
      *target = nullptr;
  };

  loader(&_pixel, pixel);
  loader(&_vertex, vertex);
  loader(&_geometry, geometry);
  _layout = kh_init_attributes();
}

bool Shader::add(const Data& data)
{
  int r;
  khiter_t i = kh_put_attributes(_layout, data.name, &r);
  if(r > 0)
    kh_value(_layout, i) = data;

  return r <= 0; // return false if it failed or already existed
}

GLuint Shader::Create(Backend* backend) const
{
  auto program = glCreateProgram();
  backend->LogError("glCreateProgram");

  auto fn = [](Backend* backend, GLuint program, const char* src, int type) {
    if(src)
    {
      auto shader = glCreateShader(type);
      backend->LogError("glCreateShader");
      glShaderSource(shader, 1, &src, NULL);
      backend->LogError("glShaderSource");
      glCompileShader(shader);
      backend->LogError("glCompileShader");
      glAttachShader(program, shader);
      backend->LogError("glAttachShader");
    }
  };

  fn(backend, program, _vertex, GL_VERTEX_SHADER);
  fn(backend, program, _pixel, GL_FRAGMENT_SHADER);
  fn(backend, program, _geometry, GL_GEOMETRY_SHADER);

  glLinkProgram(program);
  backend->LogError("glLinkProgram");

  for(khint_t i = 0; i < kh_end(_layout); ++i)
  {
    if(kh_val(_layout, i).kind == VERTEX)
    {
      auto loc = glGetAttribLocation(program, kh_key(_layout, i));
      backend->LogError("glGetAttribLocation");
      glEnableVertexAttribArray(loc);
      backend->LogError("glEnableVertexAttribArray");
    }
  }

  glValidateProgram(program);
  GLint status;
  glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
  if(status == GL_FALSE)
  {
    char buffer[2048];
    int l;
    glGetProgramInfoLog(program, 2048, &l, buffer);
    buffer[l] = 0;
    (backend->_log)(backend->_root, FG_Level_WARNING, "Validation failed: %s", buffer);
  }
  return program;
}

void Shader::Destroy(Backend* backend, unsigned int shader) const
{
  glDeleteProgram(shader);
  backend->LogError("glDeleteProgram");
}

void Shader::SetUniform(unsigned int shader, const Attribute& attr)
{
  auto loc = glGetUniformLocation(shader, attr.name);
  switch(attr.type)
  {
  case GL_FLOAT_MAT2: return glUniformMatrix2fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT2x3: return glUniformMatrix2x3fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT2x4: return glUniformMatrix2x4fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT3x2: return glUniformMatrix3x2fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT3: return glUniformMatrix3fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT3x4: return glUniformMatrix3x4fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT4x2: return glUniformMatrix4x2fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT4x3: return glUniformMatrix4x3fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_MAT4: return glUniformMatrix4fv(loc, 1, GL_FALSE, attr.pdata);
  case GL_FLOAT_VEC2:
  case GL_INT_VEC2:
  case GL_UNSIGNED_INT_VEC2:
  case GL_BOOL_VEC2: return glUniform2fv(loc, 2, attr.data);
  case GL_FLOAT_VEC3:
  case GL_INT_VEC3:
  case GL_UNSIGNED_INT_VEC3:
  case GL_BOOL_VEC3: return glUniform3fv(loc, 3, attr.data);
  case GL_FLOAT_VEC4:
  case GL_INT_VEC4:
  case GL_UNSIGNED_INT_VEC4:
  case GL_BOOL_VEC4: return glUniform4fv(loc, 4, attr.data);
  }
  glUniform1f(loc, attr.data[0]);
}

void Shader::SetVertices(Backend* backend, unsigned int shader, size_t stride) const
{
  for(khint_t i = 0; i < kh_end(_layout); ++i)
  {
    if(kh_exist(_layout, i) && kh_val(_layout, i).kind == VERTEX)
    {
      auto loc       = glGetAttribLocation(shader, kh_key(_layout, i));
      backend->LogError("glGetAttribLocation");
      auto typecount = GetTypeCount(kh_value(_layout, i).type);
      glEnableVertexAttribArray(loc);
      glVertexAttribPointer(loc, typecount >> 24, typecount & 0x00FFFFFF, GL_FALSE, stride,
                            (void*)kh_value(_layout, i).offset);
      backend->LogError("glVertexAttribPointer");
    }
  }
}

int Shader::GetTypeCount(int type)
{
  int count = 1;
  switch(type)
  {
  case GL_FLOAT_VEC2:
    type  = GL_FLOAT;
    count = 2;
    break;
  case GL_FLOAT_MAT2:
    type  = GL_FLOAT;
    count = 2 * 2;
    break;
  case GL_FLOAT_MAT2x3:
    type  = GL_FLOAT;
    count = 2 * 3;
    break;
  case GL_FLOAT_MAT2x4:
    type  = GL_FLOAT;
    count = 2 * 4;
    break;
  case GL_FLOAT_VEC3:
    type  = GL_FLOAT;
    count = 3;
    break;
  case GL_FLOAT_MAT3x2:
    type  = GL_FLOAT;
    count = 3 * 2;
    break;
  case GL_FLOAT_MAT3:
    type  = GL_FLOAT;
    count = 3 * 3;
    break;
  case GL_FLOAT_MAT3x4:
    type  = GL_FLOAT;
    count = 3 * 4;
    break;
  case GL_FLOAT_VEC4:
    type  = GL_FLOAT;
    count = 4;
    break;
  case GL_FLOAT_MAT4x2:
    type  = GL_FLOAT;
    count = 4 * 2;
    break;
  case GL_FLOAT_MAT4x3:
    type  = GL_FLOAT;
    count = 4 * 3;
    break;
  case GL_FLOAT_MAT4:
    type  = GL_FLOAT;
    count = 4 * 4;
    break;
  case GL_INT_VEC2:
    type  = GL_INT;
    count = 2;
    break;
  case GL_INT_VEC3:
    type  = GL_INT;
    count = 3;
    break;
  case GL_INT_VEC4:
    type  = GL_INT;
    count = 4;
    break;
  case GL_UNSIGNED_INT_VEC2:
    type  = GL_UNSIGNED_INT;
    count = 2;
    break;
  case GL_UNSIGNED_INT_VEC3:
    type  = GL_UNSIGNED_INT;
    count = 3;
    break;
  case GL_UNSIGNED_INT_VEC4:
    type  = GL_UNSIGNED_INT;
    count = 4;
    break;
  case GL_BOOL_VEC2:
    type  = GL_BOOL;
    count = 2;
    break;
  case GL_BOOL_VEC3:
    type  = GL_BOOL;
    count = 3;
    break;
  case GL_BOOL_VEC4:
    type  = GL_BOOL;
    count = 4;
    break;
  }
  return type | (count << 24);
}

Shader& Shader::operator=(const Shader& copy)
{
  destruct();
  init(copy._pixel, copy._vertex, copy._geometry);

  for(khint_t i = 0; i < kh_end(_layout); ++i)
  {
    if(kh_exist(_layout, i))
      add(kh_value(_layout, i));
  }
  return *this;
}

Shader& Shader::operator=(Shader&& mov)
{
  destruct();
  _pixel        = mov._pixel;
  _vertex       = mov._vertex;
  _geometry     = mov._geometry;
  _layout       = mov._layout;
  mov._pixel    = nullptr;
  mov._vertex   = nullptr;
  mov._geometry = nullptr;
  mov._layout   = nullptr;
  return *this;
}
Attribute::Attribute(const char* n, int t, float* d) : name(n), type(t)
{
  int num = Shader::GetTypeCount(type) >> 24;
  if(num > 4)
    pdata = d;
  else
    memcpy(data, d, num * sizeof(float));
}
Attribute::~Attribute() {}
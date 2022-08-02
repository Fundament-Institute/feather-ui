// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "ShaderObject.hpp"
#include "BackendGL.hpp"
#include <cassert>

using namespace GL;

bool ShaderObject::is_valid() const noexcept
{
  if(glIsShader(_ref) == GL_FALSE)
    return false;

  GLint isCompiled = GL_FALSE;
  glGetShaderiv(_ref, GL_COMPILE_STATUS, &isCompiled);
  return isCompiled == GL_TRUE;
}

GLExpected<std::string> ShaderObject::log() const noexcept
{
  if(glIsShader(_ref) == GL_FALSE)
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "ShaderObject::Log");

  GLint len;
  glGetShaderiv(_ref, GL_INFO_LOG_LENGTH, &len); // this includes the null terminator
  GL_ERROR("glGetShaderiv");
  std::string log;
  log.resize(len);
  glGetShaderInfoLog(_ref, len, &len, log.data());
  GL_ERROR("glGetShaderInfoLog");
  log.resize(len); // this value doesn't, which is what we want
  return GLExpected<std::string>(log);
}

GLExpected<ShaderObject> ShaderObject::create(const char* src, int type) noexcept
{
  auto shader = glCreateShader(type);
  GL_ERROR("glCreateShader");
  ShaderObject obj(shader);
  glShaderSource(shader, 1, &src, NULL);
  GL_ERROR("glShaderSource");
  glCompileShader(shader);
  GL_ERROR("glCompileShader");

  return std::move(obj);
}

GLenum ShaderObject::get_type(const FG_ShaderParameter& param)
{
  if(param.width > 1 && param.type != FG_ShaderType_Float)
    return 0;

  switch(param.type)
  {
  case FG_ShaderType_Texture: return GL_TEXTURE0 + param.length;
  case FG_ShaderType_Float:
    switch(param.width)
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
  case FG_ShaderType_Double:
    if(param.length == 1)
      return GL_DOUBLE;
    return 0;
  case FG_ShaderType_Half:
    if(param.length == 1)
      return GL_HALF_FLOAT;
    return 0;
  case FG_ShaderType_Int:
    switch(param.length)
    {
    case 1: return GL_INT;
    case 2: return GL_INT_VEC2;
    case 3: return GL_INT_VEC3;
    case 4: return GL_INT_VEC4;
    }
  case FG_ShaderType_UInt:
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
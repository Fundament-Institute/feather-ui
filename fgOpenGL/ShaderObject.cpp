// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ShaderObject.hpp"
#include "ProviderGL.hpp"
#include <cassert>

using namespace GL;

GLExpected<bool> ShaderObject::is_valid() const noexcept
{
  if(auto e = CALLGL(glIsShader, _ref); e.has_error())
    return std::move(e.error());
  else if(glIsShader(_ref) == GL_FALSE)
    return false;

  GLint isCompiled = GL_FALSE;
  RETURN_ERROR(CALLGL(glGetShaderiv, _ref, GL_COMPILE_STATUS, &isCompiled));
  return isCompiled == GL_TRUE;
}

GLExpected<std::string> ShaderObject::log() const noexcept
{
  if(auto e = CALLGL(glIsShader, _ref); e.has_error())
    return std::move(e.error());
  else if(glIsShader(_ref) == GL_FALSE)
    return CUSTOM_ERROR(ERR_INVALID_REF, "ShaderObject has invalid _ref");

  GLint len;
  RETURN_ERROR(CALLGL(glGetShaderiv, _ref, GL_INFO_LOG_LENGTH, &len)); // this includes the null terminator
  std::string log;
  log.resize(len);
  RETURN_ERROR(CALLGL(glGetShaderInfoLog, _ref, len, &len, log.data()));
  log.resize(len); // this value doesn't include a null terminator, which is what we want
  return GLExpected<std::string>(log);
}

GLExpected<Owned<ShaderObject>> ShaderObject::create(const char* src, int type, Provider* backend) noexcept
{
  auto shader = glCreateShader(type);
  GL_ERROR("glCreateShader");
  Owned<ShaderObject> obj{ shader };
  RETURN_ERROR(CALLGL(glShaderSource, shader, 1, &src, nullptr));
  RETURN_ERROR(CALLGL(glCompileShader, shader));

  GLint status;
  RETURN_ERROR(CALLGL(glGetShaderiv, shader, GL_COMPILE_STATUS, &status));
  if(status == GL_FALSE)
  {
    if(auto e = obj.log())
    {
      CUSTOM_ERROR(ERR_COMPILATION_FAILURE, e.value().c_str()).log(backend);
      return CUSTOM_ERROR(ERR_COMPILATION_FAILURE, "glCompileShader");
    }
    else
      return e.error();
  }

  return obj;
}

GLenum ShaderObject::get_type(const FG_ShaderParameter& param)
{
  if(param.width > 1 && param.type != FG_Shader_Type_Float)
    return 0;

  switch(param.type)
  {
  case FG_Shader_Type_Texture: return GL_TEXTURE0 + param.length;
  case FG_Shader_Type_Float:
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
  case FG_Shader_Type_Double:
    if(param.length == 1)
      return GL_DOUBLE;
    return 0;
  case FG_Shader_Type_Half:
    if(param.length == 1)
      return GL_HALF_FLOAT;
    return 0;
  case FG_Shader_Type_Int:
    switch(param.length)
    {
    case 1: return GL_INT;
    case 2: return GL_INT_VEC2;
    case 3: return GL_INT_VEC3;
    case 4: return GL_INT_VEC4;
    }
  case FG_Shader_Type_UInt:
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
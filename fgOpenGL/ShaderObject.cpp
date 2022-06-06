// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "ShaderObject.h"
#include "BackendGL.h"
#include <assert.h>

using namespace GL;

bool ShaderObject::is_valid() const
{
  if(glIsShader(_ref) == GL_FALSE)
    return false;

  GLint isCompiled = GL_FALSE;
  glGetShaderiv(_ref, GL_COMPILE_STATUS, &isCompiled);
  return isCompiled == GL_TRUE;
}

GLExpected<std::string> ShaderObject::log() const
{
  if(glIsShader(_ref) == GL_FALSE)
    return GLError(ERR_INVALID_PARAMETER, "ShaderObject::Log");

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

GLExpected<ShaderObject> ShaderObject::create(const char* src, int type)
{
  auto shader = glCreateShader(type);
  GL_ERROR("glCreateShader");
  glShaderSource(shader, 1, &src, NULL);
  GL_ERROR("glShaderSource");
  glCompileShader(shader);
  GL_ERROR("glCompileShader");
  return ShaderObject(shader);
}

GLExpected<void> ShaderObject::set_uniform(const char* name, GLenum type, float* data)
{
  if(type >= GL_TEXTURE0 && type <= GL_TEXTURE31)
  {
    if(name) // Names are optional for textures
    {
      auto loc = glGetUniformLocation(_ref, name);
      GL_ERROR("glGetUniformLocation");
      if(loc > 0)
        type = GL_TEXTURE0 + loc - 1;
    }

    glActiveTexture(type);
    GL_ERROR("glActiveTexture");
    glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<GLuint*>(data));
    GL_ERROR("glBindTexture");
  }
  else
  {
    auto loc = glGetUniformLocation(_ref, name);
    GL_ERROR("glGetUniformLocation");
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
    case GL_DOUBLE:
      return GLError(ERR_INVALID_PARAMETER, "Cannot assign GL_DOUBLE!");
    default: glUniform1f(loc, data[0]); break;
    }
    GL_ERROR("glUniform1f");
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
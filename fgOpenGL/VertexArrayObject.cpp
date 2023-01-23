
#include "ProviderGL.hpp"
#include "glad/glad.h"
#include <cstring>
#include <memory>
#include "VertexArrayObject.hpp"
#include "EnumMapping.hpp"

using namespace GL;

#ifndef USE_EMULATED_VAOS

GLExpected<VertexArrayObject> VertexArrayObject::create(GLuint program, std::span<FG_VertexParameter> parameters,
                                                        std::span<std::pair<GLuint, GLsizei>> vbuffers,
                                                        GLuint indices) noexcept
{
  GLuint id;
  RETURN_ERROR(CALLGL(glGenVertexArrays, 1, &id));

  VertexArrayObject vao(id);
  if(GLExpected<void> e = vao.bind())
  {
    if(indices)
    {
      RETURN_ERROR(CALLGL(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, indices));
    }

    for(auto& param : parameters)
    {
      if(param.index >= vbuffers.size())
      {
        return CUSTOM_ERROR(ERR_INVALID_SHADER_INDEX, "Shader parameter index exceeds buffer count");
      }

      RETURN_ERROR(CALLGL(glBindBuffer, GL_ARRAY_BUFFER, vbuffers[param.index].first));
      auto loc = CALLGL(glGetAttribLocation, program, param.name);
      if(loc.has_error())
        return std::move(loc.error());

      GL_ERROR("glGetAttribLocation");
      RETURN_ERROR(CALLGL(glEnableVertexAttribArray, loc.value()));

      if(param.type > ArraySize(ShaderTypeMapping))
        return CUSTOM_ERROR(ERR_INVALID_ENUM, "param.type is not valid shader type");

      GLenum type   = ShaderTypeMapping[param.type];
      size_t offset = param.offset;
      RETURN_ERROR(CALLGL(glVertexAttribPointer, loc.value(), param.length, type, GL_FALSE, vbuffers[param.index].second,
                          reinterpret_cast<void*>(offset)));
      if(glVertexAttribDivisorARB)
      {
        RETURN_ERROR(CALLGL(glVertexAttribDivisorARB, loc.value(), param.step));
      }

      RETURN_ERROR(CALLGL(glBindBuffer, GL_ARRAY_BUFFER, 0));
    }
  }
  else
    return std::move(e.error());

  if(indices)
    RETURN_ERROR(CALLGL(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, 0));

  return vao;
}

VertexArrayObject::~VertexArrayObject()
{
  if(glIsVertexArray(_vaoID))
    glDeleteVertexArrays(1, &_vaoID);
}

GLExpected<void> VertexArrayObject::bind() { return CALLGL(glBindVertexArray, _vaoID); }

GLExpected<void> VertexArrayObject::unbind() { return CALLGL(glBindVertexArray, 0); }

#else

VertexArrayObject::VertexArrayObject() {}

GLExpected<VertexArrayObject> VertexArrayObject::create(GLuint program, std::span<FG_VertexParameter> parameters,
                                                        std::span<std::pair<GLuint, GLsizei>> vbuffers,
                                                        GLuint indices) noexcept
{
  VertexArrayObject vao;
  vao._indexBuffer = indices;

  vao._attribs.reserve(parameters.size());
  vao._vertexBuffer.reserve(vbuffers.size());

  for(size_t i = 0; i < vbuffers.size(); ++i)
  {
    size_t start = vao._attribs.size();

    for(auto& param : parameters)
    {
      if(param.index == i)
      {
        if(param.type > ArraySize(ShaderTypeMapping))
          return CUSTOM_ERROR(ERR_INVALID_ENUM, "param.type is not valid shader type");

        auto loc = CALLGL(glGetAttribLocation, program, param.name);
        if(loc.has_error())
          return std::move(loc.error());
        vao._attribs.push_back(VertexAttrib{ loc.value(), parameters[i].length, ShaderTypeMapping[param.type],
                                             static_cast<uint16_t>(param.step) });
      }
    }

    vao._vertexBuffer.push_back(VertexBuffer{ vbuffers[i].first, vbuffers[i].second,
                                              std::span(vao._attribs).subspan(start, vao._attribs.size() - start) });
  }

  return {};
}

VertexArrayObject::~VertexArrayObject() {}

GLExpected<void> VertexArrayObject::bind()
{
  if(_indexBuffer != 0)
  {
    RETURN_ERROR(CALLGL(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, _indexBuffer));
  }

  for(auto& buffer : _vertexBuffer)
  {
    RETURN_ERROR(CALLGL(glBindBuffer, GL_ARRAY_BUFFER, buffer.id));
    char* offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      RETURN_ERROR(CALLGL(glEnableVertexAttribArray, attrib.location));

      RETURN_ERROR(
        CALLGL(glVertexAttribPointer, attrib.location, attrib.numElements, attrib.type, GL_FALSE, buffer.stride, offset));
      if(glVertexAttribDivisorARB)
      {
        RETURN_ERROR(CALLGL(glVertexAttribDivisorARB, attrib.location, attrib.divisor));
      }

      offset += Context::GetBytes(attrib.type) * attrib.numElements;
    }
  }

  return CALLGL(glBindBuffer, GL_ARRAY_BUFFER, 0);
}

GLExpected<void> VertexArrayObject::unbind()
{
  if(_indexBuffer != 0)
  {
    RETURN_ERROR(CALLGL(glBindBuffer, GL_ELEMENT_ARRAY_BUFFER, 0));
  }

  for(auto& buffer : _vertexBuffer)
  {
    RETURN_ERROR(CALLGL(glBindBuffer, GL_ARRAY_BUFFER, buffer.id));
    GLuint offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      RETURN_ERROR(CALLGL(glDisableVertexAttribArray, attrib.location));
    }
  }

  return CALLGL(glBindBuffer, GL_ARRAY_BUFFER, 0);
}
#endif
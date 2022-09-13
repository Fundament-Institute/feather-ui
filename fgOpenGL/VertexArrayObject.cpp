
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
  glGenVertexArrays(1, &id);
  GL_ERROR("glGenVertexArrays");

  VertexArrayObject vao(id);
  if(GLExpected<void> e = vao.bind())
  {
    if(indices)
    {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
      GL_ERROR("glBindBuffer");
    }

    for(auto& param : parameters)
    {
      if(param.index >= vbuffers.size())
      {
        return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Parameter index exceeds buffer count");
      }

      glBindBuffer(GL_ARRAY_BUFFER, vbuffers[param.index].first);
      GL_ERROR("glBindBuffer");
      auto loc = glGetAttribLocation(program, param.name);
      GL_ERROR("glGetAttribLocation");
      glEnableVertexAttribArray(loc);
      GL_ERROR("glEnableVertexAttribArray");

      if(param.type > ArraySize(ShaderTypeMapping))
        return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "param.type is not valid shader type");

      GLenum type   = ShaderTypeMapping[param.type];
      size_t offset = param.offset;
      glVertexAttribPointer(loc, param.length, type, GL_FALSE, vbuffers[param.index].second,
                            reinterpret_cast<void*>(offset));
      GL_ERROR("glVertexAttribPointer");
      if(glVertexAttribDivisorARB)
      {
        glVertexAttribDivisorARB(loc, param.step);
        GL_ERROR("glVertexAttribDivisorARB");
      }

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      GL_ERROR("glBindBuffer");
    }
  }
  else
    return std::move(e.error());

  if(indices)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  return vao;
}

VertexArrayObject::~VertexArrayObject()
{
  if(glIsVertexArray(_vaoID))
    glDeleteVertexArrays(1, &_vaoID);
}

GLExpected<void> VertexArrayObject::bind()
{
  glBindVertexArray(_vaoID);
  GL_ERROR("glBindVertexArray");
  return {};
}

GLExpected<void> VertexArrayObject::unbind()
{
  glBindVertexArray(0);
  GL_ERROR("glBindVertexArray");
  return {};
}

#else

VertexArrayObject::VertexArrayObject() {}

GLExpected<VertexArrayObject> VertexArrayObject::create(GLuint shader, std::span<FG_VertexParameter> parameters,
                                                        GLuint* vbuffers, GLsizei* vstrides, size_t n_vbuffers,
                                                        GLuint indices)
{
  VertexArrayObject vao;
  vao._indexBuffer = indices;

  vao._attribs.reserve(parameters.size());
  vao._vertexBuffer.reserve(n_vbuffers);

  for(size_t i = 0; i < n_vbuffers; ++i)
  {
    size_t start = vao._attribs.size();

    for(auto& param : parameters)
    {
      if(param.index == i)
      {
        if(param.type > ArraySize(ShaderTypeMapping))
          return GLError(ERR_INVALID_PARAMETER, "param.type is not valid shader type");

        vao._attribs.push_back(VertexAttrib{ glGetAttribLocation(shader, param.name), parameters[i].length,
                                             ShaderTypeMapping[param.type], static_cast<uint16_t>(param.step) });
      }
    }

    vao._vertexBuffer.push_back(
      VertexBuffer{ vbuffers[i], vstrides[i], std::span(vao._attribs).subspan(start, vao._attribs.size() - start) });
  }
}

VertexArrayObject::~VertexArrayObject() {}

GLExpected<void> VertexArrayObject::Bind()
{
  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    GL_ERROR("glBindBuffer");
  }

  for(auto& buffer : _vertexBuffer)
  {
    glBindBuffer(GL_ARRAY_BUFFER, buffer.id);
    GL_ERROR("glBindBuffer");
    char* offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      glEnableVertexAttribArray(attrib.location);
      GL_ERROR("glEnableVertexAttribArray");

      glVertexAttribPointer(attrib.location, attrib.numElements, attrib.type, GL_FALSE, buffer.stride, offset);
      GL_ERROR("glVertexAttribPointer");
      if(glVertexAttribDivisor)
      {
        glVertexAttribDivisor(attrib.location, attrib.divisor);
        GL_ERROR("glVertexAttribDivisor");
      }
      else if(glVertexAttribDivisorARB)
      {
        glVertexAttribDivisorARB(attrib.location, attrib.divisor);
        GL_ERROR("glVertexAttribDivisorARB");
      }

      offset += Context::GetBytes(attrib.type) * attrib.numElements;
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  GL_ERROR("glBindBuffer");
}

GLExpected<void> VertexArrayObject::Unbind()
{
  if(_indexBuffer != 0)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    GL_ERROR("glBindBuffer");
  }

  for(auto& buffer : _vertexBuffer)
  {
    glBindBuffer(GL_ARRAY_BUFFER, buffer.id);
    GL_ERROR("glBindBuffer");
    GLuint offset = 0;
    for(auto& attrib : buffer.attributes)
    {
      glDisableVertexAttribArray(attrib.location);
      GL_ERROR("glEnableVertexAttribArray");
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  GL_ERROR("glBindBuffer");
}
#endif
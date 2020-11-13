
#ifndef GL__VAO_H
#define GL__VAO_H

#include "backend.h"
#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vector>
#include <utility>

namespace GL {
  class Backend;

  class VAO
  {
  private:
#ifndef USE_EMULATED_VAOS
    /// OpenGL object ID of VertexArrayObject
    GLuint _vaoID;
#else
    class VertexAttrib
    {
    public:
      /// Index for this attribute used by the GLSL vertex shader
      GLuint location;

      /// Number of elements per vertex. For example, a position attribute specifying XYZ would have numElements = 3.
      GLint numElements;

      /// OpenGL datatype of attribute elements. GL_FLOAT, GL_INT, GL_UNSIGNED_INT, etc.
      GLenum type;
    };

    size_t _n_attribs;

    VertexAttrib* _attribs;

    /// Total size in bytes of all vertex attributes
    size_t _stride;

    /// OpenGL object ID of underlying vertex buffer for the VAO
    GLuint _vertexBuffer;

    /// OpenGL object ID of underlying index buffer for the VAO
    GLuint _indexBuffer;
#endif

    Backend* _backend;

  public:
    VAO(Backend* backend, GLuint shader, const FG_ShaderParameter* parameters, size_t n_parameters, GLuint buffer,
        size_t stride, GLuint indices);
    ~VAO();

    void Bind();
    void Unbind();
  };
}

#endif

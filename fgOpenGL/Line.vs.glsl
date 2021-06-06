TXT(#version 110\n
attribute vec2 vPos;\n
uniform mat4 MVP;\n
uniform vec2 ViewPort; //Width and Height of the viewport\n
varying vec2 vLineCenter;\n
void main(void)\n
{\n
  vec4 pp = MVP * vec4(vPos.xy, 0, 1);\n
  gl_Position = pp;\n
  vLineCenter = 0.5*(pp.xy + vec2(1, 1))*ViewPort;\n
};\n
)
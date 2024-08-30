TXT(#version 110\n
uniform mat4 MVP;\n
uniform vec2 Inflate;\n
attribute vec2 vPos;\n
varying vec2 pos;

void main()\n
{\n
  pos = vPos.xy;\n
  pos -= vec2(0.5,0.5);\n
  pos *= Inflate;\n
  pos += vec2(0.5,0.5);\n
  gl_Position = MVP * vec4(pos, 0, 1);\n
}\n
)
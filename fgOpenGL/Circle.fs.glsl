TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n
const float PI = 3.14159265359;\n
\n
float fmod(float x, float d)\n
{\n
  return x - floor(x/d)*d;\n
}\n
float smootharc(vec2 arcs, float angle, float anglew)\n
{\n
    angle = fmod(angle-arcs.x+anglew, PI*2.0) - anglew; // Instead of enclosing within [0, 2PI], we shift to [-anglew, PI*2-anglew] which prevents the smoothstep from breaking\n
    return smoothstep(-anglew, anglew, angle) - smoothstep(arcs.y-anglew, arcs.y+anglew, angle);\n
}\n
float getarc(vec2 arcs, float angle, float anglew)\n
{\n
  if(arcs.y > PI) // we can only deal with an empty circle, not a full circle, so we invert the arc if it's greater than PI\n
  {\n
    arcs.x += arcs.y; // Note: this is probably mathematically equivelent to constructing something using abs()\n
    arcs.y = PI*2.0 - arcs.y;\n
    return 1.0 - smootharc(arcs, angle, anglew);\n
  }\n
  return smootharc(arcs, angle, anglew);\n
}\n
void main()\n
{\n
  float angle = atan(-pos.y + 0.5, pos.x - 0.5);\n
  float anglew = fwidth(angle)*0.5;\n
  float r = distance(pos, vec2(0.5, 0.5))*2.0;\n
  float w = fwidth(r)*0.5;\n
  float ol = (DimBorderBlur.z / DimBorderBlur.x)*2.0;\n
  \n
  float s = 1.0 - smoothstep(1.0 - ol - w, 1.0 - ol + w, r);\n
  float alpha = smoothstep(1.0+w, 1.0-w, r);\n
  vec4 fill = vec4(Fill.rgb, 1.0);\n
  vec4 edge = vec4(Outline.rgb, 1.0);\n
  gl_FragColor = (fill*Fill.a*s*getarc(Corners.xy, angle, anglew)) + (edge*Outline.a*clamp(alpha-s,0.0,1.0)*getarc(Corners.zw, angle, anglew));\n
}
)
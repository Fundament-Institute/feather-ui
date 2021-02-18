TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n

float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }\n
float linetopoint(vec2 p1, vec2 p2, vec2 p)\n
{\n
  vec2 n = p2 - p1;\n
  n = vec2(n.y, -n.x);\n
  return dot(normalize(n), p1 - p);\n
}\n
void main()\n
{\n
  vec2 d = DimBorderBlur.xy;\n
  vec2 p = pos * d;\n
  vec4 c = Corners;\n
  vec2 dist;\n
  vec2 p2 = vec2(c.w*d.x, 0.0);\n
  float r1 = linetopoint(p2, vec2(0.0, d.y), p);\n
  float r2 = -linetopoint(p2, d, p);\n
  float r = max(r1, r2);\n
  r = max(r, p.y - d.y);\n
  \n
  // Ideally we would get DPI for both height and width, but for now we just assume DPI isn't weird
  float w = fwidth(p.x) * (1.0 + DimBorderBlur.w);\n
  float s = 1.0 - linearstep(1.0 - DimBorderBlur.z - w, 1.0 - DimBorderBlur.z, r);\n
  float alpha = linearstep(1.0, 1.0 - w, r);\n
  vec4 fill = vec4(Fill.rgb, 1.0);\n
  vec4 edge = vec4(Outline.rgb, 1.0);\n
  gl_FragColor = (fill*Fill.a*s) + (edge*Outline.a*clamp(alpha - s,0.0,1.0));\n
}\n
)
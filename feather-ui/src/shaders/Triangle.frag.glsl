#version 450  
in vec2 pos;  
layout(binding=1) uniform vec4 PosDim;  
layout(binding=2) uniform vec4 DimBorderBlur;  
layout(binding=3) uniform vec4 Corners;  
layout(binding=4) uniform vec4 Fill;  
layout(binding=5) uniform vec4 Outline;  
out vec4 fragColor;

float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }  
float linetopoint(vec2 p1, vec2 p2, vec2 p)  
{  
  vec2 n = p2 - p1;  
  n = vec2(n.y, -n.x);  
  return dot(normalize(n), p1 - p);  
}  
void main()  
{  
  vec2 d = PosDim.zw;  
  vec2 p = pos * d + vec2(-0.5,0.5);  
  vec4 c = Corners;  
  vec2 dist;  
  vec2 p2 = vec2(c.w*d.x, 0.0);  
  float r1 = linetopoint(p2, vec2(0.0, d.y), p);  
  float r2 = -linetopoint(p2, d, p);  
  float r = max(r1, r2);  
  r = max(r, p.y - d.y);  
    
  // Ideally we would get DPI for both height and width, but for now we just assume DPI isn't weird
  float w = fwidth(p.x) * (1.0 + DimBorderBlur.w);  
  float s = 1.0 - linearstep(1.0 - DimBorderBlur.z - w*2.0, 1.0 - DimBorderBlur.z - w, r);  
  float alpha = linearstep(1.0 - w, 1.0 - w*2.0, r);  
  fragColor = (vec4(Fill.rgb, 1.0)*Fill.a*s) + (vec4(Outline.rgb, 1.0)*Outline.a*clamp(alpha - s,0.0,1.0));  
}  

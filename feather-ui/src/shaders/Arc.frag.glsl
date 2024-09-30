#version 450
in vec2 pos;
layout(binding=2) uniform vec4 DimBorderBlur;  
layout(binding=3) uniform vec4 Corners;  
layout(binding=4) uniform vec4 Fill;  
layout(binding=5) uniform vec4 Outline;  
out vec4 fragColor;

const float PI=3.14159265359;

float linearstep(float low,float high,float x){return clamp((x-low)/(high-low),0.,1.);}
vec2 rotate(vec2 p,float a){return vec2(p.x*cos(a)+p.y*sin(a),p.x*sin(a)-p.y*cos(a));}
vec4 sRGB(vec4 linearRGB){return vec4(1.055*pow(linearRGB.rgb,vec3(1./2.4)-.055),linearRGB.a);}
vec4 linearRGB(vec4 sRGB){return vec4(pow((sRGB.rgb+.055)/1.055,vec3(2.4)),sRGB.a);}
void main()
{
  float l=(DimBorderBlur.x+DimBorderBlur.y)*.5;
  vec2 uv=(pos*2.)-1.;
  float width=fwidth(pos.x);
  float w1=(1.+DimBorderBlur.w)*width;
  
  float border=(DimBorderBlur.z/l)*2.;// double because UV is in range [-1,1], not [0,1]
  float t=.50-(Corners.z/l)+w1*1.5;
  // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()
  // acts as a ceil() function, creating one extra half pixel.
  float r=1.-t+w1;
  
  // SDF for circle
  float d0=abs(length(uv)-r)-t+border;
  float d1=abs(length(uv)-r)-t;
  
  // SDF for lines that make up arc
  vec2 omega1=rotate(uv,Corners.x-Corners.y);
  vec2 omega2=rotate(uv,Corners.x+Corners.y);
  float d;
  
  // TODO: This cannot deal with non-integer circle radii, but it might be generalizable to those cases.
  if(abs(-omega1.y)+abs(omega2.y)<width){
    d=((Corners.y/PI)-.5)*2.*width;
  }else if(Corners.y>PI*.5){
    d=max(-omega1.y,omega2.y);
  }else{
    d=min(-omega1.y,omega2.y);
  }
  
  // Compensate for blur so the circle is still full or empty at 2pi and 0.
  d+=(clamp(Corners.y/PI,0.,1.)-.5)*2.*(DimBorderBlur.w*width)+border;
  
  float d2=d-border+w1;
  float d3=min(d,omega1.x+Corners.y)+w1;
  
  // Merge results of both SDFs
  float s=linearstep(-w1,w1,min(-d0,d2)-w1);
  float alpha=linearstep(-w1,w1,min(-d1,d3)-w1);
  
  // Output to screen
  fragColor=vec4(Fill.rgb,1)*Fill.a*s+vec4(Outline.rgb,1)*Outline.a*clamp(alpha-s,0.,1.);
}


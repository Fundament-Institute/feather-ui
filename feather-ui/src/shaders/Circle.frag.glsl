#version 450
in vec2 pos;  
layout(binding=2) uniform vec4 DimBorderBlur;  
layout(binding=3) uniform vec4 Corners;  
layout(binding=4) uniform vec4 Fill;  
layout(binding=5) uniform vec4 Outline;  
out vec4 fragColor;
const float PI = 3.14159265359;  
  
float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }  
void main()  
{  
    float l = (DimBorderBlur.x + DimBorderBlur.y) * 0.5;  
    vec2 uv = (pos*2.0) - 1.0;  
    float w1 = (1.0 + DimBorderBlur.w)*fwidth(pos.x);   
    
    float border = (DimBorderBlur.z / l) * 2.0; // double because UV is in range [-1,1], not [0,1]  
    float t = 0.50 - (Corners.x / l);  
    // We have to compensate for needing to do smoothstep starting from 0, which combined with abs()  
    // acts as a ceil() function, creating one extra half pixel.  
    float r = 1.0 - t - w1;  
    
    // SDF for circle  
    float inner = (Corners.y / l) * 2.0;  
    float d0 = abs(length(uv) - r + (border*0.5) - (inner*0.5)) - t + (border*0.5) + (inner*0.5);  
    float d1 = abs(length(uv) - r) - t;  
    float s = pow(linearstep(w1*2.0, 0.0, d0), 2.2);   
    float alpha = pow(linearstep(w1*2.0, 0.0, d1), 2.2);   
  
  // Output to screen   
  fragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));  
}



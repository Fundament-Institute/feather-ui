#version 450  
in vec2 pos;  
layout(binding=1) uniform vec4 PosDim;  
layout(binding=2) uniform vec4 DimBorderBlur;  
layout(binding=3) uniform vec4 Corners;  
layout(binding=4) uniform vec4 Fill;  
layout(binding=5) uniform vec4 Outline;  
out vec4 fragColor;

float linearstep(float low, float high, float x) { return clamp((x - low) / (high - low), 0.0, 1.0); }  
float rectangle(vec2 samplePosition, vec2 halfSize, vec4 edges) {  
    float edge = 20.0;  
    if(samplePosition.x > 0.0)  
      edge = (samplePosition.y < 0.0) ? edges.y : edges.z;  
    else  
      edge = (samplePosition.y < 0.0) ? edges.x : edges.w;  
    
    vec2 componentWiseEdgeDistance = abs(samplePosition) - halfSize + vec2(edge);  
    float outsideDistance = length(max(componentWiseEdgeDistance, 0.0));  
    float insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0);  
    return outsideDistance + insideDistance - edge;  
}  

void main()  
{  
    // Ideally we would get DPI for both height and width, but for now we just assume DPI isn't weird
    float w = fwidth(PosDim.z*pos.x) * 0.5 * (1.0 + DimBorderBlur.w);  
    vec2 uv = (pos * PosDim.zw) - (PosDim.zw * 0.5);  
    
    float dist = rectangle(uv, PosDim.zw * 0.5, Corners);  
  	float alpha = linearstep(w, -w, dist);  
    float s = linearstep(w, -w, dist + DimBorderBlur.z);   
    
    // Output to screen  
    //fragColor = vec4(dist,dist,dist,1);      
    fragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));  
}  

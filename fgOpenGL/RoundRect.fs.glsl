TXT(#version 110\n
varying vec2 pos;\n
uniform vec4 DimBorderBlur;\n
uniform vec4 Corners;\n
uniform vec4 Fill;\n
uniform vec4 Outline;\n

float rectangle(vec2 samplePosition, vec2 halfSize, vec4 edges) {\n
    float edge = 20.0;\n
    if(samplePosition.x > 0.0)\n
      edge = (samplePosition.y < 0.0) ? edges.y : edges.z;\n
    else\n
      edge = (samplePosition.y < 0.0) ? edges.x : edges.w;\n
    
    vec2 componentWiseEdgeDistance = abs(samplePosition) - halfSize + vec2(edge);\n
    float outsideDistance = length(max(componentWiseEdgeDistance, 0.0));\n
    float insideDistance = min(max(componentWiseEdgeDistance.x, componentWiseEdgeDistance.y), 0.0);\n
    return outsideDistance + insideDistance - edge;\n
}\n

void main()\n
{\n
    //float w = fwidth(((pos.x*DimBorderBlur.x + pos.y*DimBorderBlur.y) / 2.0)) + DimBorderBlur.w;\n
    float w = 0.701 + DimBorderBlur.w;\n
    vec2 uv = (pos * DimBorderBlur.xy) - (DimBorderBlur.xy * 0.5);\n
    
    float dist = rectangle(uv, DimBorderBlur.xy * 0.5, Corners);\n
  	float alpha = smoothstep(0.0 + w, 0.0 - w, dist);\n
    float s = smoothstep(-DimBorderBlur.z + w, -DimBorderBlur.z - w, dist);\n
    
    // Output to screen\n
    //gl_FragColor = vec4(dist,dist,dist,1);\n    
    gl_FragColor = (vec4(Fill.rgb, 1)*Fill.a*s) + (vec4(Outline.rgb, 1)*Outline.a*clamp(alpha - s, 0.0, 1.0));\n
}\n
)
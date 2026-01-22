#version 450

layout(location = 0) in vec2 inPos;
layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    gl_PointSize = 2.0;
    outColor = vec3(1.0, 1.0, 1.0);
}

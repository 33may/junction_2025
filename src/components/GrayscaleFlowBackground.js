import React, { useEffect, useRef } from "react";

// Minimal, aesthetic grayscale iso-line background (WebGL)
// Usage: <GrayscaleFlowBackground /> near the root of your app.
// Props let you tweak subtlety without breaking the vibe.

export default function GrayscaleFlowBackground({
  motion = 0.06,        // overall animation speed (gentle)
  scale = 1.35,         // pattern zoom (lower = bigger blobs)
  lineGap = 0.18,       // spacing between contour lines (0.12–0.28)
  lineThickness = 0.035,// thickness of lines (0.02–0.06)
  bg = 0.52,            // background gray 0..1
  ink = 0.46            // line gray 0..1 (keep close to bg for low contrast)
}) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const startRef = useRef(performance.now());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", { antialias: true, premultipliedAlpha: false });
    if (!gl) return;

    // Enable derivatives for fwidth in fragment shader
    gl.getExtension("OES_standard_derivatives");

    // --- Vertex shader (fullscreen triangle) ---
    const vs = `
      attribute vec2 aPos;
      varying vec2 vUv;
      void main(){
        vUv = (aPos + 1.0) * 0.5; // map clip-space to 0..1
        gl_Position = vec4(aPos, 0.0, 1.0);
      }
    `;

    // --- Fragment shader (fbm field -> crisp iso-lines -> subtle grayscale) ---
    const fs = `
      #extension GL_OES_standard_derivatives : enable
      precision highp float;
      varying vec2 vUv;
      uniform vec2 uRes;
      uniform float uTime;
      uniform float uScale;
      uniform float uMotion;
      uniform float uGap;      // spacing between lines
      uniform float uThick;    // line thickness
      uniform float uBg;       // background gray
      uniform float uInk;      // line gray

      // ---- Simplex noise (2D) by Ian McEwan, Ashima Arts (public domain) ----
      vec3 mod289(vec3 x){return x - floor(x * (1.0/289.0)) * 289.0;}
      vec2 mod289(vec2 x){return x - floor(x * (1.0/289.0)) * 289.0;}
      vec3 permute(vec3 x){return mod289(((x*34.0)+1.0)*x);} 
      float snoise(vec2 v){
        const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
        vec2 i = floor(v + dot(v, C.yy));
        vec2 x0 = v - i + dot(i, C.xx);
        vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1; i = mod289(i);
        vec3 p = permute(permute(i.y + vec3(0.0,i1.y,1.0)) + i.x + vec3(0.0,i1.x,1.0));
        vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
        m = m*m; m = m*m;
        vec3 x = 2.0 * fract(p * C.www) - 1.0;
        vec3 h = abs(x) - 0.5; vec3 ox = floor(x + 0.5); vec3 a0 = x - ox;
        m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
        vec3 g; g.x = a0.x*x0.x + h.x*x0.y; g.yz = a0.yz*x12.xz + h.yz*x12.yw;
        return 130.0 * dot(m, g);
      }

      float fbm(vec2 p){
        float f=0.0, a=0.5; 
        for(int i=0;i<5;i++){ f += a*snoise(p); p*=2.02; a*=0.5; }
        return f;
      }

      // Crisp iso-lines using fwidth for smooth antialiasing
      float isoLines(float v, float gap, float thick){
        float m = mod(v, gap) - gap*0.5; // center bands around 0
        float w = fwidth(v) * 1.5 + thick; // smooth edge + thickness
        return 1.0 - smoothstep(-w, w, abs(m)); // 1 = line, 0 = background
      }

      void main(){
        vec2 uv = (vUv - 0.5);
        uv.x *= uRes.x / uRes.y; // preserve aspect

        vec2 p = uv * uScale;
        float t = uTime * uMotion;
        float field = fbm(p + vec2(0.0, t*0.4)) * 0.9 + fbm(p*0.5 - vec2(t*0.25, 0.0)) * 0.1;

        float lines = isoLines(field*2.0, uGap, uThick);
        float c = mix(uBg, uInk, lines);

        // very soft vignette for depth
        float r = length(uv)*1.1; 
        c = mix(c, c*0.98, smoothstep(0.6, 1.1, r));

        gl_FragColor = vec4(vec3(c), 1.0);
      }
    `;

    const createShader = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(sh));
      }
      return sh;
    };

    const vsObj = createShader(gl.VERTEX_SHADER, vs);
    const fsObj = createShader(gl.FRAGMENT_SHADER, fs);
    const program = gl.createProgram();
    gl.attachShader(program, vsObj);
    gl.attachShader(program, fsObj);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);

    // Fullscreen triangle
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1,
        3, -1,
        -1, 3,
      ]),
      gl.STATIC_DRAW
    );
    const aPos = gl.getAttribLocation(program, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Uniform locations (no duplicates!)
    const uRes = gl.getUniformLocation(program, "uRes");
    const uTime = gl.getUniformLocation(program, "uTime");
    const uScale = gl.getUniformLocation(program, "uScale");
    const uMotion = gl.getUniformLocation(program, "uMotion");
    const uGap = gl.getUniformLocation(program, "uGap");
    const uThick = gl.getUniformLocation(program, "uThick");
    const uBg = gl.getUniformLocation(program, "uBg");
    const uInk = gl.getUniformLocation(program, "uInk");

    // Runtime sanity checks (lightweight "tests")
    console.assert(uRes !== null, "Uniform uRes should exist");
    console.assert(uTime !== null, "Uniform uTime should exist");
    console.assert(uScale !== null, "Uniform uScale should exist");
    console.assert(uMotion !== null, "Uniform uMotion should exist");
    console.assert(uGap !== null, "Uniform uGap should exist");
    console.assert(uThick !== null, "Uniform uThick should exist");
    console.assert(uBg !== null, "Uniform uBg should exist");
    console.assert(uInk !== null, "Uniform uInk should exist");

    const setSize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const { innerWidth: w, innerHeight: h } = window;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uRes, canvas.width, canvas.height);
    };

    setSize();
    window.addEventListener("resize", setSize);

    const loop = () => {
      const t = (performance.now() - startRef.current) / 1000;
      gl.uniform1f(uTime, t);
      gl.uniform1f(uScale, scale);
      gl.uniform1f(uMotion, motion);
      gl.uniform1f(uGap, lineGap);
      gl.uniform1f(uThick, lineThickness);
      gl.uniform1f(uBg, bg);
      gl.uniform1f(uInk, ink);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", setSize);
      gl.deleteBuffer(buf);
      gl.deleteProgram(program);
      gl.deleteShader(vsObj);
      gl.deleteShader(fsObj);
    };
  }, [motion, scale, lineGap, lineThickness, bg, ink]);

  // Fixed background container; inline styles so no Tailwind required
  return (
    <div
      aria-hidden
      style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, zIndex: -1 }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100vw", height: "100vh", filter: "saturate(0)" }}
      />
    </div>
  );
}

// Simple demo harness (manual test): renders some foreground text over the background
export function BackgroundDemo(){
  return (
    <div>
      <GrayscaleFlowBackground motion={0.05} scale={1.3} lineGap={0.2} lineThickness={0.03} bg={0.52} ink={0.49} />
      <div style={{ position: "relative", padding: 24 }}>
        <h1 style={{ margin: 0 }}>Foreground Content</h1>
        <p style={{ maxWidth: 560 }}>Verify that the background is smooth, low-contrast, and non-distracting while text remains fully readable on top.</p>
      </div>
    </div>
  );
}

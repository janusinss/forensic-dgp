(() => {
  'use strict';
  // Independent ambient mesh inspired by the public DARK MONO preview.
  // Decorative only: no uploaded pixels or model output enter the shader.
  const workspace = document.getElementById('workspace');
  if (!workspace || !window.WebGLRenderingContext) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const canvas = document.createElement('canvas');
  canvas.className = 'ambient-shader';
  canvas.setAttribute('aria-hidden', 'true');
  const gl = canvas.getContext('webgl', {alpha:false, antialias:false, depth:false, stencil:false, powerPreference:'low-power'});
  if (!gl) return;
  const vertexSource = 'attribute vec2 a;void main(){gl_Position=vec4(a,0.0,1.0);}';
  const fragmentSource = `
    precision mediump float;
    uniform vec2 u_resolution;
    uniform vec2 u_pointer;
    uniform float u_time;
    uniform float u_has_pointer;
    float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
    void main() {
      vec2 uv=gl_FragCoord.xy/u_resolution;
      vec2 p=uv-0.5;
      p.x*=u_resolution.x/u_resolution.y;
      float t=u_time*0.11;
      vec2 q=vec2(p.x+0.34*sin(p.y*2.1+t),p.y+0.25*cos(p.x*1.8-t*0.8));
      float a=0.5+0.5*sin(q.x*3.6+sin(q.y*2.7+t)+t);
      float b=0.5+0.5*cos(q.y*4.0-sin(q.x*2.5-t*0.7)-t*0.8);
      float c=0.5+0.5*sin((q.x+q.y)*2.7+t*0.6);
      vec3 base=vec3(0.025,0.06,0.055);
      vec3 blue=vec3(0.035,0.26,0.18);
      vec3 deep=vec3(0.025,0.11,0.09);
      vec3 ice=vec3(0.27,0.55,0.41);
      vec3 color=mix(base,blue,smoothstep(0.17,0.88,a*b));
      color=mix(color,deep,0.34*c);
      color+=ice*pow(clamp(a*(1.0-b),0.0,1.0),3.0)*0.32;
      vec2 mouse=vec2(u_pointer.x,1.0-u_pointer.y);
      float spot=exp(-dot((uv-mouse)*vec2(u_resolution.x/u_resolution.y,1.0),(uv-mouse)*vec2(u_resolution.x/u_resolution.y,1.0))*5.0);
      color+=vec3(0.035,0.085,0.055)*spot*u_has_pointer;
      float grain=(hash(floor(gl_FragCoord.xy))-0.5)*0.035;
      gl_FragColor=vec4(clamp(color+grain,0.0,1.0),1.0);
    }`;
  function compile(type, source) {
    const shader=gl.createShader(type); gl.shaderSource(shader,source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) { gl.deleteShader(shader); return null; }
    return shader;
  }
  const vs=compile(gl.VERTEX_SHADER,vertexSource), fs=compile(gl.FRAGMENT_SHADER,fragmentSource);
  if (!vs || !fs) return;
  const program=gl.createProgram(); gl.attachShader(program,vs); gl.attachShader(program,fs); gl.linkProgram(program);
  gl.deleteShader(vs); gl.deleteShader(fs);
  if (!gl.getProgramParameter(program,gl.LINK_STATUS)) { gl.deleteProgram(program); return; }
  gl.useProgram(program);
  const buffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
  const position=gl.getAttribLocation(program,'a'); gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
  const resolution=gl.getUniformLocation(program,'u_resolution'), timer=gl.getUniformLocation(program,'u_time');
  const pointer=gl.getUniformLocation(program,'u_pointer'), hasPointer=gl.getUniformLocation(program,'u_has_pointer');
  let host=null, raf=0, last=0, elapsed=0, active=false, disposed=false, lost=false, pointerX=0.5,pointerY=0.5,pointerOn=0;
  function resize() {
    if (!host) return;
    const rect=host.getBoundingClientRect();
    const dpr=Math.min(window.devicePixelRatio||1,1,Math.sqrt(1200000/Math.max(1,rect.width*rect.height)));
    const w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));
    if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  }
  function paint(now) {
    if (!active || disposed || lost) return;
    if (!motion.matches && last && now-last<1000/24) { raf=requestAnimationFrame(paint); return; }
    if (last && !motion.matches) elapsed+=Math.min((now-last)/1000,0.1);
    last=now;gl.viewport(0,0,canvas.width,canvas.height);
    gl.uniform2f(resolution,canvas.width,canvas.height);gl.uniform2f(pointer,pointerX,pointerY);
    gl.uniform1f(hasPointer,pointerOn);gl.uniform1f(timer,motion.matches?0:elapsed);
    gl.drawArrays(gl.TRIANGLES,0,6);
    raf=motion.matches?0:requestAnimationFrame(paint);
  }
  function evaluate() {
    if(disposed)return;
    const screen=workspace.querySelector('.work-screen.is-active');
    const name=screen&&screen.dataset.screen;
    const candidate=['inspect','options','run'].includes(name)?screen.querySelector('.work-visual'):null;
    const visible=!!candidate&&!document.hidden&&candidate.getClientRects().length>0;
    if(candidate!==host){if(host) resizeObserver.unobserve(host);host=candidate;if(host){host.prepend(canvas);resizeObserver.observe(host);}}
    if(!visible || lost){active=false;if(raf)cancelAnimationFrame(raf);raf=0;last=0;return;}
    active=true;if(raf)cancelAnimationFrame(raf);raf=0;last=0;resize();paint(performance.now());
  }
  const resizeObserver=new ResizeObserver(()=>{if(active){resize();if(motion.matches)paint(performance.now());}});
  const observer=new MutationObserver(evaluate);
  observer.observe(workspace,{subtree:true,attributes:true,attributeFilter:['class','hidden']});
  function move(event) { if(!active||!host||motion.matches)return;const r=host.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom){pointerOn=0;return;}pointerX=(event.clientX-r.left)/r.width;pointerY=(event.clientY-r.top)/r.height;pointerOn=1; }
  document.addEventListener('pointermove',move,{passive:true});
  document.addEventListener('visibilitychange',evaluate);
  if(motion.addEventListener)motion.addEventListener('change',evaluate);else motion.addListener(evaluate);
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();lost=true;active=false;if(raf)cancelAnimationFrame(raf);canvas.style.display='none';});
  // A lost context falls back to the CSS background for this page session.
  window.addEventListener('pagehide',()=>{disposed=true;if(raf)cancelAnimationFrame(raf);observer.disconnect();resizeObserver.disconnect();},{once:true});
  evaluate();
})();

/* A frozen, clean frame becomes a reviewed Picture. No FAN or person tracking. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const {makeView,pointerToSource} = window.captureGeometry;
  let mode='picture', phase='review', view, cancelDecode;
  const video = $('capture-video'), canvas = $('capture-canvas'), ctx = canvas.getContext('2d');
  const clean = document.createElement('canvas'), cleanCtx = clean.getContext('2d');
  let tokenPromise, media, fileUrl, streamId, streamTimer, streamImage, latestBlob, frameId, rect, faces = [];
  let revision = 0, frozen = false, pending = false, drawing = false, gesture;
  let savedUpload, fileReady=false;
  const status = text => { $('capture-status').textContent = text; };
  const error = text => { $('capture-error').textContent = text; $('capture-error').classList.toggle('hidden', !text); };
  const kind = () => $('capture-source').value;
  const sourceReady = () => media || fileUrl || streamId;
  const validRect = () => rect && [rect.x, rect.y, rect.width, rect.height].every(Number.isInteger) && rect.x >= 0 && rect.y >= 0 && rect.width >= 16 && rect.height >= 16 && rect.width <= 1600 && rect.height <= 1600 && rect.x + rect.width <= clean.width && rect.y + rect.height <= clean.height;
  const update = () => {
    $('capture-freeze').disabled = pending || frozen || !sourceReady() || (kind()==='video'&&(video.seeking||video.readyState<2));
    $('capture-resume').disabled = pending || !frozen || !sourceReady();
    $('capture-stop').disabled = !sourceReady() && !pending;
    $('capture-use').disabled = pending || window.restorationInput?.busy || !frameId || !validRect();
    $('capture-start').disabled = pending;
    $('capture-source').disabled = pending;
    $('capture-video-file').disabled = false;
    $('capture-freeze').classList.toggle('hidden',frozen);
    $('capture-resume').classList.toggle('hidden',!frozen);
    $('capture-selection').classList.toggle('hidden',!frameId);
    $('capture-start').classList.toggle('hidden',kind()!=='webcam');
    const screen=document.querySelector('[data-screen=inspect]');
    screen.dataset.captureState=frozen?'frozen':sourceReady()?'live':'idle';
    $('capture-source-settings').classList.toggle('hidden',frozen);
    $('capture-source-summary').classList.toggle('hidden',!frozen);
    $('capture-source-name').textContent=kind()==='video'?($('capture-video-file').files[0]?.name||'Video file'):kind()==='webcam'?'Webcam':'RTSP camera';
    $('capture-source-name').title=$('capture-source-name').textContent;
    $('capture-timeline').classList.toggle('hidden',frozen||kind()!=='video'||!fileUrl);
    document.querySelector('.capture-transport').classList.toggle('hidden',!sourceReady()&&!pending);
    video.controls=mode==='video'&&phase==='capture'&&kind()==='video'&&!!fileUrl&&!pending&&!frozen;
    video.dispatchEvent(new CustomEvent('capture:pending',{detail:{pending}}));
    $('capture-precise').disabled=pending||!validRect();
    $('capture-change-frame').disabled=pending;
    $('capture-full-frame').disabled=pending;
    $('capture-faces').querySelectorAll('button,select').forEach(control=>{control.disabled=pending;});
    for(const key of ['x','y','width','height'])$('capture-'+key).disabled=pending;

  };
  async function request(path, options = {}) {
    if(!tokenPromise)tokenPromise=(async()=>{
      const response=await fetch('/capture/status',{cache:'no-store'});
      if(!response.ok)throw new Error('The capture service is unavailable. Restart the app.');
      return (await response.json()).token;
    })().catch(problem=>{tokenPromise=undefined;throw problem;});
    const requestToken=await tokenPromise;
    const response = await fetch('/capture' + path, {...options,headers:{...options.headers,'X-Capture-Token':requestToken}});
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(typeof problem.detail === 'string' ? problem.detail : 'The capture request stopped. Check the source and try again.');
    }
    return response;
  }
  const json = value => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
  function draw() {
    if (!clean.width) return;
    if(!view)view=makeView(clean.width,clean.height);
    if(canvas.width!==view.width||canvas.height!==view.height){canvas.width=view.width;canvas.height=view.height;}
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(clean,view.x,view.y,view.width,view.height,0,0,view.width,view.height);
    ctx.save();ctx.translate(-view.x,-view.y);
    const line = Math.max(1, view.width / 300); ctx.lineWidth = line;
    faces.forEach((box,index) => { ctx.strokeStyle='#b8d9e5'; ctx.strokeRect(...box); ctx.font = `${Math.max(10,view.width/45)}px sans-serif`; ctx.fillStyle='#101d1d'; ctx.fillRect(box[0],box[1],line*16,line*10); ctx.fillStyle='#eaf2ea'; ctx.fillText(String(index+1),box[0]+line*2,box[1]+line*8); });
    if (rect) { ctx.strokeStyle='#dbec87'; ctx.lineWidth=line*1.5; ctx.strokeRect(rect.x,rect.y,rect.width,rect.height); ctx.fillStyle='#dbec87'; const handle=line*5;ctx.fillRect(rect.x+rect.width-handle,rect.y+rect.height-handle,handle,handle); }
    ctx.restore();
  }
  function fullFrame() {
    if(!clean.width)return;view=makeView(clean.width,clean.height);$('capture-full-frame').setAttribute('aria-pressed','true');$('capture-view-label').textContent='Full frame';draw();
  }
  function setRect(next) {
    rect=next;
    if(!drawing)$('capture-crop-controls').classList.remove('hidden');
    for (const key of ['x','y','width','height']) $('capture-'+key).value=rect[key];
    $('capture-crop-size').textContent = validRect() ? `${rect.width} x ${rect.height} pixels. Drag to move or resize.` : 'Keep the crop inside the frame, 16-1600 pixels per side.';
    draw(); update();
  }
  function selectFace(index) {
    if(pending)return;

    const [x,y,width,height]=faces[index];
    const margin=Math.round(Math.max(width,height)*.15);
    const left=Math.max(0,x-margin), top=Math.max(0,y-margin);
    const selected={x:left,y:top,width:Math.min(1600,clean.width-left,width+margin*2),height:Math.min(1600,clean.height-top,height+margin*2)};
    view=makeView(clean.width,clean.height,selected);
    $('capture-full-frame').setAttribute('aria-pressed','false');$('capture-view-label').textContent=`Face ${index+1} selected. Preview zoomed for crop adjustment.`;
    setRect(selected);
    [...$('capture-faces').children].filter(child=>child.tagName==='BUTTON').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
    if($('capture-face-select'))$('capture-face-select').value=String(index);
  }
  async function stop() {
    revision++;cancelDecode?.();cancelDecode=undefined;pending=false;fileReady=false; clearTimeout(streamTimer); streamTimer=undefined;
    media?.getTracks().forEach(track=>track.stop()); media=undefined;
    video.pause(); video.srcObject=null; video.removeAttribute('src'); video.load();
    if(fileUrl)URL.revokeObjectURL(fileUrl);fileUrl=undefined;
    const old=streamId;streamId=undefined;latestBlob=undefined;streamImage?.close?.();streamImage=undefined;
    const image=$('capture-stream');if(image.dataset.blobUrl)URL.revokeObjectURL(image.dataset.blobUrl);delete image.dataset.blobUrl;image.removeAttribute('src');
    if(old)request('/streams/'+old,{method:'DELETE',keepalive:true}).catch(()=>error('Stream cleanup could not be confirmed. The local idle timeout will stop it.'));
    $('capture-video').classList.add('hidden');$('capture-stream').classList.add('hidden');$('capture-timeline').classList.add('hidden');
    if(!frozen)$('capture-empty').classList.remove('hidden');
    update();
  }
  // Temporary navigation preserves opened files; live/incomplete sources still stop.
  function suspendSource() {
    if(kind()!=='video'||!fileUrl||!fileReady){stop();return;}
    revision++;pending=false;video.pause();update();
  }
  function clearFrame() {
    drawing=false;gesture=undefined;frozen=false;view=undefined;frameId=undefined;rect=undefined;faces=[];clean.width=clean.height=0;canvas.width=canvas.height=0;
    $('capture-canvas').classList.add('hidden');$('capture-crop-controls').classList.add('hidden');$('capture-faces').replaceChildren();
  }
  async function pollStream(version) {
    if(version!==revision||!streamId)return;
    try {
      const response=await request('/streams/'+streamId+'/frame');
      if(version!==revision)return;
      if(response.status===200){const blob=await response.blob();const image=await createImageBitmap(blob);if(version!==revision){image.close();return;} streamImage?.close?.();streamImage=image;latestBlob=blob;
        const displayUrl=URL.createObjectURL(blob);const img=$('capture-stream');const old=img.dataset.blobUrl;img.src=displayUrl;img.dataset.blobUrl=displayUrl;if(old)URL.revokeObjectURL(old);
        if(!frozen){img.classList.remove('hidden');$('capture-empty').classList.add('hidden');status('Live camera. Freeze a frame when the face is visible.');}}
      streamTimer=setTimeout(()=>pollStream(version),frozen?1000:180);
    }catch(problem){if(version===revision){error(problem.message);await stop();status('Camera stream stopped.');}}
  }
  async function start() {
    const selectedKind=kind(),selectedFile=$('capture-video-file').files[0],sourceFocus=document.activeElement;
    stop();clearFrame();error('');pending=true;update();const version=revision;
    try {
      if(selectedKind==='webcam') {
        if(!navigator.mediaDevices?.getUserMedia)throw new Error('Webcam access requires this local app or a secure browser address.');
        const tracks=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
        if(version!==revision){tracks.getTracks().forEach(t=>t.stop());return;}
        media=tracks;video.srcObject=media;await video.play();if(version!==revision)return;
        media.getVideoTracks()[0]?.addEventListener('ended',()=>{if(media===tracks&&version===revision){error('Webcam disconnected. Start it again to continue.');stop();}});
      } else if(selectedKind==='video') {
        const file=selectedFile;
        if(!file||!file.size||file.size>500*1024**2)throw new Error('Choose a video file up to 500 MB.');
        fileUrl=URL.createObjectURL(file);video.src=fileUrl;
        await new Promise((resolve,reject)=>{
          let settled=false;const timer=setTimeout(()=>finish(new Error('The video could not open within 15 seconds. Choose another file.')),15000);
          const finish=problem=>{if(settled)return;settled=true;clearTimeout(timer);video.onloadeddata=null;video.onerror=null;cancelDecode=undefined;problem?reject(problem):resolve();};
          cancelDecode=()=>finish(new Error('Video opening cancelled.'));
          video.onloadeddata=()=>finish();video.onerror=()=>finish(new Error('This browser cannot decode the video. Try MP4 or WebM.'));video.load();
        });
        if(version!==revision)return;
        if(!Number.isFinite(video.duration)||video.duration<=0||video.duration>86400)throw new Error('Choose a video with a readable duration up to 24 hours.');
        // Some H.264 files report loadeddata before Chrome has produced a frame.
        // A tiny seek decodes the first frame before capture becomes available.
        if(video.getVideoPlaybackQuality?.().totalVideoFrames===0){
          await new Promise((resolve,reject)=>{
            let settled=false;const timer=setTimeout(()=>finish(new Error('The first video frame could not be decoded. Try another file.')),15000);
            const finish=problem=>{if(settled)return;settled=true;clearTimeout(timer);video.onseeked=null;video.onerror=null;cancelDecode=undefined;problem?reject(problem):resolve();};
            cancelDecode=()=>finish(new Error('Video opening cancelled.'));video.onseeked=()=>finish();video.onerror=()=>finish(new Error('The video frame could not be decoded.'));
            video.currentTime=Math.min(.001,video.duration/2);
          });
          if(version!==revision)return;
        }
        fileReady=true;
        $('capture-quality').textContent=`Original · ${video.videoWidth} × ${video.videoHeight}`;
        $('capture-quality').title='Original source resolution. This file has no alternate quality levels.';
        $('capture-timeline').classList.remove('hidden');
      } else {
        const connection={url:$('capture-url').value.trim(),username:$('capture-user').value,password:$('capture-password').value};
        $('capture-password').value='';
        const result=await(await request('/streams',json(connection))).json();
        if(version!==revision){await request('/streams/'+result.stream_id,{method:'DELETE'});return;}
        streamId=result.stream_id;$('capture-password').value='';pollStream(version);status('Connecting to the local camera.');
      }
      if(version!==revision)return;
      if(selectedKind!=='rtsp'){$('capture-video').classList.remove('hidden');$('capture-empty').classList.add('hidden');status('Find a frame with a visible face, then select Freeze frame.');}
      if(selectedKind==='video'&&document.activeElement===sourceFocus&&(sourceFocus===$('capture-video-file')||sourceFocus===document.body))video.focus({preventScroll:true});
    }catch(problem){if(version===revision){stop();error(problem.name==='NotAllowedError'?'Webcam access was denied. Allow it in the browser or use a video file.':problem.message);}}
    finally{if(version===revision){pending=false;update();}}
  }
  async function freeze() {
    if(pending||frozen)return;
    if(kind()==='video'&&(video.seeking||video.readyState<2)){error('Wait for the video frame to finish loading or seeking before freezing.');return;}
    const source=kind()==='rtsp'?streamImage:video;
    const width=kind()==='rtsp'?source?.width:video.videoWidth,height=kind()==='rtsp'?source?.height:video.videoHeight;
    if(!width||!height){error('Wait until a video frame is available.');return;}
    if(width>4096||height>4096||width*height>8388608){error('Use a source frame up to 4096 pixels per side and 8 megapixels.');return;}
    const sourceKind=kind(),sourceTime=video.currentTime;
    const wasPlaying=!video.paused;video.pause();pending=true;error('');update();const version=revision;
    clean.width=width;clean.height=height;cleanCtx.drawImage(source,0,0);canvas.width=width;canvas.height=height;
    status('Checking the frozen frame for face boxes.');
    try {
      const blob=await new Promise(resolve=>clean.toBlob(resolve,'image/png'));
      if(version!==revision)return;
      if(!blob||blob.size>24*1024**2)throw new Error('The frozen frame exceeds 24 MB. Choose a smaller source.');
      const data=new FormData();data.append('file',blob,'frozen-frame.png');data.append('source_kind',sourceKind);if(sourceKind==='video')data.append('source_time',String(sourceTime));
      const result=await(await request('/frames',{method:'POST',body:data})).json();if(version!==revision)return;
      frameId=result.frame_id;faces=result.faces.slice().sort((a,b)=>b[2]*b[3]-a[2]*a[3]);frozen=true;view=makeView(clean.width,clean.height);$('capture-full-frame').setAttribute('aria-pressed','true');$('capture-view-label').textContent='Full frame';
      $('capture-video').classList.add('hidden');$('capture-stream').classList.add('hidden');$('capture-canvas').classList.remove('hidden');
      $('capture-faces').replaceChildren();
      if(faces.length>2){
        const select=document.createElement('select');select.id='capture-face-select';select.setAttribute('aria-label','Detected faces');
        const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='Choose a face';select.append(placeholder);
        faces.forEach((box,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent='Face '+(index+1);select.append(option);});
        select.addEventListener('change',()=>{if(select.value!=='')selectFace(Number(select.value));});$('capture-faces').append(select);
      }else faces.forEach((box,index)=>{const button=document.createElement('button');button.type='button';button.textContent='Face '+(index+1);button.setAttribute('aria-pressed','false');button.addEventListener('click',()=>selectFace(index));$('capture-faces').append(button);});
      draw();status(result.detection_warning|| (faces.length?'Select a face box or drag on the preview to crop. Press Enter there to create a crop with the keyboard.':'No face boxes found. Drag on the preview or press Enter there to create a crop.'));
    }catch(problem){if(version===revision){error(problem.message);if(wasPlaying)video.play().catch(()=>status('Press Play or Freeze frame to retry.'));}}
    finally{if(version===revision){pending=false;update();}}
  }
  function point(event) {return pointerToSource(event.clientX,event.clientY,canvas.getBoundingClientRect(),view||makeView(clean.width,clean.height));}
  canvas.addEventListener('pointerdown',event=>{
    if(!frameId||pending)return;const p=point(event);error('');
    const resize=rect&&Math.abs(p.x-rect.x-rect.width)<20/p.scale&&Math.abs(p.y-rect.y-rect.height)<20/p.scale;
    const move=rect&&p.x>=rect.x&&p.x<=rect.x+rect.width&&p.y>=rect.y&&p.y<=rect.y+rect.height;
    if(!resize&&!move){const index=faces.findIndex(([x,y,w,h])=>p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h);if(index>=0){selectFace(index);return;}}
    gesture={start:p,rect:rect?{...rect}:undefined,mode:resize?'resize':move?'move':'draw'};drawing=true;canvas.setPointerCapture(event.pointerId);
    if(gesture.mode==='draw')setRect({x:p.x,y:p.y,width:0,height:0});
  });
  canvas.addEventListener('pointermove',event=>{
    if(!drawing)return;const p=point(event),s=gesture.start,r=gesture.rect;
    if(gesture.mode==='move')setRect({...r,x:Math.max(0,Math.min(clean.width-r.width,r.x+p.x-s.x)),y:Math.max(0,Math.min(clean.height-r.height,r.y+p.y-s.y))});
    else if(gesture.mode==='resize')setRect({...r,width:Math.max(16,Math.min(1600,clean.width-r.x,p.x-r.x)),height:Math.max(16,Math.min(1600,clean.height-r.y,p.y-r.y))});
    else setRect({x:Math.min(s.x,p.x),y:Math.min(s.y,p.y),width:Math.min(1600,Math.abs(p.x-s.x)),height:Math.min(1600,Math.abs(p.y-s.y))});
  });
  for(const name of ['pointerup','pointercancel'])canvas.addEventListener(name,()=>{drawing=false;if(rect)setRect(rect);});
  canvas.addEventListener('keydown',event=>{
    if(pending||!frameId)return;
    if(event.key==='Enter'&&!rect){
      event.preventDefault();
      if(clean.width<16||clean.height<16){error('Use a source frame at least 16 pixels per side to create a crop.');return;}
      const width=Math.min(clean.width,1600,Math.max(16,Math.round(clean.width/2)));
      const height=Math.min(clean.height,1600,Math.max(16,Math.round(clean.height/2)));
      setRect({x:Math.floor((clean.width-width)/2),y:Math.floor((clean.height-height)/2),width,height});
      status('Crop created. Use arrow keys to move it, Shift and arrow keys to resize, or adjust it precisely.');
      return;
    }
    if(!rect||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();
    const next={...rect},dx=event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0,dy=event.key==='ArrowDown'?1:event.key==='ArrowUp'?-1:0;
    if(event.shiftKey){next.width=Math.max(16,Math.min(1600,clean.width-next.x,next.width+dx));next.height=Math.max(16,Math.min(1600,clean.height-next.y,next.height+dy));}
    else{next.x=Math.max(0,Math.min(clean.width-next.width,next.x+dx));next.y=Math.max(0,Math.min(clean.height-next.height,next.y+dy));}setRect(next);
  });
  for(const key of ['x','y','width','height'])$('capture-'+key).addEventListener('input',()=>{if(rect)setRect({...rect,[key]:Number($('capture-'+key).value)});});
  $('capture-use').addEventListener('click',async()=>{
    if(!frameId||!validRect()||pending)return;pending=true;update();error('');const version=revision;
    try{const result=await(await request('/frames/'+frameId+'/crop',json(rect))).json();if(version!==revision)return;
      const match=/^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(result.image);if(!match)throw new Error('The service returned an invalid captured picture.');const bytes=Uint8Array.from(atob(match[1]),c=>c.charCodeAt(0));if(version!==revision)return;const file=new File([bytes],'captured-face.png',{type:'image/png'});
      const accepted=await window.restorationInput.use(file,{captureId:result.capture_id,label:'Captured face',isCurrent:()=>version===revision});
      if(accepted&&version===revision){
        $('capture-restore-upload').classList.toggle('hidden',!savedUpload);
        suspendSource();
        phase='review';renderPhase();$('brush-upload-heading').focus({preventScroll:true});
      }
      else if(version===revision)error($('brush-error').textContent||'The captured Picture could not be checked. Adjust the crop and try again.');
    }catch(problem){if(version===revision)error(problem.message);}finally{if(version===revision){pending=false;update();}}
  });
  $('capture-restore-upload').addEventListener('click',async()=>{if(savedUpload){const accepted=await window.restorationInput.use(savedUpload);if(accepted)$('capture-restore-upload').classList.add('hidden');}});
  window.addEventListener('picture:changed',event=>{if(!event.detail.captured){savedUpload=event.detail.file;$('capture-restore-upload').classList.add('hidden');}});
  $('capture-back').addEventListener('click',()=>{phase='capture';renderPhase();});
  $('capture-change-frame').addEventListener('click',()=>{if(kind()==='video'&&$('capture-video-file').files[0])start();else{stop();clearFrame();update();status('Start the camera to choose another frame.');}});
  $('capture-camera-dialog').addEventListener('close',()=>{$('capture-password').value='';});
  $('capture-connect').addEventListener('click',()=>{$('capture-camera-dialog').close();start();});
  $('capture-start').addEventListener('click',start);$('capture-stop').addEventListener('click',()=>{stop();status('Source stopped. Frozen frames remain available.');});$('capture-freeze').addEventListener('click',freeze);
  $('capture-resume').addEventListener('click',()=>{clearFrame();error('');if(kind()==='rtsp'){clearTimeout(streamTimer);pollStream(revision);$('capture-stream').classList.remove('hidden');}else{$('capture-video').classList.remove('hidden');video.play().catch(e=>error(e.message));}status('Find another frame to capture.');update();});
  function changeSource() {
    stop();clearFrame();error('');$('capture-video-field').classList.toggle('hidden',kind()!=='video');$('capture-rtsp-field').classList.toggle('hidden',kind()!=='rtsp');$('capture-start').textContent=kind()==='webcam'?'Start webcam':'Connect camera';
    status(kind()==='video'?'Choose a video file. It opens automatically.':'Start the selected camera when you are ready.');update();
  }
  $('capture-source').addEventListener('change',changeSource);
  $('capture-video-file').addEventListener('change',()=>{if(mode==='video'&&kind()==='video'&&$('capture-video-file').files[0])start();});
  $('capture-video-file').addEventListener('click',()=>{$('capture-video-file').value='';});
  $('capture-full-frame').addEventListener('click',fullFrame);
  function renderPhase(){
    const reviewing=mode==='picture'||phase==='review', screen=document.querySelector('[data-screen=inspect]');
    screen.dataset.inputMode=mode;screen.dataset.inputPhase=reviewing?'review':'capture';
    $('input-mode-picture').setAttribute('aria-pressed',String(mode==='picture'));$('input-mode-video').setAttribute('aria-pressed',String(mode==='video'));
    $('picture-upload-field').classList.toggle('hidden',mode!=='picture');$('capture-controls').classList.toggle('hidden',reviewing);
    $('picture-preview').classList.toggle('hidden',!reviewing);$('capture-preview').classList.toggle('hidden',reviewing);
    $('capture-back').classList.toggle('hidden',mode!=='video'||!reviewing);
    const heading=$('brush-upload-heading');heading.tabIndex=-1;
    heading.textContent=mode==='picture'?'Inspect the original':reviewing?'Review picture':'Capture a face';
    heading.nextElementSibling.textContent=mode==='video'&&!reviewing?'Choose a frame, then select a face or drag to crop.':'Use a close-up of one face looking forward or slightly to the side.';
    placePreview();update();
  }
  function setMode(next) {
    if(next===mode)return;
    suspendSource();mode=next;phase=mode==='video'?'capture':'review';error('');
    status(sourceReady()
      ? frozen?'Frozen frame kept. Resume video to choose another frame, or adjust this crop.':'Video paused. Press Play or Freeze frame to continue.'
      : frozen?'Frozen frame kept. Use Change frame to reopen the source.':'Choose a video file or camera source.');
    renderPhase();
  }
  const narrowPreview=matchMedia('(max-width:800px)'),visual=document.querySelector('.capture-visual');
  function placePreview(){
    const capturing=mode==='video'&&phase==='capture';
    if(narrowPreview.matches){
      const anchor=capturing?$('capture-source-summary'):$('picture-upload-field');
      anchor.after(visual);
    }else document.querySelector('[data-screen=inspect]').append(visual);
    $('capture-empty').textContent=narrowPreview.matches?'Choose a video file or start a camera above.':'Choose a video file or start a camera on the left.';
    $('picture-empty').querySelector('p').textContent=narrowPreview.matches?'Choose a picture above, or switch to Video to capture a face.':'Choose a picture on the left, or switch to Video to capture a face.';
  }
  narrowPreview.addEventListener('change',placePreview);
  renderPhase();
  $('input-mode-picture').addEventListener('click',()=>setMode('picture'));$('input-mode-video').addEventListener('click',()=>setMode('video'));
  function syncPicturePreview() {
    const src=$('upload-preview').getAttribute('src');
    if(src)$('picture-main-preview').src=src;else $('picture-main-preview').removeAttribute('src');
    $('picture-main-preview').classList.toggle('hidden',!src);$('picture-empty').classList.toggle('hidden',!!src);
  }
  new MutationObserver(syncPicturePreview).observe($('upload-preview'),{attributes:true,attributeFilter:['src']});
  syncPicturePreview();
  // Native controls own shortcuts while the player has focus. Outside it,
  // Space works only in this file's active Capture view and never edits a form.
  document.addEventListener('keydown',event=>{
    if(event.code!=='Space'||event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
    if(mode!=='video'||phase!=='capture'||kind()!=='video'||!fileUrl||pending||frozen||video.readyState<2||video.seeking)return;
    if(!document.querySelector('[data-screen=inspect]').classList.contains('is-active')||document.querySelector('dialog[open]'))return;
    const target=event.target;
    if(event.composedPath().includes(video)||target.closest?.('input,textarea,select,button,a,[role=button],[role=slider],[contenteditable]:not([contenteditable="false"])'))return;
    event.preventDefault();
    if(video.paused)video.play().catch(problem=>error(problem.message));else video.pause();
  });
  for(const name of ['seeking','seeked','loadeddata','canplay','waiting'])video.addEventListener(name,update);
  new MutationObserver(()=>{if(!document.querySelector('[data-screen=inspect]').classList.contains('is-active'))suspendSource();}).observe(document.querySelector('[data-screen=inspect]'),{attributes:true,attributeFilter:['class']});
  window.addEventListener('restoration:busy', update);
  window.addEventListener('pagehide',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden)suspendSource();});
  update();
})();

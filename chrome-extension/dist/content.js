var M=Object.defineProperty;var k=(h,t,e)=>t in h?M(h,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):h[t]=e;var T=(h,t,e)=>k(h,typeof t!="symbol"?t+"":t,e);class E{constructor(){T(this,"overlays",new Map);T(this,"audioContext");this.init()}init(){setTimeout(()=>{this.observeVideos()},500),this.observeVideos()}observeVideos(){new MutationObserver(o=>{o.forEach(n=>{n.addedNodes.forEach(s=>{s.nodeType===Node.ELEMENT_NODE&&s.querySelectorAll("video").forEach(r=>{this.attachOverlay(r)})})})}).observe(document.body,{childList:!0,subtree:!0}),document.querySelectorAll("video").forEach(o=>{this.attachOverlay(o)})}attachOverlay(t){if(this.overlays.has(t))return;const e=document.createElement("div");e.className="video-tutor-overlay",e.style.cssText=`
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
    `;const o=this.createControlBar(t);e.appendChild(o);const n=this.createChatPanel();e.appendChild(n);const s=this.createTranscriptPanel();if(e.appendChild(s),!t.parentElement)return;let a=t.parentElement;const r=t.closest("#movie_player")||t.closest(".html5-video-player")||t.closest("[data-layer]");r&&(a=r),a.style.position="relative",a.appendChild(e);const i={element:e,isVisible:!1,transcripts:[]};this.overlays.set(t,i)}createControlBar(t){const e=document.createElement("div");e.className="tutor-control-bar slide-in-bottom",e.style.cssText=`
      position: absolute;
      bottom: 16px;
      right: 16px;
      padding: 12px 16px;
      display: flex;
      gap: 12px;
      pointer-events: auto;
      z-index: 10000;
      cursor: move;
      border-radius: 16px;
    `,this.makeDraggable(e);const o=document.createElement("button");o.innerHTML=`
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="6"/>
      </svg>
      <span style="margin-left: 6px;">Record</span>
    `,o.className="ai-button ai-button-record",o.style.cssText=`
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `,o.onclick=()=>this.handleRecordClick(t);const n=document.createElement("button");n.innerHTML=`
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
      </svg>
      <span style="margin-left: 6px;">Ask AI</span>
    `,n.className="ai-button ai-button-ask",n.style.cssText=`
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `,n.onclick=()=>this.handleAskClick(t);const s=document.createElement("button");s.innerHTML=`
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <span style="margin-left: 6px;">Transcript</span>
    `,s.className="ai-button ai-button-transcript",s.style.cssText=`
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `,s.onclick=()=>this.handleTranscriptClick(t);const a=document.createElement("button");return a.innerHTML=`
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
        <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
      </svg>
    `,a.className="ai-button ai-button-settings",a.style.cssText=`
      padding: 10px 12px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `,e.appendChild(o),e.appendChild(n),e.appendChild(s),e.appendChild(a),e}createChatPanel(){const t=document.createElement("div");t.className="tutor-chat-panel slide-in-right",t.style.cssText=`
      position: absolute;
      top: 16px;
      right: 16px;
      width: 380px;
      max-height: 520px;
      padding: 20px;
      display: none;
      flex-direction: column;
      gap: 16px;
      pointer-events: auto;
      cursor: move;
      border-radius: 20px;
    `,this.makeDraggable(t);const e=document.createElement("div");e.className="ai-panel-header",e.style.cssText=`
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;const o=document.createElement("h3");o.className="ai-panel-title",o.innerHTML=`
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px; vertical-align: middle;">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
      </svg>
      AI Tutor Chat
    `,o.style.cssText=`
      margin: 0;
      display: flex;
      align-items: center;
    `;const n=document.createElement("button");n.className="ai-close-button",n.innerHTML=`
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 18L18 6M6 6l12 12"/>
      </svg>
    `,n.onclick=()=>{t.style.display="none"},e.appendChild(o),e.appendChild(n);const s=document.createElement("div");s.className="messages-container",s.style.cssText=`
      flex: 1;
      overflow-y: auto;
      max-height: 350px;
      padding-right: 8px;
      margin-right: -8px;
    `;const a=document.createElement("div");a.style.cssText=`
      display: flex;
      gap: 12px;
      align-items: flex-end;
    `;const r=document.createElement("input");r.type="text",r.placeholder="Ask me about this video...",r.className="ai-input",r.style.cssText=`
      flex: 1;
      padding: 12px 16px;
      font-size: 14px;
      line-height: 1.5;
    `;const i=document.createElement("button");return i.className="ai-button ai-button-ask",i.innerHTML=`
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
      </svg>
    `,i.style.cssText=`
      padding: 12px 16px;
      border: none;
      min-width: 48px;
    `,r.addEventListener("keydown",c=>{c.stopPropagation(),c.key==="Enter"&&!c.shiftKey&&(c.preventDefault(),this.handleSendMessage(r,s))}),r.addEventListener("keyup",c=>{c.stopPropagation()}),r.addEventListener("keypress",c=>{c.stopPropagation()}),i.onclick=()=>this.handleSendMessage(r,s),a.appendChild(r),a.appendChild(i),t.appendChild(e),t.appendChild(s),t.appendChild(a),t}createTranscriptPanel(){const t=document.createElement("div");t.className="tutor-transcript-panel slide-in-left",t.style.cssText=`
      position: absolute;
      top: 16px;
      left: 16px;
      width: 420px;
      max-height: 520px;
      padding: 20px;
      display: none;
      flex-direction: column;
      gap: 16px;
      pointer-events: auto;
      cursor: move;
      border-radius: 20px;
    `,this.makeDraggable(t);const e=document.createElement("div");e.className="ai-panel-header",e.style.cssText=`
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;const o=document.createElement("h3");o.className="ai-panel-title",o.innerHTML=`
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px; vertical-align: middle;">
        <path d="M19 11H5m14-4H5m14 8H5m6 4H5"/>
      </svg>
      Live Transcript
    `,o.style.cssText=`
      margin: 0;
      display: flex;
      align-items: center;
    `;const n=document.createElement("button");n.className="ai-close-button",n.innerHTML=`
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 18L18 6M6 6l12 12"/>
      </svg>
    `,n.onclick=()=>{t.style.display="none"},e.appendChild(o),e.appendChild(n);const s=document.createElement("div");s.className="transcript-content",s.style.cssText=`
      flex: 1;
      overflow-y: auto;
      max-height: 420px;
      background: rgba(255, 255, 255, 0.03);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 16px;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      color: var(--text-secondary);
      font-family: 'Inter', monospace;
      padding-right: 12px;
      margin-right: -8px;
    `;const a=document.createElement("div");return a.innerHTML=`
      <div style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" style="margin-bottom: 16px; opacity: 0.5;">
          <path d="M12 14l9-5-9-5-9 5 9 5z"/>
          <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/>
        </svg>
        <div style="font-weight: 500; margin-bottom: 8px;">Ready to transcribe</div>
        <div style="font-size: 13px; opacity: 0.7;">Click "Record" to start capturing live transcript</div>
      </div>
    `,s.appendChild(a),t.appendChild(e),t.appendChild(s),t}makeDraggable(t){let e=!1,o=0,n=0,s=0,a=0;const r=()=>{const l=t.getBoundingClientRect(),d=t.parentElement.getBoundingClientRect();return{x:l.left-d.left,y:l.top-d.top}},i=(l,d)=>{const m=t.parentElement.getBoundingClientRect(),x=t.getBoundingClientRect(),y=0,w=0,f=m.width-x.width,b=m.height-x.height,v=Math.max(y,Math.min(f,l)),C=Math.max(w,Math.min(b,d));return{x:v,y:C}},c=l=>{const d=l.target;if(d.tagName==="BUTTON"||d.tagName==="INPUT"||d.tagName==="svg"||d.tagName==="path"||d.closest("button")||d.closest("input"))return;e=!0,t.style.cursor="grabbing",t.style.userSelect="none",o=l.clientX,n=l.clientY;const u=r();s=u.x,a=u.y,t.style.position="absolute",t.style.right="auto",t.style.bottom="auto",l.preventDefault()},p=l=>{if(!e)return;l.preventDefault();const d=l.clientX-o,u=l.clientY-n,m=s+d,x=a+u,y=i(m,x);t.style.left=`${y.x}px`,t.style.top=`${y.y}px`},g=()=>{e&&(e=!1,t.style.cursor="move",t.style.userSelect="auto")};t.addEventListener("mousedown",c),document.addEventListener("mousemove",p),document.addEventListener("mouseup",g),t.addEventListener("dragstart",l=>l.preventDefault())}async setupAudioCapture(t){try{console.log("🎓 VideoTutor: Setting up direct video audio capture...");let e;try{if(t.captureStream)e=t.captureStream(),console.log("🎓 VideoTutor: Direct video audio capture successful");else throw new Error("captureStream not supported")}catch(s){console.log("🎓 VideoTutor: Direct video capture failed, trying tab capture...",s);try{const a=await chrome.runtime.sendMessage({action:"requestTabCapture"});if(a.streamId)e=await navigator.mediaDevices.getUserMedia({audio:{mandatory:{chromeMediaSource:"tab",chromeMediaSourceId:a.streamId}}}),console.log("🎓 VideoTutor: Tab audio capture successful");else throw new Error("Tab capture failed")}catch(a){console.log("🎓 VideoTutor: Tab capture failed, trying display media...",a);try{e=await navigator.mediaDevices.getDisplayMedia({audio:{echoCancellation:!1,noiseSuppression:!1,sampleRate:16e3},video:!1}),console.log("🎓 VideoTutor: Display media audio capture successful")}catch(r){console.log("🎓 VideoTutor: Display media failed, using microphone...",r),e=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:!0,noiseSuppression:!0,sampleRate:16e3}}),console.log("🎓 VideoTutor: Microphone audio capture successful")}}}this.audioContext=new AudioContext({sampleRate:16e3});const o=this.audioContext.createMediaStreamSource(e),n=this.overlays.get(t);n&&(n.audioStream=e,this.connectWebSocket(t)),this.processAudioStream(o,t),console.log("🎓 VideoTutor: Audio processing started")}catch(e){console.error("🎓 VideoTutor: Failed to capture audio:",e);const o=this.overlays.get(t);if(o){const n=o.element.querySelector("button");n.textContent="❌ No Mic",n.className="bg-gray-600 text-white px-3 py-1 rounded text-sm",n.disabled=!0}throw e}}connectWebSocket(t){try{const e=this.getVideoId();console.log("🎓 VideoTutor: Connecting WebSocket with videoId:",e);const o=new WebSocket(`ws://localhost:8000/audio?video_id=${encodeURIComponent(e)}`),n=this.overlays.get(t);n&&(n.websocket=o),o.onopen=()=>{console.log("🎓 VideoTutor: Audio WebSocket connected")},o.onmessage=s=>{try{const a=JSON.parse(s.data);a.type==="transcription"&&(console.log("🎓 VideoTutor: Raw transcription received:",a.text),a.text.trim()?(console.log("🎓 VideoTutor: Adding transcription to storage:",a.text),n&&(n.transcripts.push({timestamp:a.timestamp||Date.now(),text:a.text}),console.log("🎓 VideoTutor: Total transcripts stored:",n.transcripts.length),this.updateTranscriptPanel(t))):console.log("🎓 VideoTutor: Skipping empty transcription"))}catch(a){console.error("🎓 VideoTutor: Error parsing WebSocket message:",a)}},o.onerror=s=>{console.error("🎓 VideoTutor: WebSocket error:",s)},o.onclose=s=>{console.log("🎓 VideoTutor: WebSocket closed:",s.code,s.reason)}}catch(e){console.error("🎓 VideoTutor: Failed to create WebSocket:",e)}}async processAudioStream(t,e){try{await this.audioContext.audioWorklet.addModule("data:text/javascript,class%20AudioProcessor%20extends%20AudioWorkletProcessor%20%7B%0A%20%20process(inputs%2C%20outputs%2C%20parameters)%20%7B%0A%20%20%20%20const%20input%20%3D%20inputs%5B0%5D%3B%0A%20%20%20%20if%20(input%20%26%26%20input%5B0%5D)%20%7B%0A%20%20%20%20%20%20this.port.postMessage(%7B%20audioData%3A%20Array.from(input%5B0%5D)%20%7D)%3B%0A%20%20%20%20%7D%0A%20%20%20%20return%20true%3B%0A%20%20%7D%0A%7D%0AregisterProcessor(%27audio-processor%27%2C%20AudioProcessor)%3B");const o=new AudioWorkletNode(this.audioContext,"audio-processor");o.port.onmessage=n=>{var r;const{audioData:s}=n.data,a=this.overlays.get(e);if(((r=a==null?void 0:a.websocket)==null?void 0:r.readyState)===WebSocket.OPEN&&s){const i=this.convertToPCM(new Float32Array(s));a.websocket.send(i)}},t.connect(o),o.connect(this.audioContext.destination)}catch(o){console.log("🎓 VideoTutor: AudioWorklet not supported, falling back to ScriptProcessor",o);const n=this.audioContext.createScriptProcessor(4096,1,1);n.onaudioprocess=s=>{var c;const r=s.inputBuffer.getChannelData(0),i=this.overlays.get(e);if(((c=i==null?void 0:i.websocket)==null?void 0:c.readyState)===WebSocket.OPEN){const p=this.convertToPCM(r);i.websocket.send(p)}},t.connect(n),n.connect(this.audioContext.destination)}}convertToPCM(t){const e=new ArrayBuffer(t.length*2),o=new Int16Array(e);for(let n=0;n<t.length;n++){const s=Math.max(-1,Math.min(1,t[n]));o[n]=s<0?s*32768:s*32767}return e}async handleRecordClick(t){const e=this.overlays.get(t);if(!e)return;const o=e.element.querySelector(".ai-button-record");try{e.audioStream?(console.log("🎓 VideoTutor: Stopping audio capture..."),o.innerHTML=`
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="6"/>
          </svg>
          <span style="margin-left: 6px;">Record</span>
        `,o.classList.remove("recording"),this.stopAudioCapture(t)):(console.log("🎓 VideoTutor: Starting audio capture..."),o.innerHTML=`
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg>
          <span style="margin-left: 6px;">Stop</span>
        `,o.classList.add("recording"),await this.setupAudioCapture(t))}catch(n){console.error("🎓 VideoTutor: Audio capture error:",n),o.innerHTML=`
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 18L18 6M6 6l12 12"/>
        </svg>
        <span style="margin-left: 6px;">Failed</span>
      `,o.classList.remove("recording"),o.style.opacity="0.6",o.style.cursor="not-allowed"}}stopAudioCapture(t){const e=this.overlays.get(t);e&&(e.audioStream&&(e.audioStream.getTracks().forEach(o=>o.stop()),e.audioStream=void 0),e.websocket&&(e.websocket.close(),e.websocket=void 0))}handleAskClick(t){const e=this.overlays.get(t);if(!e)return;const o=e.element.querySelector(".tutor-chat-panel");o.style.display==="none"?o.style.display="flex":o.style.display="none"}handleTranscriptClick(t){const e=this.overlays.get(t);if(!e)return;const o=e.element.querySelector(".tutor-transcript-panel");o.style.display==="none"?(o.style.display="flex",this.updateTranscriptPanel(t)):o.style.display="none"}updateTranscriptPanel(t){const e=this.overlays.get(t);if(!e)return;const n=e.element.querySelector(".tutor-transcript-panel").querySelector(".transcript-content");if(e.transcripts.length===0){n.innerHTML=`
        <div style="color: #9CA3AF; font-style: italic; text-align: center; padding: 20px;">
          Start recording to see live transcript...
        </div>
      `;return}const s=e.transcripts.map(a=>`[${new Date(a.timestamp).toLocaleTimeString()}] ${a.text}`).join(`

`);n.textContent=s,n.scrollTop=n.scrollHeight}async handleSendMessage(t,e){var s,a,r,i;const o=t.value.trim();if(!o)return;this.addMessage(e,o,"user"),t.value="";const n=this.addLoadingMessage(e);try{const c=this.getVideoId(),p=this.getCurrentTimestamp();console.log("🎓 VideoTutor: Sending query with videoId:",c,"message:",o);const g=await fetch("http://localhost:8000/query",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({videoId:c,timestamp:p,prompt:o})});if(n.remove(),!g.ok)throw new Error(`HTTP ${g.status}: ${g.statusText}`);const l=(s=g.body)==null?void 0:s.getReader();let d="";const u=this.addMessage(e,"","assistant");for(;l;){const{done:m,value:x}=await l.read();if(m)break;const w=new TextDecoder().decode(x).split(`
`);for(const f of w)if(f.startsWith("data: ")){const b=f.slice(6);if(b==="[DONE]")break;try{const v=JSON.parse(b);(i=(r=(a=v.choices)==null?void 0:a[0])==null?void 0:r.delta)!=null&&i.content&&(d+=v.choices[0].delta.content,u.textContent=d)}catch{}}}}catch(c){console.error("🎓 VideoTutor: Error sending message:",c),n.parentNode&&n.remove();const p=c instanceof Error?c.message:"Unknown error";this.addMessage(e,`Error: ${p}`,"error")}}addLoadingMessage(t){const e=document.createElement("div");e.className="message-loading slide-in-left",e.style.cssText=`
      padding: 16px 20px;
      margin-bottom: 12px;
      font-size: 14px;
      line-height: 1.5;
      display: flex;
      align-items: center;
      gap: 12px;
      animation-delay: 0.1s;
    `;const o=document.createElement("div");o.style.cssText=`
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: var(--ai-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    `,o.innerHTML=`
      <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
      </svg>
    `;const n=document.createElement("div");n.style.cssText=`
      flex: 1;
      display: flex;
      align-items: center;
      gap: 8px;
    `;const s=document.createElement("span");s.textContent="AI is thinking",s.style.cssText=`
      font-weight: 500;
      color: var(--text-secondary);
    `;const a=document.createElement("div");a.style.cssText=`
      display: flex;
      gap: 4px;
    `;for(let r=0;r<3;r++){const i=document.createElement("div");i.style.cssText=`
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--text-muted);
        animation: pulse 1.5s infinite;
        animation-delay: ${r*.2}s;
      `,a.appendChild(i)}return n.appendChild(s),n.appendChild(a),e.appendChild(o),e.appendChild(n),t.appendChild(e),t.scrollTop=t.scrollHeight,e}addMessage(t,e,o){const n=document.createElement("div"),s=o==="user"?"slide-in-right":"slide-in-left";if(n.className=`message-${o} ${s}`,o==="user")n.style.cssText=`
        padding: 12px 18px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
        animation-delay: 0.05s;
      `,n.textContent=e;else if(o==="assistant"){n.style.cssText=`
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
        word-wrap: break-word;
        display: flex;
        gap: 12px;
        animation-delay: 0.1s;
      `;const a=document.createElement("div");a.style.cssText=`
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--ai-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        align-self: flex-start;
      `,a.innerHTML=`
        <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
          <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
        </svg>
      `;const r=document.createElement("div");r.style.cssText=`
        flex: 1;
        padding-top: 2px;
      `,r.textContent=e,n.appendChild(a),n.appendChild(r)}else n.style.cssText=`
        padding: 12px 18px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 90%;
        margin: 0 auto 12px auto;
        word-wrap: break-word;
        text-align: center;
        animation-delay: 0.1s;
      `,n.textContent=e;return t.appendChild(n),t.scrollTop=t.scrollHeight,n}getVideoId(){const t=window.location.href,e=t.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);return e?e[1]:t}getCurrentTimestamp(){const t=document.querySelector("video");return t?t.currentTime:0}}new E;

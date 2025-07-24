var T=Object.defineProperty;var k=(d,t,e)=>t in d?T(d,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):d[t]=e;var x=(d,t,e)=>k(d,typeof t!="symbol"?t+"":t,e);class w{constructor(){x(this,"overlays",new Map);x(this,"audioContext");this.init()}init(){setTimeout(()=>{this.observeVideos()},500),this.observeVideos()}observeVideos(){new MutationObserver(n=>{n.forEach(o=>{o.addedNodes.forEach(r=>{r.nodeType===Node.ELEMENT_NODE&&r.querySelectorAll("video").forEach(a=>{this.attachOverlay(a)})})})}).observe(document.body,{childList:!0,subtree:!0}),document.querySelectorAll("video").forEach(n=>{this.attachOverlay(n)})}attachOverlay(t){if(this.overlays.has(t))return;const e=document.createElement("div");e.className="video-tutor-overlay",e.style.cssText=`
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
    `;const n=this.createControlBar(t);e.appendChild(n);const o=this.createChatPanel();e.appendChild(o);const r=this.createTranscriptPanel();if(e.appendChild(r),!t.parentElement)return;let s=t.parentElement;const a=t.closest("#movie_player")||t.closest(".html5-video-player")||t.closest("[data-layer]");a&&(s=a),s.style.position="relative",s.appendChild(e);const c={element:e,isVisible:!1,transcripts:[]};this.overlays.set(t,c)}createControlBar(t){const e=document.createElement("div");e.className="tutor-control-bar",e.style.cssText=`
      position: absolute;
      bottom: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      gap: 8px;
      pointer-events: auto;
      z-index: 10000;
      cursor: move;
      transition: opacity 0.3s ease;
    `,this.makeDraggable(e);const n=document.createElement("button");n.textContent="🎤 Record",n.style.cssText=`
      background: #dc2626;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `,n.onclick=()=>this.handleRecordClick(t);const o=document.createElement("button");o.textContent="Ask",o.style.cssText=`
      background: #2563eb;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `,o.onclick=()=>this.handleAskClick(t);const r=document.createElement("button");r.textContent="📝 Transcript",r.style.cssText=`
      background: #059669;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `,r.onclick=()=>this.handleTranscriptClick(t);const s=document.createElement("button");return s.textContent="⚙️",s.style.cssText=`
      background: #6b7280;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
    `,e.appendChild(n),e.appendChild(o),e.appendChild(r),e.appendChild(s),e}createChatPanel(){const t=document.createElement("div");t.className="tutor-chat-panel",t.style.cssText=`
      position: absolute;
      top: 10px;
      right: 10px;
      width: 300px;
      max-height: 400px;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(12px);
      border-radius: 8px;
      padding: 16px;
      display: none;
      flex-direction: column;
      gap: 12px;
      pointer-events: auto;
      color: white;
      font-family: system-ui, sans-serif;
      cursor: move;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    `,this.makeDraggable(t);const e=document.createElement("div");e.className="messages-container",e.style.cssText=`
      flex: 1;
      overflow-y: auto;
      max-height: 300px;
    `;const n=document.createElement("div");n.style.cssText=`
      display: flex;
      gap: 8px;
    `;const o=document.createElement("input");o.type="text",o.placeholder="Ask about the video...",o.className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white",o.style.cssText=`
      flex: 1;
      padding: 8px 12px;
      background: #374151;
      border: 1px solid #4B5563;
      border-radius: 4px;
      color: white;
      outline: none;
    `;const r=document.createElement("button");return r.textContent="Send",r.style.cssText=`
      background: #2563eb;
      color: white;
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    `,o.addEventListener("keydown",s=>{s.stopPropagation(),s.key==="Enter"&&(s.preventDefault(),this.handleSendMessage(o,e))}),o.addEventListener("keyup",s=>{s.stopPropagation()}),o.addEventListener("keypress",s=>{s.stopPropagation()}),r.onclick=()=>this.handleSendMessage(o,e),n.appendChild(o),n.appendChild(r),t.appendChild(e),t.appendChild(n),t}createTranscriptPanel(){const t=document.createElement("div");t.className="tutor-transcript-panel",t.style.cssText=`
      position: absolute;
      top: 10px;
      left: 10px;
      width: 350px;
      max-height: 400px;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(12px);
      border-radius: 8px;
      padding: 16px;
      display: none;
      flex-direction: column;
      gap: 12px;
      pointer-events: auto;
      color: white;
      font-family: system-ui, sans-serif;
      cursor: move;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    `,this.makeDraggable(t);const e=document.createElement("div");e.style.cssText=`
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    `;const n=document.createElement("h3");n.textContent="Live Transcript",n.style.cssText=`
      margin: 0;
      font-size: 16px;
      font-weight: 600;
    `;const o=document.createElement("button");o.textContent="×",o.style.cssText=`
      background: none;
      border: none;
      color: white;
      font-size: 20px;
      cursor: pointer;
      padding: 0;
      width: 24px;
      height: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
    `,o.onclick=()=>{t.style.display="none"},e.appendChild(n),e.appendChild(o);const r=document.createElement("div");r.className="transcript-content",r.style.cssText=`
      flex: 1;
      overflow-y: auto;
      max-height: 320px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      padding: 8px;
      font-size: 14px;
      line-height: 1.4;
      white-space: pre-wrap;
    `;const s=document.createElement("div");return s.textContent="Start recording to see live transcript...",s.style.cssText=`
      color: #9CA3AF;
      font-style: italic;
      text-align: center;
      padding: 20px;
    `,r.appendChild(s),t.appendChild(e),t.appendChild(r),t}makeDraggable(t){let e=!1,n=0,o=0,r=0,s=0,a=0,c=0;const l=i=>{const p=i.target;p.tagName==="BUTTON"||p.tagName==="INPUT"||(r=i.clientX-a,s=i.clientY-c,i.target===t&&(e=!0,t.style.cursor="grabbing"))},u=()=>{r=n,s=o,e=!1,t.style.cursor="move"},h=i=>{e&&(i.preventDefault(),n=i.clientX-r,o=i.clientY-s,a=n,c=o,t.style.left=`${n}px`,t.style.top=`${o}px`,t.style.right="auto",t.style.bottom="auto")};t.addEventListener("mousedown",l),document.addEventListener("mousemove",h),document.addEventListener("mouseup",u)}async setupAudioCapture(t){try{console.log("🎓 VideoTutor: Requesting audio capture permissions...");let e;try{e=await navigator.mediaDevices.getDisplayMedia({audio:{echoCancellation:!0,noiseSuppression:!0,sampleRate:16e3},video:!1}),console.log("🎓 VideoTutor: Screen audio capture successful")}catch(r){console.log("🎓 VideoTutor: Screen audio failed, trying microphone...",r),e=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:!0,noiseSuppression:!0,sampleRate:16e3}}),console.log("🎓 VideoTutor: Microphone audio capture successful")}this.audioContext=new AudioContext({sampleRate:16e3});const n=this.audioContext.createMediaStreamSource(e),o=this.overlays.get(t);o&&(o.audioStream=e,this.connectWebSocket(t)),this.processAudioStream(n,t),console.log("🎓 VideoTutor: Audio processing started")}catch(e){console.error("🎓 VideoTutor: Failed to capture audio:",e);const n=this.overlays.get(t);if(n){const o=n.element.querySelector("button");o.textContent="❌ No Mic",o.className="bg-gray-600 text-white px-3 py-1 rounded text-sm",o.disabled=!0}throw e}}connectWebSocket(t){try{const e=new WebSocket("ws://localhost:8000/audio"),n=this.overlays.get(t);n&&(n.websocket=e),e.onopen=()=>{console.log("🎓 VideoTutor: Audio WebSocket connected")},e.onmessage=o=>{try{const r=JSON.parse(o.data);r.type==="transcription"&&r.text.trim()&&(console.log("🎓 VideoTutor: Transcription:",r.text),n&&(n.transcripts.push({timestamp:r.timestamp||Date.now(),text:r.text}),this.updateTranscriptPanel(t)))}catch(r){console.error("🎓 VideoTutor: Error parsing WebSocket message:",r)}},e.onerror=o=>{console.error("🎓 VideoTutor: WebSocket error:",o)},e.onclose=o=>{console.log("🎓 VideoTutor: WebSocket closed:",o.code,o.reason)}}catch(e){console.error("🎓 VideoTutor: Failed to create WebSocket:",e)}}processAudioStream(t,e){const n=this.audioContext.createScriptProcessor(4096,1,1);n.onaudioprocess=o=>{var c;const s=o.inputBuffer.getChannelData(0),a=this.overlays.get(e);if(((c=a==null?void 0:a.websocket)==null?void 0:c.readyState)===WebSocket.OPEN){const l=this.convertToPCM(s);a.websocket.send(l)}},t.connect(n),n.connect(this.audioContext.destination)}convertToPCM(t){const e=new ArrayBuffer(t.length*2),n=new Int16Array(e);for(let o=0;o<t.length;o++){const r=Math.max(-1,Math.min(1,t[o]));n[o]=r<0?r*32768:r*32767}return e}async handleRecordClick(t){const e=this.overlays.get(t);if(!e)return;const n=e.element.querySelector("button");try{e.audioStream?(console.log("🎓 VideoTutor: Stopping audio capture..."),n.textContent="🎤 Record",n.className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm",this.stopAudioCapture(t)):(console.log("🎓 VideoTutor: Starting audio capture..."),n.textContent="⏸️ Stop",n.className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm",await this.setupAudioCapture(t))}catch(o){console.error("🎓 VideoTutor: Audio capture error:",o),n.textContent="❌ Failed",n.className="bg-gray-600 text-white px-3 py-1 rounded text-sm"}}stopAudioCapture(t){const e=this.overlays.get(t);e&&(e.audioStream&&(e.audioStream.getTracks().forEach(n=>n.stop()),e.audioStream=void 0),e.websocket&&(e.websocket.close(),e.websocket=void 0))}handleAskClick(t){const e=this.overlays.get(t);if(!e)return;const n=e.element.querySelector(".tutor-chat-panel");n.style.display==="none"?n.style.display="flex":n.style.display="none"}handleTranscriptClick(t){const e=this.overlays.get(t);if(!e)return;const n=e.element.querySelector(".tutor-transcript-panel");n.style.display==="none"?(n.style.display="flex",this.updateTranscriptPanel(t)):n.style.display="none"}updateTranscriptPanel(t){const e=this.overlays.get(t);if(!e)return;const o=e.element.querySelector(".tutor-transcript-panel").querySelector(".transcript-content");if(e.transcripts.length===0){o.innerHTML=`
        <div style="color: #9CA3AF; font-style: italic; text-align: center; padding: 20px;">
          Start recording to see live transcript...
        </div>
      `;return}const r=e.transcripts.map(s=>`[${new Date(s.timestamp).toLocaleTimeString()}] ${s.text}`).join(`

`);o.textContent=r,o.scrollTop=o.scrollHeight}async handleSendMessage(t,e){var r,s,a,c;const n=t.value.trim();if(!n)return;this.addMessage(e,n,"user"),t.value="";const o=this.addLoadingMessage(e);try{const l=this.getVideoId(),u=this.getCurrentTimestamp(),h=await fetch("http://localhost:8000/query",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({videoId:l,timestamp:u,prompt:n})});o.remove();const i=(r=h.body)==null?void 0:r.getReader();let p="";const b=this.addMessage(e,"","assistant");for(;i;){const{done:f,value:v}=await i.read();if(f)break;const C=new TextDecoder().decode(v).split(`
`);for(const m of C)if(m.startsWith("data: ")){const y=m.slice(6);if(y==="[DONE]")break;try{const g=JSON.parse(y);(c=(a=(s=g.choices)==null?void 0:s[0])==null?void 0:a.delta)!=null&&c.content&&(p+=g.choices[0].delta.content,b.textContent=p)}catch{}}}}catch(l){console.error("Error sending message:",l),o.parentNode&&o.remove(),this.addMessage(e,"Error: Could not send message","error")}}addLoadingMessage(t){const e=document.createElement("div");e.className="message-loading",e.style.cssText=`
      padding: 8px 12px;
      border-radius: 8px;
      margin-bottom: 8px;
      background: #374151;
      color: white;
      font-size: 14px;
      line-height: 1.4;
      display: flex;
      align-items: center;
      gap: 8px;
    `;const n=document.createElement("span");n.innerHTML='Thinking<span class="loading-dots"></span>';const o=document.createElement("style");return o.textContent=`
      .loading-dots::after {
        content: '';
        animation: loading-dots 1.5s infinite;
      }
      
      @keyframes loading-dots {
        0% { content: ''; }
        25% { content: '.'; }
        50% { content: '..'; }
        75% { content: '...'; }
        100% { content: ''; }
      }
    `,document.head.querySelector("style[data-loading-dots]")||(o.setAttribute("data-loading-dots","true"),document.head.appendChild(o)),e.appendChild(n),t.appendChild(e),t.scrollTop=t.scrollHeight,e}addMessage(t,e,n){const o=document.createElement("div");return o.className=`message-${n}`,o.style.cssText=`
      padding: 8px 12px;
      border-radius: 8px;
      margin-bottom: 8px;
      background: ${n==="user"?"#2563EB":n==="error"?"#DC2626":"#374151"};
      color: white;
      font-size: 14px;
      line-height: 1.4;
    `,o.textContent=e,t.appendChild(o),t.scrollTop=t.scrollHeight,o}getVideoId(){const t=window.location.href,e=t.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);return e?e[1]:t}getCurrentTimestamp(){const t=document.querySelector("video");return t?t.currentTime:0}}new w;

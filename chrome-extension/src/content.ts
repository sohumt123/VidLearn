interface VideoTutorOverlay {
  element: HTMLDivElement;
  isVisible: boolean;
  audioStream?: MediaStream;
  websocket?: WebSocket;
  transcripts: Array<{timestamp: number, text: string}>;
}

// Extend HTMLVideoElement to include captureStream
declare global {
  interface HTMLVideoElement {
    captureStream?(): MediaStream;
  }
}

class VideoTutor {
  private overlays: Map<HTMLVideoElement, VideoTutorOverlay> = new Map();
  private audioContext?: AudioContext;

  constructor() {
    this.init();
  }

  private init() {
    // Wait a bit for the page to fully load
    setTimeout(() => {
      this.observeVideos();
    }, 500);
    
    // Also try immediately
    this.observeVideos();
  }

  private observeVideos() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const videos = (node as Element).querySelectorAll('video');
            videos.forEach((video) => {
              this.attachOverlay(video as HTMLVideoElement);
            });
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    // Check for existing videos
    const existingVideos = document.querySelectorAll('video');
    existingVideos.forEach((video) => {
      this.attachOverlay(video as HTMLVideoElement);
    });
  }

  private attachOverlay(video: HTMLVideoElement) {
    if (this.overlays.has(video)) {
      return;
    }

    const overlayContainer = document.createElement('div');
    overlayContainer.className = 'video-tutor-overlay';
    overlayContainer.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
    `;

    const controlBar = this.createControlBar(video);
    overlayContainer.appendChild(controlBar);

    const chatPanel = this.createChatPanel();
    overlayContainer.appendChild(chatPanel);

    const transcriptPanel = this.createTranscriptPanel();
    overlayContainer.appendChild(transcriptPanel);

    if (!video.parentElement) {
      return;
    }

    // Try to find a better container - look for YouTube's video container
    let targetContainer = video.parentElement;
    
    // Look for YouTube's player container
    const playerContainer = video.closest('#movie_player') || 
                           video.closest('.html5-video-player') || 
                           video.closest('[data-layer]');
                           
    if (playerContainer) {
      targetContainer = playerContainer as HTMLElement;
    }

    targetContainer.style.position = 'relative';
    targetContainer.appendChild(overlayContainer);

    const overlay: VideoTutorOverlay = {
      element: overlayContainer,
      isVisible: false,
      transcripts: []
    };

    this.overlays.set(video, overlay);
    
    // Don't automatically start audio capture - wait for user interaction
  }

  private createControlBar(video: HTMLVideoElement): HTMLDivElement {
    const controlBar = document.createElement('div');
    controlBar.className = 'tutor-control-bar slide-in-bottom';
    controlBar.style.cssText = `
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
    `;
    
    // Make control bar draggable
    this.makeDraggable(controlBar);

    const recordButton = document.createElement('button');
    recordButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="6"/>
      </svg>
      <span style="margin-left: 6px;">Record</span>
    `;
    recordButton.className = 'ai-button ai-button-record';
    recordButton.style.cssText = `
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `;
    recordButton.onclick = () => this.handleRecordClick(video);

    const askButton = document.createElement('button');
    askButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
      </svg>
      <span style="margin-left: 6px;">Ask AI</span>
    `;
    askButton.className = 'ai-button ai-button-ask';
    askButton.style.cssText = `
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `;
    askButton.onclick = () => this.handleAskClick(video);

    const transcriptButton = document.createElement('button');
    transcriptButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <span style="margin-left: 6px;">Transcript</span>
    `;
    transcriptButton.className = 'ai-button ai-button-transcript';
    transcriptButton.style.cssText = `
      padding: 10px 16px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `;
    transcriptButton.onclick = () => this.handleTranscriptClick(video);

    const settingsButton = document.createElement('button');
    settingsButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
        <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
      </svg>
    `;
    settingsButton.className = 'ai-button ai-button-settings';
    settingsButton.style.cssText = `
      padding: 10px 12px;
      pointer-events: auto;
      display: flex;
      align-items: center;
      border: none;
    `;

    controlBar.appendChild(recordButton);
    controlBar.appendChild(askButton);
    controlBar.appendChild(transcriptButton);
    controlBar.appendChild(settingsButton);

    return controlBar;
  }

  private createChatPanel(): HTMLDivElement {
    const chatPanel = document.createElement('div');
    chatPanel.className = 'tutor-chat-panel slide-in-right';
    chatPanel.style.cssText = `
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
    `;
    
    // Make chat panel draggable
    this.makeDraggable(chatPanel);

    // Modern header
    const header = document.createElement('div');
    header.className = 'ai-panel-header';
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;

    const title = document.createElement('h3');
    title.className = 'ai-panel-title';
    title.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px; vertical-align: middle;">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
      </svg>
      AI Tutor Chat
    `;
    title.style.cssText = `
      margin: 0;
      display: flex;
      align-items: center;
    `;

    const closeButton = document.createElement('button');
    closeButton.className = 'ai-close-button';
    closeButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 18L18 6M6 6l12 12"/>
      </svg>
    `;
    closeButton.onclick = () => {
      chatPanel.style.display = 'none';
    };

    header.appendChild(title);
    header.appendChild(closeButton);

    const messagesContainer = document.createElement('div');
    messagesContainer.className = 'messages-container';
    messagesContainer.style.cssText = `
      flex: 1;
      overflow-y: auto;
      max-height: 350px;
      padding-right: 8px;
      margin-right: -8px;
    `;

    const inputContainer = document.createElement('div');
    inputContainer.style.cssText = `
      display: flex;
      gap: 12px;
      align-items: flex-end;
    `;

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Ask me about this video...';
    input.className = 'ai-input';
    input.style.cssText = `
      flex: 1;
      padding: 12px 16px;
      font-size: 14px;
      line-height: 1.5;
    `;

    const sendButton = document.createElement('button');
    sendButton.className = 'ai-button ai-button-ask';
    sendButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
      </svg>
    `;
    sendButton.style.cssText = `
      padding: 12px 16px;
      border: none;
      min-width: 48px;
    `;

    input.addEventListener('keydown', (e) => {
      // Prevent YouTube keyboard shortcuts when typing
      e.stopPropagation();
      
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage(input, messagesContainer);
      }
    });
    
    input.addEventListener('keyup', (e) => {
      // Prevent YouTube keyboard shortcuts when typing
      e.stopPropagation();
    });
    
    input.addEventListener('keypress', (e) => {
      // Prevent YouTube keyboard shortcuts when typing
      e.stopPropagation();
    });

    sendButton.onclick = () => this.handleSendMessage(input, messagesContainer);

    inputContainer.appendChild(input);
    inputContainer.appendChild(sendButton);

    chatPanel.appendChild(header);
    chatPanel.appendChild(messagesContainer);
    chatPanel.appendChild(inputContainer);

    return chatPanel;
  }

  private createTranscriptPanel(): HTMLDivElement {
    const transcriptPanel = document.createElement('div');
    transcriptPanel.className = 'tutor-transcript-panel slide-in-left';
    transcriptPanel.style.cssText = `
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
    `;
    
    // Make transcript panel draggable
    this.makeDraggable(transcriptPanel);

    const header = document.createElement('div');
    header.className = 'ai-panel-header';
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;

    const title = document.createElement('h3');
    title.className = 'ai-panel-title';
    title.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px; vertical-align: middle;">
        <path d="M19 11H5m14-4H5m14 8H5m6 4H5"/>
      </svg>
      Live Transcript
    `;
    title.style.cssText = `
      margin: 0;
      display: flex;
      align-items: center;
    `;

    const closeButton = document.createElement('button');
    closeButton.className = 'ai-close-button';
    closeButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 18L18 6M6 6l12 12"/>
      </svg>
    `;
    closeButton.onclick = () => {
      transcriptPanel.style.display = 'none';
    };

    header.appendChild(title);
    header.appendChild(closeButton);

    const transcriptContent = document.createElement('div');
    transcriptContent.className = 'transcript-content';
    transcriptContent.style.cssText = `
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
    `;

    const placeholder = document.createElement('div');
    placeholder.innerHTML = `
      <div style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" style="margin-bottom: 16px; opacity: 0.5;">
          <path d="M12 14l9-5-9-5-9 5 9 5z"/>
          <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"/>
        </svg>
        <div style="font-weight: 500; margin-bottom: 8px;">Ready to transcribe</div>
        <div style="font-size: 13px; opacity: 0.7;">Click "Record" to start capturing live transcript</div>
      </div>
    `;
    transcriptContent.appendChild(placeholder);

    transcriptPanel.appendChild(header);
    transcriptPanel.appendChild(transcriptContent);

    return transcriptPanel;
  }

  private makeDraggable(element: HTMLElement) {
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let elementStartX = 0;
    let elementStartY = 0;

    const getElementPosition = () => {
      const rect = element.getBoundingClientRect();
      const parentRect = element.parentElement!.getBoundingClientRect();
      return {
        x: rect.left - parentRect.left,
        y: rect.top - parentRect.top
      };
    };

    const constrainPosition = (x: number, y: number) => {
      const parent = element.parentElement!;
      const parentRect = parent.getBoundingClientRect();
      const elementRect = element.getBoundingClientRect();
      
      // Calculate boundaries
      const minX = 0;
      const minY = 0;
      const maxX = parentRect.width - elementRect.width;
      const maxY = parentRect.height - elementRect.height;
      
      // Constrain within bounds
      const constrainedX = Math.max(minX, Math.min(maxX, x));
      const constrainedY = Math.max(minY, Math.min(maxY, y));
      
      return { x: constrainedX, y: constrainedY };
    };

    const dragStart = (e: MouseEvent) => {
      // Only allow dragging if not clicking on buttons, inputs, or SVGs
      const target = e.target as HTMLElement;
      if (target.tagName === 'BUTTON' || 
          target.tagName === 'INPUT' || 
          target.tagName === 'svg' ||
          target.tagName === 'path' ||
          target.closest('button') ||
          target.closest('input')) {
        return;
      }

      isDragging = true;
      element.style.cursor = 'grabbing';
      element.style.userSelect = 'none';
      
      // Store initial mouse position
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      
      // Store initial element position
      const pos = getElementPosition();
      elementStartX = pos.x;
      elementStartY = pos.y;
      
      // Ensure element is positioned absolutely
      element.style.position = 'absolute';
      element.style.right = 'auto';
      element.style.bottom = 'auto';
      
      e.preventDefault();
    };

    const drag = (e: MouseEvent) => {
      if (!isDragging) return;
      
      e.preventDefault();
      
      // Calculate movement delta
      const deltaX = e.clientX - dragStartX;
      const deltaY = e.clientY - dragStartY;
      
      // Calculate new position
      const newX = elementStartX + deltaX;
      const newY = elementStartY + deltaY;
      
      // Apply constraints
      const constrainedPos = constrainPosition(newX, newY);
      
      // Apply position
      element.style.left = `${constrainedPos.x}px`;
      element.style.top = `${constrainedPos.y}px`;
    };

    const dragEnd = () => {
      if (!isDragging) return;
      
      isDragging = false;
      element.style.cursor = 'move';
      element.style.userSelect = 'auto';
    };

    // Add event listeners
    element.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
    
    // Prevent default drag behavior on images and other elements
    element.addEventListener('dragstart', (e) => e.preventDefault());
  }

  private async setupAudioCapture(video: HTMLVideoElement) {
    try {
      console.log('🎓 VideoTutor: Setting up direct video audio capture...');
      
      let stream;
      
      // Method 1: Try to capture audio directly from the video element
      try {
        if (video.captureStream) {
          stream = video.captureStream();
          console.log('🎓 VideoTutor: Direct video audio capture successful');
        } else {
          throw new Error('captureStream not supported');
        }
      } catch (videoError) {
        console.log('🎓 VideoTutor: Direct video capture failed, trying tab capture...', videoError);
        
        // Method 2: Try tab capture using chrome.tabCapture API
        try {
          // Request tab capture permission via background script
          const response = await chrome.runtime.sendMessage({action: 'requestTabCapture'});
          if (response.streamId) {
            stream = await navigator.mediaDevices.getUserMedia({
              audio: {
                mandatory: {
                  chromeMediaSource: 'tab',
                  chromeMediaSourceId: response.streamId
                }
              }
            } as any);
            console.log('🎓 VideoTutor: Tab audio capture successful');
          } else {
            throw new Error('Tab capture failed');
          }
        } catch (tabError) {
          console.log('🎓 VideoTutor: Tab capture failed, trying display media...', tabError);
          
          // Method 3: Fallback to getDisplayMedia with audio
          try {
            stream = await navigator.mediaDevices.getDisplayMedia({
              audio: {
                echoCancellation: false,
                noiseSuppression: false,
                sampleRate: 16000
              },
              video: false
            });
            console.log('🎓 VideoTutor: Display media audio capture successful');
          } catch (displayError) {
            console.log('🎓 VideoTutor: Display media failed, using microphone...', displayError);
            
            // Method 4: Final fallback to microphone
            stream = await navigator.mediaDevices.getUserMedia({
              audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
              }
            });
            console.log('🎓 VideoTutor: Microphone audio capture successful');
          }
        }
      }

      this.audioContext = new AudioContext({ sampleRate: 16000 });
      const source = this.audioContext.createMediaStreamSource(stream);
      
      const overlay = this.overlays.get(video);
      if (overlay) {
        overlay.audioStream = stream;
        this.connectWebSocket(video);
      }

      this.processAudioStream(source, video);
      console.log('🎓 VideoTutor: Audio processing started');
      
    } catch (error) {
      console.error('🎓 VideoTutor: Failed to capture audio:', error);
      
      // Show user-friendly error
      const overlay = this.overlays.get(video);
      if (overlay) {
        const recordButton = overlay.element.querySelector('button') as HTMLButtonElement;
        recordButton.textContent = '❌ No Mic';
        recordButton.className = 'bg-gray-600 text-white px-3 py-1 rounded text-sm';
        recordButton.disabled = true;
      }
      
      throw error;
    }
  }

  private connectWebSocket(video: HTMLVideoElement) {
    try {
      const videoId = this.getVideoId();
      console.log('🎓 VideoTutor: Connecting WebSocket with videoId:', videoId);
      const ws = new WebSocket(`ws://localhost:8000/audio?video_id=${encodeURIComponent(videoId)}`);
      const overlay = this.overlays.get(video);
      
      if (overlay) {
        overlay.websocket = ws;
      }

      ws.onopen = () => {
        console.log('🎓 VideoTutor: Audio WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'transcription') {
            console.log('🎓 VideoTutor: Raw transcription received:', data.text);
            
            if (data.text.trim()) {
              console.log('🎓 VideoTutor: Adding transcription to storage:', data.text);
              
              // Store transcript in overlay
              if (overlay) {
                overlay.transcripts.push({
                  timestamp: data.timestamp || Date.now(),
                  text: data.text
                });
                
                console.log('🎓 VideoTutor: Total transcripts stored:', overlay.transcripts.length);
                
                // Update transcript panel if it's visible
                this.updateTranscriptPanel(video);
              }
            } else {
              console.log('🎓 VideoTutor: Skipping empty transcription');
            }
          }
        } catch (error) {
          console.error('🎓 VideoTutor: Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('🎓 VideoTutor: WebSocket error:', error);
      };

      ws.onclose = (event) => {
        console.log('🎓 VideoTutor: WebSocket closed:', event.code, event.reason);
      };
    } catch (error) {
      console.error('🎓 VideoTutor: Failed to create WebSocket:', error);
    }
  }

  private async processAudioStream(source: MediaStreamAudioSourceNode, video: HTMLVideoElement) {
    try {
      // Try to use modern AudioWorkletNode
      await this.audioContext!.audioWorklet.addModule('data:text/javascript,class%20AudioProcessor%20extends%20AudioWorkletProcessor%20%7B%0A%20%20process(inputs%2C%20outputs%2C%20parameters)%20%7B%0A%20%20%20%20const%20input%20%3D%20inputs%5B0%5D%3B%0A%20%20%20%20if%20(input%20%26%26%20input%5B0%5D)%20%7B%0A%20%20%20%20%20%20this.port.postMessage(%7B%20audioData%3A%20Array.from(input%5B0%5D)%20%7D)%3B%0A%20%20%20%20%7D%0A%20%20%20%20return%20true%3B%0A%20%20%7D%0A%7D%0AregisterProcessor(%27audio-processor%27%2C%20AudioProcessor)%3B');
      
      const workletNode = new AudioWorkletNode(this.audioContext!, 'audio-processor');
      workletNode.port.onmessage = (event) => {
        const { audioData } = event.data;
        const overlay = this.overlays.get(video);
        if (overlay?.websocket?.readyState === WebSocket.OPEN && audioData) {
          const pcmData = this.convertToPCM(new Float32Array(audioData));
          overlay.websocket.send(pcmData);
        }
      };
      
      source.connect(workletNode);
      workletNode.connect(this.audioContext!.destination);
    } catch (error) {
      console.log('🎓 VideoTutor: AudioWorklet not supported, falling back to ScriptProcessor', error);
      // Fallback to ScriptProcessor (with warning suppression)
      const processor = this.audioContext!.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        const overlay = this.overlays.get(video);
        if (overlay?.websocket?.readyState === WebSocket.OPEN) {
          const pcmData = this.convertToPCM(inputData);
          overlay.websocket.send(pcmData);
        }
      };

      source.connect(processor);
      processor.connect(this.audioContext!.destination);
    }
  }

  private convertToPCM(float32Array: Float32Array): ArrayBuffer {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new Int16Array(buffer);
    
    for (let i = 0; i < float32Array.length; i++) {
      const sample = Math.max(-1, Math.min(1, float32Array[i]));
      view[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    
    return buffer;
  }

  private async handleRecordClick(video: HTMLVideoElement) {
    const overlay = this.overlays.get(video);
    if (!overlay) return;

    const recordButton = overlay.element.querySelector('.ai-button-record') as HTMLButtonElement;
    
    try {
      if (!overlay.audioStream) {
        console.log('🎓 VideoTutor: Starting audio capture...');
        recordButton.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg>
          <span style="margin-left: 6px;">Stop</span>
        `;
        recordButton.classList.add('recording');
        
        await this.setupAudioCapture(video);
      } else {
        console.log('🎓 VideoTutor: Stopping audio capture...');
        recordButton.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="6"/>
          </svg>
          <span style="margin-left: 6px;">Record</span>
        `;
        recordButton.classList.remove('recording');
        
        this.stopAudioCapture(video);
      }
    } catch (error) {
      console.error('🎓 VideoTutor: Audio capture error:', error);
      recordButton.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 18L18 6M6 6l12 12"/>
        </svg>
        <span style="margin-left: 6px;">Failed</span>
      `;
      recordButton.classList.remove('recording');
      recordButton.style.opacity = '0.6';
      recordButton.style.cursor = 'not-allowed';
    }
  }

  private stopAudioCapture(video: HTMLVideoElement) {
    const overlay = this.overlays.get(video);
    if (!overlay) return;

    if (overlay.audioStream) {
      overlay.audioStream.getTracks().forEach(track => track.stop());
      overlay.audioStream = undefined;
    }

    if (overlay.websocket) {
      overlay.websocket.close();
      overlay.websocket = undefined;
    }
  }

  private handleAskClick(video: HTMLVideoElement) {
    const overlay = this.overlays.get(video);
    if (!overlay) return;

    const chatPanel = overlay.element.querySelector('.tutor-chat-panel') as HTMLDivElement;
    if (chatPanel.style.display === 'none') {
      chatPanel.style.display = 'flex';
    } else {
      chatPanel.style.display = 'none';
    }
  }

  private handleTranscriptClick(video: HTMLVideoElement) {
    const overlay = this.overlays.get(video);
    if (!overlay) return;

    const transcriptPanel = overlay.element.querySelector('.tutor-transcript-panel') as HTMLDivElement;
    if (transcriptPanel.style.display === 'none') {
      transcriptPanel.style.display = 'flex';
      this.updateTranscriptPanel(video);
    } else {
      transcriptPanel.style.display = 'none';
    }
  }

  private updateTranscriptPanel(video: HTMLVideoElement) {
    const overlay = this.overlays.get(video);
    if (!overlay) return;

    const transcriptPanel = overlay.element.querySelector('.tutor-transcript-panel') as HTMLDivElement;
    const transcriptContent = transcriptPanel.querySelector('.transcript-content') as HTMLDivElement;
    
    if (overlay.transcripts.length === 0) {
      transcriptContent.innerHTML = `
        <div style="color: #9CA3AF; font-style: italic; text-align: center; padding: 20px;">
          Start recording to see live transcript...
        </div>
      `;
      return;
    }

    // Format transcripts with timestamps
    const transcriptText = overlay.transcripts.map(t => {
      const time = new Date(t.timestamp).toLocaleTimeString();
      return `[${time}] ${t.text}`;
    }).join('\n\n');

    transcriptContent.textContent = transcriptText;
    transcriptContent.scrollTop = transcriptContent.scrollHeight;
  }

  private async handleSendMessage(input: HTMLInputElement, messagesContainer: HTMLDivElement) {
    const message = input.value.trim();
    if (!message) return;

    this.addMessage(messagesContainer, message, 'user');
    input.value = '';

    // Add loading message with dots animation
    const loadingElement = this.addLoadingMessage(messagesContainer);

    try {
      const videoId = this.getVideoId();
      const timestamp = this.getCurrentTimestamp();
      
      console.log('🎓 VideoTutor: Sending query with videoId:', videoId, 'message:', message);

      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          videoId,
          timestamp,
          prompt: message
        })
      });

      // Remove loading message
      loadingElement.remove();

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      let assistantMessage = '';
      const messageElement = this.addMessage(messagesContainer, '', 'assistant');

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.choices?.[0]?.delta?.content) {
                assistantMessage += parsed.choices[0].delta.content;
                const textDiv = messageElement.querySelector('div:last-child') as HTMLDivElement;
                if (textDiv) {
                  textDiv.innerHTML = this.parseMarkdown(assistantMessage);
                }
              }
            } catch (e) {
              // Ignore parsing errors
            }
          }
        }
      }
    } catch (error) {
      console.error('🎓 VideoTutor: Error sending message:', error);
      // Remove loading message if still there
      if (loadingElement.parentNode) {
        loadingElement.remove();
      }
      
      // Show more detailed error
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      this.addMessage(messagesContainer, `Error: ${errorMsg}`, 'error');
    }
  }

  private addLoadingMessage(container: HTMLDivElement): HTMLDivElement {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-loading slide-in-left';
    messageDiv.style.cssText = `
      padding: 16px 20px;
      margin-bottom: 12px;
      font-size: 14px;
      line-height: 1.5;
      display: flex;
      align-items: center;
      gap: 12px;
      animation-delay: 0.1s;
    `;

    const avatar = document.createElement('div');
    avatar.style.cssText = `
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: var(--ai-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    `;
    avatar.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
        <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
      </svg>
    `;

    const textContainer = document.createElement('div');
    textContainer.style.cssText = `
      flex: 1;
      display: flex;
      align-items: center;
      gap: 8px;
    `;

    const thinkingText = document.createElement('span');
    thinkingText.textContent = 'AI is thinking';
    thinkingText.style.cssText = `
      font-weight: 500;
      color: var(--text-secondary);
    `;

    const dotsContainer = document.createElement('div');
    dotsContainer.style.cssText = `
      display: flex;
      gap: 4px;
    `;

    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('div');
      dot.style.cssText = `
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--text-muted);
        animation: pulse 1.5s infinite;
        animation-delay: ${i * 0.2}s;
      `;
      dotsContainer.appendChild(dot);
    }

    textContainer.appendChild(thinkingText);
    textContainer.appendChild(dotsContainer);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(textContainer);
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv;
  }

  private parseMarkdown(text: string): string {
    return text
      // Headers
      .replace(/^### (.*$)/gm, '<h3 style="margin: 8px 0 4px 0; font-size: 16px; font-weight: 600; color: var(--text-primary);">$1</h3>')
      .replace(/^## (.*$)/gm, '<h2 style="margin: 10px 0 6px 0; font-size: 18px; font-weight: 600; color: var(--text-primary);">$1</h2>')
      .replace(/^# (.*$)/gm, '<h1 style="margin: 12px 0 8px 0; font-size: 20px; font-weight: 600; color: var(--text-primary);">$1</h1>')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong style="font-weight: 600; color: var(--text-primary);">$1</strong>')
      // Italic
      .replace(/\*(.*?)\*/g, '<em style="font-style: italic; color: var(--text-secondary);">$1</em>')
      // Code
      .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; color: var(--text-primary);">$1</code>')
      // Bullets
      .replace(/^- (.*$)/gm, '<div style="margin: 4px 0; padding-left: 16px; position: relative;"><span style="position: absolute; left: 0; color: var(--text-primary);">•</span>$1</div>')
      // Numbers
      .replace(/^\d+\. (.*$)/gm, '<div style="margin: 4px 0; padding-left: 20px; position: relative;"><span style="position: absolute; left: 0; color: var(--text-primary); font-weight: 500;">1.</span>$1</div>')
      // Line breaks
      .replace(/\n/g, '<br>');
  }

  private addMessage(container: HTMLDivElement, text: string, type: 'user' | 'assistant' | 'error'): HTMLDivElement {
    const messageDiv = document.createElement('div');
    const animationClass = type === 'user' ? 'slide-in-right' : 'slide-in-left';
    messageDiv.className = `message-${type} ${animationClass}`;
    
    if (type === 'user') {
      messageDiv.style.cssText = `
        padding: 12px 18px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
        animation-delay: 0.05s;
      `;
      messageDiv.textContent = text;
    } else if (type === 'assistant') {
      messageDiv.style.cssText = `
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
        word-wrap: break-word;
        display: flex;
        gap: 12px;
        animation-delay: 0.1s;
      `;

      const avatar = document.createElement('div');
      avatar.style.cssText = `
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--ai-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        align-self: flex-start;
      `;
      avatar.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
          <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
        </svg>
      `;

      const textContent = document.createElement('div');
      textContent.style.cssText = `
        flex: 1;
        padding-top: 2px;
      `;
      textContent.innerHTML = this.parseMarkdown(text);

      messageDiv.appendChild(avatar);
      messageDiv.appendChild(textContent);
    } else { // error
      messageDiv.style.cssText = `
        padding: 12px 18px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 90%;
        margin: 0 auto 12px auto;
        word-wrap: break-word;
        text-align: center;
        animation-delay: 0.1s;
      `;
      messageDiv.textContent = text;
    }
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    return messageDiv;
  }

  private getVideoId(): string {
    const url = window.location.href;
    const youtubeMatch = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);
    return youtubeMatch ? youtubeMatch[1] : url;
  }

  private getCurrentTimestamp(): number {
    const video = document.querySelector('video') as HTMLVideoElement;
    return video ? video.currentTime : 0;
  }

  /*
  private setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
      switch (request.action) {
        case 'toggleOverlay':
          // Handle overlay toggle from popup
          sendResponse({ success: true });
          break;
        case 'toggleExtension':
          // Handle extension state toggle
          sendResponse({ success: true });
          break;
        default:
          // Don't handle unknown actions
          break;
      }
      // Always return false for synchronous response
      return false;
    });
  }
  */
}

new VideoTutor();
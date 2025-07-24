interface VideoTutorOverlay {
  element: HTMLDivElement;
  isVisible: boolean;
  audioStream?: MediaStream;
  websocket?: WebSocket;
  transcripts: Array<{timestamp: number, text: string}>;
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
    controlBar.className = 'tutor-control-bar';
    controlBar.style.cssText = `
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
    `;
    
    // Make control bar draggable
    this.makeDraggable(controlBar);

    const recordButton = document.createElement('button');
    recordButton.textContent = '🎤 Record';
    recordButton.style.cssText = `
      background: #dc2626;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `;
    recordButton.onclick = () => this.handleRecordClick(video);

    const askButton = document.createElement('button');
    askButton.textContent = 'Ask';
    askButton.style.cssText = `
      background: #2563eb;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `;
    askButton.onclick = () => this.handleAskClick(video);

    const transcriptButton = document.createElement('button');
    transcriptButton.textContent = '📝 Transcript';
    transcriptButton.style.cssText = `
      background: #059669;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
      margin-right: 4px;
    `;
    transcriptButton.onclick = () => this.handleTranscriptClick(video);

    const settingsButton = document.createElement('button');
    settingsButton.textContent = '⚙️';
    settingsButton.style.cssText = `
      background: #6b7280;
      color: white;
      padding: 8px 12px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      pointer-events: auto;
    `;

    controlBar.appendChild(recordButton);
    controlBar.appendChild(askButton);
    controlBar.appendChild(transcriptButton);
    controlBar.appendChild(settingsButton);

    return controlBar;
  }

  private createChatPanel(): HTMLDivElement {
    const chatPanel = document.createElement('div');
    chatPanel.className = 'tutor-chat-panel';
    chatPanel.style.cssText = `
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
    `;
    
    // Make chat panel draggable
    this.makeDraggable(chatPanel);

    const messagesContainer = document.createElement('div');
    messagesContainer.className = 'messages-container';
    messagesContainer.style.cssText = `
      flex: 1;
      overflow-y: auto;
      max-height: 300px;
    `;

    const inputContainer = document.createElement('div');
    inputContainer.style.cssText = `
      display: flex;
      gap: 8px;
    `;

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Ask about the video...';
    input.className = 'flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white';
    input.style.cssText = `
      flex: 1;
      padding: 8px 12px;
      background: #374151;
      border: 1px solid #4B5563;
      border-radius: 4px;
      color: white;
      outline: none;
    `;

    const sendButton = document.createElement('button');
    sendButton.textContent = 'Send';
    sendButton.style.cssText = `
      background: #2563eb;
      color: white;
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    `;

    input.addEventListener('keydown', (e) => {
      // Prevent YouTube keyboard shortcuts when typing
      e.stopPropagation();
      
      if (e.key === 'Enter') {
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

    chatPanel.appendChild(messagesContainer);
    chatPanel.appendChild(inputContainer);

    return chatPanel;
  }

  private createTranscriptPanel(): HTMLDivElement {
    const transcriptPanel = document.createElement('div');
    transcriptPanel.className = 'tutor-transcript-panel';
    transcriptPanel.style.cssText = `
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
    `;
    
    // Make transcript panel draggable
    this.makeDraggable(transcriptPanel);

    const header = document.createElement('div');
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Live Transcript';
    title.style.cssText = `
      margin: 0;
      font-size: 16px;
      font-weight: 600;
    `;

    const closeButton = document.createElement('button');
    closeButton.textContent = '×';
    closeButton.style.cssText = `
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
      max-height: 320px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      padding: 8px;
      font-size: 14px;
      line-height: 1.4;
      white-space: pre-wrap;
    `;

    const placeholder = document.createElement('div');
    placeholder.textContent = 'Start recording to see live transcript...';
    placeholder.style.cssText = `
      color: #9CA3AF;
      font-style: italic;
      text-align: center;
      padding: 20px;
    `;
    transcriptContent.appendChild(placeholder);

    transcriptPanel.appendChild(header);
    transcriptPanel.appendChild(transcriptContent);

    return transcriptPanel;
  }

  private makeDraggable(element: HTMLElement) {
    let isDragging = false;
    let currentX = 0;
    let currentY = 0;
    let initialX = 0;
    let initialY = 0;
    let xOffset = 0;
    let yOffset = 0;

    const dragStart = (e: MouseEvent) => {
      // Only allow dragging if not clicking on buttons or inputs
      const target = e.target as HTMLElement;
      if (target.tagName === 'BUTTON' || target.tagName === 'INPUT') {
        return;
      }

      initialX = e.clientX - xOffset;
      initialY = e.clientY - yOffset;

      if (e.target === element) {
        isDragging = true;
        element.style.cursor = 'grabbing';
      }
    };

    const dragEnd = () => {
      initialX = currentX;
      initialY = currentY;
      isDragging = false;
      element.style.cursor = 'move';
    };

    const drag = (e: MouseEvent) => {
      if (isDragging) {
        e.preventDefault();
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        xOffset = currentX;
        yOffset = currentY;

        element.style.left = `${currentX}px`;
        element.style.top = `${currentY}px`;
        element.style.right = 'auto';
        element.style.bottom = 'auto';
      }
    };

    element.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
  }

  private async setupAudioCapture(video: HTMLVideoElement) {
    try {
      console.log('🎓 VideoTutor: Requesting audio capture permissions...');
      
      // Try getDisplayMedia with audio first
      let stream;
      try {
        stream = await navigator.mediaDevices.getDisplayMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000
          },
          video: false
        });
        console.log('🎓 VideoTutor: Screen audio capture successful');
      } catch (displayError) {
        console.log('🎓 VideoTutor: Screen audio failed, trying microphone...', displayError);
        
        // Fallback to getUserMedia for microphone
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000
          }
        });
        console.log('🎓 VideoTutor: Microphone audio capture successful');
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
      const ws = new WebSocket('ws://localhost:8000/audio');
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
          if (data.type === 'transcription' && data.text.trim()) {
            console.log('🎓 VideoTutor: Transcription:', data.text);
            
            // Store transcript in overlay
            if (overlay) {
              overlay.transcripts.push({
                timestamp: data.timestamp || Date.now(),
                text: data.text
              });
              
              // Update transcript panel if it's visible
              this.updateTranscriptPanel(video);
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

  private processAudioStream(source: MediaStreamAudioSourceNode, video: HTMLVideoElement) {
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

    const recordButton = overlay.element.querySelector('button') as HTMLButtonElement;
    
    try {
      if (!overlay.audioStream) {
        console.log('🎓 VideoTutor: Starting audio capture...');
        recordButton.textContent = '⏸️ Stop';
        recordButton.className = 'bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm';
        
        await this.setupAudioCapture(video);
      } else {
        console.log('🎓 VideoTutor: Stopping audio capture...');
        recordButton.textContent = '🎤 Record';
        recordButton.className = 'bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm';
        
        this.stopAudioCapture(video);
      }
    } catch (error) {
      console.error('🎓 VideoTutor: Audio capture error:', error);
      recordButton.textContent = '❌ Failed';
      recordButton.className = 'bg-gray-600 text-white px-3 py-1 rounded text-sm';
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
                messageElement.textContent = assistantMessage;
              }
            } catch (e) {
              // Ignore parsing errors
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Remove loading message if still there
      if (loadingElement.parentNode) {
        loadingElement.remove();
      }
      this.addMessage(messagesContainer, 'Error: Could not send message', 'error');
    }
  }

  private addLoadingMessage(container: HTMLDivElement): HTMLDivElement {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-loading';
    messageDiv.style.cssText = `
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
    `;

    const dotsContainer = document.createElement('span');
    dotsContainer.innerHTML = 'Thinking<span class="loading-dots"></span>';
    
    const style = document.createElement('style');
    style.textContent = `
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
    `;
    
    if (!document.head.querySelector('style[data-loading-dots]')) {
      style.setAttribute('data-loading-dots', 'true');
      document.head.appendChild(style);
    }
    
    messageDiv.appendChild(dotsContainer);
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv;
  }

  private addMessage(container: HTMLDivElement, text: string, type: 'user' | 'assistant' | 'error'): HTMLDivElement {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-${type}`;
    messageDiv.style.cssText = `
      padding: 8px 12px;
      border-radius: 8px;
      margin-bottom: 8px;
      background: ${type === 'user' ? '#2563EB' : type === 'error' ? '#DC2626' : '#374151'};
      color: white;
      font-size: 14px;
      line-height: 1.4;
    `;
    messageDiv.textContent = text;
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
interface VideoTutorOverlay {
  element: HTMLDivElement;
  isVisible: boolean;
  audioStream?: MediaStream;
  websocket?: WebSocket;
}

class VideoTutor {
  private overlays: Map<HTMLVideoElement, VideoTutorOverlay> = new Map();
  private audioContext?: AudioContext;

  constructor() {
    this.init();
  }

  private init() {
    this.observeVideos();
    this.setupMessageListener();
  }

  private observeVideos() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const videos = (node as Element).querySelectorAll('video');
            videos.forEach((video) => this.attachOverlay(video as HTMLVideoElement));
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    document.querySelectorAll('video').forEach((video) => {
      this.attachOverlay(video as HTMLVideoElement);
    });
  }

  private attachOverlay(video: HTMLVideoElement) {
    if (this.overlays.has(video)) return;

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

    video.parentElement?.style.setProperty('position', 'relative');
    video.parentElement?.appendChild(overlayContainer);

    const overlay: VideoTutorOverlay = {
      element: overlayContainer,
      isVisible: false
    };

    this.overlays.set(video, overlay);
    this.setupAudioCapture(video);
  }

  private createControlBar(video: HTMLVideoElement): HTMLDivElement {
    const controlBar = document.createElement('div');
    controlBar.className = 'tutor-control-bar';
    controlBar.style.cssText = `
      position: absolute;
      bottom: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      gap: 8px;
      pointer-events: auto;
    `;

    const askButton = document.createElement('button');
    askButton.textContent = 'Ask';
    askButton.className = 'bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm';
    askButton.style.pointerEvents = 'auto';
    askButton.onclick = () => this.handleAskClick(video);

    const settingsButton = document.createElement('button');
    settingsButton.textContent = '⚙️';
    settingsButton.className = 'bg-gray-600 hover:bg-gray-700 text-white px-3 py-1 rounded text-sm';
    settingsButton.style.pointerEvents = 'auto';

    controlBar.appendChild(askButton);
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
      background: rgba(0, 0, 0, 0.9);
      border-radius: 8px;
      padding: 16px;
      display: none;
      flex-direction: column;
      gap: 12px;
      pointer-events: auto;
      color: white;
      font-family: system-ui, sans-serif;
    `;

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
    sendButton.className = 'bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded';

    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.handleSendMessage(input, messagesContainer);
      }
    });

    sendButton.onclick = () => this.handleSendMessage(input, messagesContainer);

    inputContainer.appendChild(input);
    inputContainer.appendChild(sendButton);

    chatPanel.appendChild(messagesContainer);
    chatPanel.appendChild(inputContainer);

    return chatPanel;
  }

  private async setupAudioCapture(video: HTMLVideoElement) {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: false
      });

      this.audioContext = new AudioContext();
      const source = this.audioContext.createMediaStreamSource(stream);
      
      const overlay = this.overlays.get(video);
      if (overlay) {
        overlay.audioStream = stream;
        this.connectWebSocket(video);
      }

      this.processAudioStream(source, video);
    } catch (error) {
      console.error('Failed to capture audio:', error);
    }
  }

  private connectWebSocket(video: HTMLVideoElement) {
    const ws = new WebSocket('ws://localhost:8000/audio');
    const overlay = this.overlays.get(video);
    
    if (overlay) {
      overlay.websocket = ws;
    }

    ws.onopen = () => {
      console.log('Audio WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'transcription') {
        console.log('Transcription:', data.text);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
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

  private async handleSendMessage(input: HTMLInputElement, messagesContainer: HTMLDivElement) {
    const message = input.value.trim();
    if (!message) return;

    this.addMessage(messagesContainer, message, 'user');
    input.value = '';

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
      this.addMessage(messagesContainer, 'Error: Could not send message', 'error');
    }
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

  private setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === 'toggleOverlay') {
        // Handle overlay toggle from popup
      }
      return true;
    });
  }
}

new VideoTutor();
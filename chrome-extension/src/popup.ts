interface ExtensionState {
  active: boolean;
  autoTranscribe: boolean;
  showOverlay: boolean;
}

class PopupController {
  private state: ExtensionState = {
    active: false,
    autoTranscribe: true,
    showOverlay: true
  };

  constructor() {
    this.init();
  }

  private async init() {
    await this.loadState();
    this.setupEventListeners();
    this.updateUI();
  }

  private async loadState() {
    const result = await chrome.storage.sync.get(['extensionState']);
    if (result.extensionState) {
      this.state = { ...this.state, ...result.extensionState };
    }
  }

  private async saveState() {
    await chrome.storage.sync.set({ extensionState: this.state });
  }

  private setupEventListeners() {
    const toggleButton = document.getElementById('toggleButton') as HTMLButtonElement;
    const testButton = document.getElementById('testConnection') as HTMLButtonElement;
    const autoTranscribeCheckbox = document.getElementById('autoTranscribe') as HTMLInputElement;
    const showOverlayCheckbox = document.getElementById('showOverlay') as HTMLInputElement;

    toggleButton.addEventListener('click', () => this.toggleExtension());
    testButton.addEventListener('click', () => this.testConnection());
    
    autoTranscribeCheckbox.addEventListener('change', (e) => {
      this.state.autoTranscribe = (e.target as HTMLInputElement).checked;
      this.saveState();
    });

    showOverlayCheckbox.addEventListener('change', (e) => {
      this.state.showOverlay = (e.target as HTMLInputElement).checked;
      this.saveState();
    });
  }

  private async toggleExtension() {
    this.state.active = !this.state.active;
    await this.saveState();
    
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab.id) {
      chrome.tabs.sendMessage(tab.id, { 
        action: 'toggleExtension', 
        state: this.state 
      });
    }
    
    this.updateUI();
  }

  private async testConnection() {
    const testButton = document.getElementById('testConnection') as HTMLButtonElement;
    testButton.disabled = true;
    testButton.textContent = 'Testing...';

    try {
      const response = await fetch('http://localhost:8000/health');
      if (response.ok) {
        testButton.textContent = '✅ Connected';
        testButton.style.background = '#059669';
      } else {
        throw new Error('Backend not responding');
      }
    } catch (error) {
      testButton.textContent = '❌ Failed';
      testButton.style.background = '#dc2626';
    }

    setTimeout(() => {
      testButton.disabled = false;
      testButton.textContent = 'Test Backend Connection';
      testButton.style.background = '#3b82f6';
    }, 2000);
  }

  private updateUI() {
    const statusDiv = document.getElementById('status') as HTMLDivElement;
    const toggleButton = document.getElementById('toggleButton') as HTMLButtonElement;
    const autoTranscribeCheckbox = document.getElementById('autoTranscribe') as HTMLInputElement;
    const showOverlayCheckbox = document.getElementById('showOverlay') as HTMLInputElement;

    if (this.state.active) {
      statusDiv.textContent = 'Extension Active';
      statusDiv.className = 'status active';
      toggleButton.textContent = 'Deactivate Extension';
    } else {
      statusDiv.textContent = 'Extension Inactive';
      statusDiv.className = 'status inactive';
      toggleButton.textContent = 'Activate Extension';
    }

    autoTranscribeCheckbox.checked = this.state.autoTranscribe;
    showOverlayCheckbox.checked = this.state.showOverlay;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new PopupController();
});
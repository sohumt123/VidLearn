chrome.runtime.onInstalled.addListener(() => {
  console.log('AI Video Tutor extension installed');
});

chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.tabs.sendMessage(tab.id, { action: 'toggleOverlay' });
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'requestTabCapture') {
    // Use chrome.tabCapture.getMediaStreamId for newer approach
    if (chrome.tabCapture.getMediaStreamId) {
      chrome.tabCapture.getMediaStreamId({
        consumerTabId: sender.tab?.id
      }, (streamId) => {
        if (chrome.runtime.lastError) {
          console.error('Tab capture error:', chrome.runtime.lastError);
          sendResponse({ error: chrome.runtime.lastError.message });
        } else {
          sendResponse({ streamId });
        }
      });
    } else {
      // Fallback to older capture method
      chrome.tabCapture.capture(
        {
          audio: true,
          video: false
        },
        (stream) => {
          if (chrome.runtime.lastError) {
            console.error('Tab capture error:', chrome.runtime.lastError);
            sendResponse({ error: chrome.runtime.lastError.message });
          } else {
            sendResponse({ stream });
          }
        }
      );
    }
    return true;
  }

  if (request.action === 'getTabAudio') {
    chrome.tabCapture.capture(
      {
        audio: true,
        video: false
      },
      (stream) => {
        if (chrome.runtime.lastError) {
          console.error('Tab capture error:', chrome.runtime.lastError);
          sendResponse({ error: chrome.runtime.lastError.message });
        } else {
          sendResponse({ stream });
        }
      }
    );
    return true;
  }

  if (request.action === 'startRecording') {
    sendResponse({ success: true });
  }

  if (request.action === 'stopRecording') {
    sendResponse({ success: true });
  }

  return true;
});
import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [processedFrame, setProcessedFrame] = useState('');
  const [processingActive, setProcessingActive] = useState(false);
  const [showStopMessage, setShowStopMessage] = useState(false);

  const startVideoProcessing = () => {
    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          const video = videoRef.current;
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            setProcessingActive(true);
            setShowStopMessage(false);
            const intervalId = setInterval(() => {
              sendFrameToServer(video);
            }, 1000 / 30); // Send frame to server every 1/30 seconds (30fps)
            // Store interval ID to clear interval later
            sessionStorage.setItem('intervalId', intervalId);
          };
        })
        .catch(error => {
          console.error("Something went wrong!", error);
        });
    }
  };

  const stopVideoProcessing = () => {
    const intervalId = sessionStorage.getItem('intervalId');
    clearInterval(intervalId);
    setProcessingActive(false);
    setShowStopMessage(true);
    const video = videoRef.current;
    if (video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    video.srcObject = null;
    setProcessedFrame(''); // Clear processed frame
  };

  const sendFrameToServer = (video) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');

    fetch('http://127.0.0.1:5001/video', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ image: imageData }),
    })
      .then(response => response.text())
      .then(data => {
        setProcessedFrame(`data:image/jpeg;base64,${data}`);
      })
      .catch(error => console.error('Error:', error));
  };

  return (
    <div className="App">
      <h1>Live ASL translator</h1>
      {!processingActive ? (
        <button onClick={startVideoProcessing}>Start Video Processing</button>
      ) : (
        <button onClick={stopVideoProcessing}>Stop Video Processing</button>
      )}
      <video ref={videoRef} style={{ display: 'none' }}></video>
      {showStopMessage && <p>Video processing has now stopped</p>}
      {processedFrame && !showStopMessage && <img id="processedFrame" src={processedFrame} alt="Processed Frame" width="50%" />}
    </div>
  );
}

export default App;

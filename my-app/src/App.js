import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [processedFrame, setProcessedFrame] = useState('');
  const [processingActive, setProcessingActive] = useState(false);
  const [showStopMessage, setShowStopMessage] = useState(false);
  const [aslString, setAslString] = useState('');
  const [predictedCharacter, setPredictedCharacter] = useState('');
  const [leftHandPresent, setLeftHandPresent] = useState(false);
  const lastLeftHandPresentRef = useRef(false); // Use ref to track the last left hand presence

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
    setPredictedCharacter(''); // Clear predicted character
  };

  const sendFrameToServer = (video) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');

    fetch('http://127.0.0.1:5002/video', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ image: imageData }),
    })
      .then(response => response.json())
      .then(data => {
        setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
        setPredictedCharacter(data.predicted_character); // Set the predicted character
        setLeftHandPresent(data.left_hand_present); // Set the left hand presence
      })
      .catch(error => console.error('Error:', error));
  };

  const clearAslString = () => {
    setAslString('');
  };

  useEffect(() => {
    if (predictedCharacter && leftHandPresent && !lastLeftHandPresentRef.current) {
      console.log("Predicted Character:", predictedCharacter);
      console.log("ASL string:", aslString);
      setAslString(prevString => prevString + predictedCharacter);
      lastLeftHandPresentRef.current = leftHandPresent; // Update the ref
      console.log("Updated Last Left Hand Present:", lastLeftHandPresentRef.current);
    } else if (!leftHandPresent) {
      lastLeftHandPresentRef.current = false; // Update the ref when the left hand is not present
    }
  }, [predictedCharacter, leftHandPresent, aslString]);

  return (
    <div className="App">
      <h1>Live ASL Translator</h1>
      {!processingActive ? (
        <button onClick={startVideoProcessing}>Start Video Processing</button>
      ) : (
        <button onClick={stopVideoProcessing}>Stop Video Processing</button>
      )}
      <video ref={videoRef} style={{ display: 'none' }}></video>
      {showStopMessage && <p>Video processing has now stopped</p>}
      {processedFrame && !showStopMessage && <img id="processedFrame" src={processedFrame} alt="Processed Frame" width="50%" />}
      <p>Predicted Character: {predictedCharacter}</p>
      <p>Translated Text: {aslString}</p>
      <button onClick={clearAslString}>Clear</button>
    </div>
  );
}

export default App;

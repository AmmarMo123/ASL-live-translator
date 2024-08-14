import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import './output.css';
import StartStopButton from './components/StartStopButton';
import ProcessedFrame from './components/ProcessedFrame';
import PredictedCharacter from './components/PredictedCharacter';
import TranslatedText from './components/TranslatedText';

function App() {
  const videoRef = useRef(null);
  const [processedFrame, setProcessedFrame] = useState('');
  const [processingActive, setProcessingActive] = useState(false);
  const [showStopMessage, setShowStopMessage] = useState(false);
  const [aslString, setAslString] = useState('');
  const [predictedCharacter, setPredictedCharacter] = useState('');
  const [leftHandPresent, setLeftHandPresent] = useState(false);
  const lastLeftHandPresentRef = useRef(false); // Use ref to track the last left hand presence

  // Function to start video processing
  const startVideoProcessing = () => {
    if (navigator.mediaDevices.getUserMedia) {
      // Request access to the user's webcam
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          const video = videoRef.current;
          video.srcObject = stream; // Set the video source to the webcam stream
          video.onloadedmetadata = () => {
            video.play(); // Play the video
            setProcessingActive(true); // Set the processing state to active
            setShowStopMessage(false); // Hide the stop message
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

  // Function to stop video processing
  const stopVideoProcessing = () => {
    const intervalId = sessionStorage.getItem('intervalId');
    clearInterval(intervalId); // Stop sending frames to the server
    setProcessingActive(false); // Set the processing state to inactive
    setShowStopMessage(true); // Show the stop message

    // Stop the webcam video stream
    const video = videoRef.current;
    if (video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    video.srcObject = null; // Clear the video source
    setProcessedFrame(''); // Clear processed frame
    setPredictedCharacter(''); // Clear predicted character
  };

  // Function to capture a video frame and send it to the server
  const sendFrameToServer = (video) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    // Set canvas dimensions to match the video frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame on the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a base64-encoded JPEG
    const imageData = canvas.toDataURL('image/jpeg');

    // Send the image to the server for processing
    fetch('http://127.0.0.1:5002/video', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({ image: imageData }), // Send the image data in the request body
    })
      .then(response => response.json())
      .then(data => {
        // Update the state with the processed frame and prediction results
        setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
        setPredictedCharacter(data.predicted_character); // Set the predicted character
        setLeftHandPresent(data.left_hand_present); // Set the left hand presence
      })
      .catch(error => console.error('Error:', error));
  };

  // Function to clear the accumulated ASL string
  const clearAslString = () => {
    setAslString('');
  };

  // Effect to update the ASL string when a new character is predicted
  useEffect(() => {
    if (predictedCharacter && leftHandPresent && !lastLeftHandPresentRef.current) {
      // Append the predicted character to the ASL string
      setAslString(prevString => prevString + predictedCharacter);
      lastLeftHandPresentRef.current = leftHandPresent; // Update the ref to indicate the left hand was present
    } else if (!leftHandPresent) {
      // Reset the ref when the left hand is not present
      lastLeftHandPresentRef.current = false; // Update the ref when the left hand is not present
    }
  }, [predictedCharacter, leftHandPresent, aslString]);

  return (
    <div class="App">
      <h1 class="text-3xl font-bold mb-4">Live ASL Translator</h1>
      <StartStopButton 
        processingActive={processingActive}
        startVideoProcessing={startVideoProcessing}
        stopVideoProcessing={stopVideoProcessing}
      />
      <video ref={videoRef} style={{ display: 'none' }}></video>
      {showStopMessage && <p className="text-xl mb-4 text-red-600">Video processing has now stopped</p>}
      <ProcessedFrame
        processedFrame={processedFrame} 
        showStopMessage={showStopMessage}
      />
      <PredictedCharacter predictedCharacter={predictedCharacter} />
      <TranslatedText 
        aslString={aslString}
        clearAslString={clearAslString}
      />
    </div>
  );
}

export default App;

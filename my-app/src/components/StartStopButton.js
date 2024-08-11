import React from 'react';

const StartStopButton = ({ processingActive, startVideoProcessing, stopVideoProcessing }) => {
  return (
    !processingActive ? (
      <button 
        onClick={startVideoProcessing} 
        className="bg-emerald-500 text-white py-2 px-4 rounded mb-2 hover:bg-emerald-700"
      >
        Start Video Processing
      </button>
    ) : (
      <button 
        onClick={stopVideoProcessing} 
        className="bg-red-500 text-white py-2 px-4 rounded mb-2 hover:bg-red-700"
      >
        Stop Video Processing
      </button>
    )
  );
}

export default StartStopButton;

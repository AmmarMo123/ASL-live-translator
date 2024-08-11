import React from 'react';

const ProcessedFrame = ({ processedFrame, showStopMessage }) => {
  return (
    processedFrame && !showStopMessage && <img id="processedFrame" src={processedFrame} alt="Processed Frame" className="w-1/3 mb-2" />
  );
}

export default ProcessedFrame;

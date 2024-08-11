import React from 'react';

const PredictedCharacter = ({ predictedCharacter }) => {
  return (
    <p className="text-xl mb-2">
      Predicted Character: <span className="font-semibold">{predictedCharacter}</span>
    </p>
  );
}

export default PredictedCharacter;

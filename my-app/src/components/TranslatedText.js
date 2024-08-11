import React from 'react';

const TranslatedText = ({ aslString, clearAslString }) => {
  return (
    <>
      <p className="text-xl mb-2">Translated Text: <span>{aslString}</span></p>
      <button 
        onClick={clearAslString} 
        className="bg-zinc-600 text-white py-2 px-4 rounded hover:bg-zinc-700"
      >
        Clear
      </button>
    </>
  );
}

export default TranslatedText;
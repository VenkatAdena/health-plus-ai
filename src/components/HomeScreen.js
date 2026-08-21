import React from 'react';

/**
 * HomeScreen Component
 * 
 * A simple home screen component that displays a welcome message
 * and a button with the text "Click me!"
 */
const HomeScreen = () => {
  const handleButtonClick = () => {
    console.log('Button clicked!');
    alert('Hello! You clicked the button!');
  };

  return (
    <div className="home-screen">
      <div className="container">
        <h1>Welcome to Health Plus AI</h1>
        <button onClick={handleButtonClick} className="click-button">
          Click me!
        </button>
      </div>
    </div>
  );
};

export default HomeScreen;

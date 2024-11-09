// App.js
import React, { useMemo } from "react";
import Canvas from "./Canvas";
import ProbabilityDisplay from "./ProbabilityDisplay";
import "./Apps.css";

export default function App() {
  const colors = useMemo(
    () => ["black", "red", "green", "orange", "blue", "yellow"],
    []
  );

  return (
    <div className="App">
      <h1>Digit Recognition Canvas</h1>
      <div className="content-container">
        <Canvas colors={colors} />
        <ProbabilityDisplay />
      </div>
    </div>
  );
}

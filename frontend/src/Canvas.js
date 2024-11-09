import { useEffect, useRef, useState, useCallback } from "react";
import { useDispatch } from "react-redux";
import ort from "onnxruntime-web";
import { throttle } from "lodash";
import { setProbability } from "./redux/actions";

export default function Canvas({ colors }) {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [color, setColor] = useState(colors[0]);
  const [lineWidth, setLineWidth] = useState(15);
  const dispatch = useDispatch();
  const modelRef = useRef(null); // Reference to the ONNX model session

  // Load the ONNX model when the component mounts
  useEffect(() => {
    const loadModel = async () => {
      try {
        modelRef.current = await ort.InferenceSession.create("/mnist_model.onnx");
        console.log("ONNX model loaded successfully");
      } catch (error) {
        console.error("Error loading ONNX model:", error);
      }
    };
    loadModel();
  }, []);

  // Clear the canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas && contextRef.current) {
      contextRef.current.fillStyle = "white";
      contextRef.current.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 800;
    canvas.height = 800;
    canvas.style.width = "100%";
    canvas.style.height = "auto";

    const context = canvas.getContext("2d");
    context.lineCap = "round";
    context.lineWidth = lineWidth;
    context.strokeStyle = color;
    contextRef.current = context;

    clearCanvas();
  }, [clearCanvas, color, lineWidth]);

  // Handle key down events for clearing the canvas
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "x" || event.key === "X") {
        clearCanvas();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [clearCanvas]);

  // Get mouse position
  const getMousePosition = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  };

  // Start drawing
  const startDrawing = (event) => {
    const { x, y } = getMousePosition(event);
    contextRef.current.beginPath();
    contextRef.current.moveTo(x, y);
    setIsDrawing(true);
  };

  // Draw on the canvas
  const draw = (event) => {
    if (!isDrawing) return;
    const { x, y } = getMousePosition(event);
    contextRef.current.lineTo(x, y);
    contextRef.current.stroke();
    sendThrottledRequest(); // Send the API request at regular intervals while drawing
  };

  const finishDrawing = () => {
    contextRef.current.closePath();
    setIsDrawing(false);
    // sendThrottledRequest(); // Send request after finishing a stroke
  };

  // Handle color change
  const handleColorChange = (newColor) => {
    setColor(newColor);
    contextRef.current.strokeStyle = newColor;
  };

  // Handle line width change
  const handleLineWidthChange = (event) => {
    const newLineWidth = event.target.value;
    setLineWidth(newLineWidth);
    contextRef.current.lineWidth = newLineWidth;
  };


  // Preprocess canvas image and run ONNX model inference
  const sendThrottledRequest = useCallback(
    throttle(async () => {
      const canvas = canvasRef.current;

      console.log("CANVAS: " + getResizedImageData().data)

      if (modelRef.current && canvas) {
        // Get image data from canvas and preprocess it for the model
        const imageData = canvas.getContext("2d").getImageData(0, 0, 28, 28); // Resize to 28x28 if needed by your model
        const inputTensor = preprocessImage(imageData);

        console.log("preprocessImage: " + inputTensor)


        try {
          const feeds = { input: inputTensor };
          const results = await modelRef.current.run(feeds);
          console.log(results.output.data)
          const probabilities = Array.from(results.output.data);
          
          dispatch(setProbability(probabilities));
        } catch (error) {
          console.error("Error running ONNX model inference:", error);
        }
      }
    }, 300), // Adjust throttle interval as needed
    [dispatch]
  );

  // // Preprocess the canvas image data for ONNX input
  // const preprocessImage = (imageData) => {
  //   const { data, width, height } = imageData;
  //   const floatArray = new Float32Array(width * height);

  //   for (let i = 0; i < width * height; i++) {
  //     const r = data[i * 4] / 255; // Red channel
  //     const g = data[i * 4 + 1] / 255; // Green channel
  //     const b = data[i * 4 + 2] / 255; // Blue channel
  //     const gray = 0.3 * r + 0.59 * g + 0.11 * b; // Convert to grayscale
  //     floatArray[i] = gray;
  //   }

  //   return new ort.Tensor("float32", floatArray, [1, 1, width, height]); // Shape: [batch_size, channels, width, height]
  // };
  const getResizedImageData = () => {
    const originalCanvas = canvasRef.current;
  
    // Step 1: Create a temporary canvas of size 28x28
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext("2d");
  
    // Step 2: Draw the original canvas onto the temporary canvas, resizing it
    tempContext.drawImage(originalCanvas, 0, 0, 28, 28);
  
    // Step 3: Get the resized image data from the temporary canvas
    const resizedImageData = tempContext.getImageData(0, 0, 28, 28).data;
  
    return { data: resizedImageData, width: 28, height: 28 };
  };
  

  const preprocessImage = (imageData) => {
    const { data, width, height } = getResizedImageData();
    console.log("Raw pixel data:", data); // Log raw data

    const floatArray = new Float32Array(width * height);
  
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      const gray = 0.3 * r + 0.59 * g + 0.11 * b;
      floatArray[i] = gray;
    }
  
    console.log("Processed grayscale data: " + floatArray); // Log grayscale data
  
    return new ort.Tensor("float32", floatArray, [1, 1, width, height]);
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={finishDrawing}
        onMouseLeave={finishDrawing}
      />
      <div className="controls">
        <button onClick={clearCanvas}>Clear Canvas</button>
        {colors.map((colorOption) => (
          <button
            key={colorOption}
            onClick={() => handleColorChange(colorOption)}
            style={{ backgroundColor: colorOption }}
            className={colorOption === color ? "active-color" : ""}
          />
        ))}
        <div className="line-width-slider">
          <label htmlFor="lineWidth">Line Width:</label>
          <input
            id="lineWidth"
            type="range"
            min="1"
            max="20"
            value={lineWidth}
            onChange={handleLineWidthChange}
          />
          <span>{lineWidth}px</span>
        </div>
      </div>
    </div>
  );
}

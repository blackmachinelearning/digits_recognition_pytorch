import * as onnx from 'onnxjs';

async function loadAndRunModel(inputData) {
  // Load the ONNX model
  const session = new onnx.InferenceSession();
  await session.loadModel("D:\\Projects\\MachineLearning\\PetProjects\\digits_recognition_pytorch\\digits_recognition_pytorch\\backend\\model\\mnist_model.onnx");

  // Preprocess inputData as needed and convert it to a tensor
  const inputTensor = new onnx.Tensor(new Float32Array(inputData), "float32", [1, 3, 224, 224]);

  // Run the model
  const outputMap = await session.run({ input: inputTensor });
  const outputTensor = outputMap.values().next().value;

  // Process the output tensor
  console.log(outputTensor.data);
}

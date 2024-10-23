import { type Ref, useEffect, useRef, useState } from "preact/hooks";
import * as tf from "@tensorflow/tfjs";

const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const STOP_DATA_GATHER = -1;

export const CLASS_NAMES = ["phone", "hand"];

function createModel(classNames: string[]) {
  // Initialize a new sequential model
  const model = tf.sequential();

  // Add a dense layer with specified input shape, units, and ReLU activation
  model.add(
    tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" }),
  );

  // Add a dense layer with units equal to classNames length and softmax activation
  model.add(
    tf.layers.dense({ units: classNames.length, activation: "softmax" }),
  );

  // Print the model summary
  model.summary();

  // Compile the model with appropriate loss function and metrics
  model.compile({
    optimizer: "adam",
    loss: (classNames.length === 2)
      ? "binaryCrossentropy"
      : "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function logProgress(epoch: number, logs: any) {
  console.log("Data for epoch " + epoch, logs);
}

function calculateFeaturesOnCurrentFrame(
  wcRef: Ref<HTMLVideoElement>,
  mobilenet: tf.GraphModel,
) {
  return tf.tidy(function () {
    // Grab pixels from current VIDEO frame.
    if (!wcRef.current) {
      throw new Error("Video element reference is null");
    }

    const videoFrameAsTensor = tf.browser.fromPixels(wcRef.current);
    // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
    const resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true,
    );
    const normalizedTensorFrame = resizedTensorFrame.div(255);
    const prediction = (mobilenet.predict(
      normalizedTensorFrame.expandDims(),
    ) as tf.Tensor).squeeze();

    return prediction;
  });
}

/**
 * Custom hook to manage TensorFlow operations for a webcam-based machine learning model.
 *
 * @param {string[]} classNames - Array of class names for the model.
 * @param {Ref<HTMLVideoElement>} wcRef - Reference to the HTML video element.
 *
 * @returns {Object} - An object containing various states and functions to manage the TensorFlow model.
 * @property {tf.GraphModel | undefined} featureModel - The loaded MobileNet feature model.
 * @property {tf.Sequential | undefined} model - The custom model created for training and prediction.
 * @property {number} gatherDataState - The current state of data gathering.
 * @property {Function} gatherDataForClass - Function to start or stop gathering data for a specific class.
 * @property {Function} setVideoPlaying - Function to set the video playing state.
 * @property {Function} reset - Function to reset the training data and states.
 * @property {string} statusText - The current status text for display.
 * @property {boolean} predict - Boolean indicating whether the model is in prediction mode.
 * @property {Function} trainAndPredict - Function to train the model and start prediction.
 */
export function useTensorflow(
  classNames: string[],
  wcRef: Ref<HTMLVideoElement>,
) {
  const requestRef = useRef<number>();
  const [gatherDataState, setGatherDataState] = useState(STOP_DATA_GATHER);
  const [predict, setPredict] = useState(false);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [featureModel, setFeatureModel] = useState<tf.GraphModel>();
  const [model, setModel] = useState<tf.Sequential>();
  const [trainingDataInputs, setTrainingDataInputs] = useState<
    tf.Tensor<tf.Rank>[]
  >([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState<number[]>([]);
  const [examplesCount, setExamplesCount] = useState<number[]>([]);
  const [statusText, setStatusText] = useState("");

  // Function to add training data inputs and outputs
  const addTrainingData = (imageFeatures: tf.Tensor, state: number) => {
    setTrainingDataInputs((prevInputs) => [...prevInputs, imageFeatures]);
    setTrainingDataOutputs((prevOutputs) => [...prevOutputs, state]);
  };

  // Function to update status text
  const updateStatusText = () => {
    if (examplesCount.length === 0) {
      setStatusText("No data collected");
      return;
    }

    const newStatusText = CLASS_NAMES.map(
      (name, index) => `${name} data count: ${examplesCount[index]}.`,
    ).join(" ");

    setStatusText(newStatusText);
  };

  // Function to update examples count
  const updateExamplesCount = (gatherDataState: number) => {
    setExamplesCount((prevCounts) => {
      const newCounts = [...prevCounts];
      newCounts[gatherDataState] = (newCounts[gatherDataState] || 0) + 1;
      return newCounts;
    });
  };

  const reset = () => {
    setPredict(false);
    setExamplesCount([]);

    trainingDataInputs.forEach((tensor) => tensor.dispose());

    setTrainingDataInputs([]);
    setTrainingDataOutputs([]);
    setGatherDataState(STOP_DATA_GATHER);

    console.log("Tensors in memory: " + tf.memory().numTensors);
  };

  const gatherDataForClass = (id: number) => {
    setGatherDataState(
      gatherDataState === STOP_DATA_GATHER ? id : STOP_DATA_GATHER,
    );
  };

  const dataGatherLoop = () => {
    if (!wcRef.current || !featureModel) return;

    // Ensure tensors are cleaned up.
    const imageFeatures = calculateFeaturesOnCurrentFrame(wcRef, featureModel);

    addTrainingData(imageFeatures, gatherDataState);
    updateExamplesCount(gatherDataState);

    requestRef.current = globalThis.requestAnimationFrame(dataGatherLoop);
  };

  const predictLoop = () => {
    if (!wcRef.current || !featureModel || !model) return;

    tf.tidy(function () {
      const imageFeatures = calculateFeaturesOnCurrentFrame(
        wcRef,
        featureModel,
      );
      const prediction = (model.predict(
        imageFeatures.expandDims(),
      ) as tf.Tensor).squeeze();
      const highestIndexTensor = prediction.argMax();
      const highestIndex = highestIndexTensor.arraySync() as number;
      const predictionArray = prediction.arraySync() as number[];
      const predictText = "Prediction: " + CLASS_NAMES[highestIndex] +
        " with " +
        Math.floor(predictionArray[highestIndex] * 100) + "% confidence";

      setStatusText(predictText);
    });

    requestRef.current = globalThis.requestAnimationFrame(predictLoop);
  };

  const trainAndPredict = async () => {
    if (!model) return;
    setPredict(false);

    // Shuffle the training data
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

    // Convert outputs to tensor and one-hot encode them
    const outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    const inputsAsTensor = tf.stack(trainingDataInputs);

    await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 10,
      callbacks: { onEpochEnd: logProgress },
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();

    setPredict(true);
  };

  useEffect(() => {
    const loadMobileNetFeatureModel = async () => {
      const mobilenet = await tf.loadGraphModel("/model/model.json");

      // Warm up the model by passing zeros through it once.
      tf.tidy(function () {
        const answer = mobilenet.predict(
          tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]),
        );
        console.log(answer);
      });

      setFeatureModel(mobilenet);
    };

    loadMobileNetFeatureModel();
  }, []);

  useEffect(() => {
    if (featureModel) {
      // Create and set the model once the feature model is loaded
      const newModel = createModel(classNames);
      setModel(newModel);
    }
  }, [featureModel]);

  useEffect(() => {
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER && wcRef.current) {
      dataGatherLoop();
    }
    // Cleanup function to stop the loop when conditions change
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }

      setGatherDataState((prevState) => {
        if (prevState === gatherDataState) return STOP_DATA_GATHER;
        return prevState;
      });
    };
  }, [videoPlaying, gatherDataState]);

  useEffect(() => {
    updateStatusText();
  }, [examplesCount]);

  useEffect(() => {
    if (predict) predictLoop();

    // Cleanup function to stop the loop when conditions change
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [predict]);

  return {
    featureModel,
    model,
    gatherDataState,
    gatherDataForClass,
    setVideoPlaying,
    reset,
    statusText,
    predict,
    trainAndPredict,
  };
}

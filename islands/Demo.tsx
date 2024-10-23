import { useRef } from "preact/hooks";
import { type Ref, useEffect } from "preact/hooks";
import { CLASS_NAMES, useTensorflow } from "../hooks/useTensorflow.ts";
import { Webcam } from "../components/Webcam.tsx";
import { Button } from "../components/Button.tsx";

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam(
  wc: Ref<HTMLVideoElement>,
  setVideoPlaying: (x: boolean) => void,
) {
  if (hasGetUserMedia()) {
    const constraints = {
      video: true,
      width: 640,
      height: 480,
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      if (wc.current) {
        wc.current.srcObject = stream;
        wc.current.addEventListener("loadeddata", function () {
          setVideoPlaying(true);
        });
      }
    });
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
}

export default function Demo() {
  const wc = useRef<HTMLVideoElement>(null);
  const {
    featureModel,
    gatherDataState,
    gatherDataForClass,
    setVideoPlaying,
    reset,
    statusText,
    train,
    setTrain,
    predict,
    setPredict,
  } = useTensorflow(
    CLASS_NAMES,
    wc,
  );
  const isDisabled = (id: number) =>
    gatherDataState !== id && gatherDataState !== -1 && !predict;

  useEffect(() => {
    enableCam(wc, setVideoPlaying);
  }, [wc]);

  return (
    <div class="flex gap-8 py-6">
      <div>
        <div class="flex items-center mb-1">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              featureModel ? "bg-green-500" : "bg-red-500"
            }`}
          />
          {featureModel && <span class="text-xs ml-1">ON AIR</span>}
        </div>
        <Webcam wcRef={wc} />
        <div class="space-x-4 mb-6">
          {CLASS_NAMES.map((klass: string, index: number) => (
            <Button
              onClick={() => gatherDataForClass(index)}
              disabled={isDisabled(index)}
            >
              {klass}
            </Button>
          ))}
          <Button
            onClick={() => setTrain(true)}
            disabled={train}
            class="border-green-500 hover:bg-green-200"
          >
            Start Training
          </Button>
          <Button
            disabled={train && predict}
            onClick={() => setPredict(true)}
            class="border-green-500 hover:bg-green-200"
          >
            Start Prediction
          </Button>
          <Button
            onClick={() => reset()}
            class="border-red-500 hover:bg-red-200"
          >
            Reset
          </Button>
        </div>
        <p class="font-mono text-xs">{statusText}</p>
      </div>
    </div>
  );
}

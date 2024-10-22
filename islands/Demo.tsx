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
          console.log("Video playing");
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
    trainAndPredict,
    predict
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
        <p class="mb-4">Loading{featureModel ? " complete" : "..."}</p>
        <Webcam wcRef={wc} />
        <div class="space-x-4 mb-6">
          <Button
            onClick={() => gatherDataForClass(0)}
            disabled={isDisabled(0)}
          >
            Phone
          </Button>
          <Button
            onClick={() => gatherDataForClass(1)}
            disabled={isDisabled(1)}
          >
            Hand
          </Button>
          <Button
            onClick={() => trainAndPredict()}
            class="border-green-500 hover:bg-green-200"
          >
            Train and Predict
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

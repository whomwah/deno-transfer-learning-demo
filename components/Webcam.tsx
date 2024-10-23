import { type Ref } from "preact";

interface WebcamProps {
  wcRef: Ref<HTMLVideoElement>;
}

export function Webcam(props: WebcamProps) {
  return <video ref={props.wcRef} class="mb-6" autoplay></video>;
}

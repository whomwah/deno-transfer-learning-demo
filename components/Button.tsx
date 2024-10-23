import { JSX } from "preact";
import { IS_BROWSER } from "$fresh/runtime.ts";

function mergeClassNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

export function Button(props: JSX.HTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      class={mergeClassNames(
        "px-2 py-1 border-gray-500 border-2 rounded bg-white hover:bg-gray-200 transition-colors",
        props.disabled ? "opacity-50 cursor-not-allowed" : "",
        typeof props.class === "string" ? props.class : "",
      )}
      disabled={!IS_BROWSER || props.disabled}
    />
  );
}

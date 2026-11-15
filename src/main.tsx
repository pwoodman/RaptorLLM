import React from "react";
import ReactDOM from "react-dom/client";
import "highlight.js/styles/github-dark.css";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// The splash in index.html stays up for at least six seconds so the first render is smooth, then fades out.
const SPLASH_MIN_MS = 6000;
const splash = document.getElementById("splash");
if (splash) {
  const remaining = Math.max(0, SPLASH_MIN_MS - performance.now());
  window.setTimeout(() => {
    splash.classList.add("hide");
    window.setTimeout(() => splash.remove(), 400);
  }, remaining);
}

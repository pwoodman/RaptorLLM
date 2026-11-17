import React from "react";
import ReactDOM from "react-dom/client";
import "highlight.js/styles/github-dark.css";
import App from "./App";
import "./styles.css";
import { startAppearance } from "./state/appearance";

startAppearance();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// Dismiss the splash once the application has rendered, without an artificial startup delay.
const SPLASH_MIN_MS = 0;
const splash = document.getElementById("splash");
if (splash) {
  const remaining = Math.max(0, SPLASH_MIN_MS - performance.now());
  window.setTimeout(() => {
    splash.classList.add("hide");
    window.setTimeout(() => splash.remove(), 400);
  }, remaining);
}

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Showcase from "./Showcase";
import MatrixShowcase from "./MatrixShowcase";
import "./style.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {new URLSearchParams(location.search).has("matrix") ? (
      <MatrixShowcase />
    ) : new URLSearchParams(location.search).has("showcase") ? (
      <Showcase />
    ) : (
      <App />
    )}
  </React.StrictMode>,
);

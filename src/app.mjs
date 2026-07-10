import { Application } from "pixi.js";

// Create and initialize a full-screen Pixi application (the game viewport).
// The canvas fills the window and follows its size. World->screen scaling is
// handled by the MapRenderingContext, which fits a fixed number of meters into
// the canvas height; the visible width is whatever the screen happens to be.
export async function createApp({ background = "#05080a" } = {}) {
  const app = new Application();
  await app.init({ resizeTo: window, background, antialias: true });

  Object.assign(document.body.style, { margin: "0", overflow: "hidden" });
  Object.assign(app.canvas.style, { display: "block" });
  document.body.appendChild(app.canvas);
  return app;
}

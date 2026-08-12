import { ViewerApp } from "./viewer-app.mjs";

const app = new ViewerApp();
window.addEventListener("beforeunload", () => app.dispose());
app.start();

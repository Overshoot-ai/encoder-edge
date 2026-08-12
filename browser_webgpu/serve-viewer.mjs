import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, relative as relativePath, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { modelPath } from "./model-release.mjs";

const root = fileURLToPath(new URL(".", import.meta.url));
const workspace = resolve(root, "..");
const encoderPath = modelPath(workspace);
const port = Number.parseInt(process.env.VIEWER_PORT || "3000", 10);
const runtimeRoot = join(root, "node_modules/onnxruntime-web/dist");

if (!existsSync(encoderPath)) {
  throw new Error(
    `Missing WebGPU encoder at ${encoderPath}\nRun npm run download:model, then try again.`,
  );
}
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("VIEWER_PORT must be an integer from 1 through 65535");
}

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
};

function localPath(directory, relative) {
  const path = normalize(join(directory, relative));
  const distance = relativePath(directory, path);
  if (distance.startsWith("..") || resolve(directory, distance) !== path) {
    throw new Error("Path traversal rejected");
  }
  return path;
}

const server = createServer((request, response) => {
  try {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    if (url.pathname === "/favicon.ico") {
      response.writeHead(204).end();
      return;
    }

    let path;
    if (url.pathname === "/model.onnx") path = encoderPath;
    else if (url.pathname.startsWith("/runtime/")) {
      path = localPath(runtimeRoot, decodeURIComponent(url.pathname.slice("/runtime/".length)));
    } else {
      const relative = decodeURIComponent(url.pathname.slice(1)) || "viewer.html";
      path = localPath(root, relative);
    }

    if (!existsSync(path) || statSync(path).isDirectory()) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }
    const details = statSync(path);
    response.writeHead(200, {
      "Content-Type": path === encoderPath
        ? "application/octet-stream"
        : contentTypes[extname(path)] || "application/octet-stream",
      "Content-Length": details.size,
      "Cache-Control": "no-store",
    });
    createReadStream(path).pipe(response);
  } catch (error) {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(error.stack || String(error));
  }
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use. Set VIEWER_PORT to use another port.`);
    process.exitCode = 1;
    return;
  }
  throw error;
});
server.listen(port, "127.0.0.1", () => {
  console.log(`Gemma WebGPU embedding viewer: http://localhost:${port}`);
  console.log(`Encoder: ${encoderPath}`);
});

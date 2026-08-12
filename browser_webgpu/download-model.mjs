import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync } from "node:fs";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import {
  MODEL_BYTES,
  MODEL_SHA256,
  MODEL_URL,
  modelPath,
} from "./model-release.mjs";

const root = fileURLToPath(new URL(".", import.meta.url));
const workspace = resolve(root, "..");
const destination = modelPath(workspace);
const source = process.env.VIEWER_MODEL_URL || MODEL_URL;

async function sha256(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function verify(path) {
  const details = await stat(path);
  if (details.size !== MODEL_BYTES) return false;
  return await sha256(path) === MODEL_SHA256;
}

if (existsSync(destination)) {
  process.stdout.write("Verifying cached WebGPU encoder... ");
  if (!await verify(destination)) {
    throw new Error(
      `The encoder at ${destination} does not match the pinned release. Remove it and run npm run viewer again.`,
    );
  }
  console.log("ready");
  process.exit(0);
}

await mkdir(dirname(destination), { recursive: true });
const temporary = `${destination}.download-${process.pid}`;
console.log(`Downloading the WebGPU encoder (${(MODEL_BYTES / 1_000_000).toFixed(0)} MB)...`);

try {
  const response = await fetch(source, {
    redirect: "follow",
    headers: { "User-Agent": "Overshoot-encoder-edge" },
  });
  if (!response.ok || !response.body) {
    throw new Error(`Model download failed: HTTP ${response.status} ${response.statusText}`);
  }

  let downloaded = 0;
  let reported = 0;
  const hash = createHash("sha256");
  const verifier = new Transform({
    transform(chunk, _encoding, callback) {
      downloaded += chunk.length;
      hash.update(chunk);
      const percent = Math.floor(downloaded / MODEL_BYTES * 100);
      if (percent >= reported + 10) {
        reported = percent;
        process.stdout.write(`${Math.min(percent, 100)}% `);
      }
      callback(null, chunk);
    },
  });
  await pipeline(
    Readable.fromWeb(response.body),
    verifier,
    createWriteStream(temporary, { flags: "wx" }),
  );
  process.stdout.write("\n");

  const digest = hash.digest("hex");
  if (downloaded !== MODEL_BYTES || digest !== MODEL_SHA256) {
    throw new Error(
      `Downloaded encoder failed verification: ${downloaded} bytes, sha256 ${digest}`,
    );
  }
  await rename(temporary, destination);
  console.log(`Encoder ready: ${destination}`);
} catch (error) {
  await rm(temporary, { force: true });
  throw error;
}

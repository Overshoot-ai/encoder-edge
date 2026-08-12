import { createReadStream, existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { chromium } from "playwright";

const execFileAsync = promisify(execFile);

const root = fileURLToPath(new URL(".", import.meta.url));
const workspace = resolve(root, "..");
const artifactRoot = join(workspace, "artifacts", "browser-webgpu");
const modelFile = process.env.BENCHMARK_MODEL
  || "gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu-matmulclip.onnx";
const modelPath = resolve(artifactRoot, modelFile);
const fixtureFile = process.env.BENCHMARK_FIXTURE || "fixture/fixture.json";
const fixturePath = resolve(artifactRoot, fixtureFile);
const outputPath = process.env.BENCHMARK_OUTPUT
  ? resolve(process.env.BENCHMARK_OUTPUT)
  : join(workspace, "benchmark-results", "browser-webgpu", "m4-webgpu.json");
const rounds = Number.parseInt(process.env.BENCHMARK_ROUNDS || "10", 10);
const profile = process.env.BENCHMARK_PROFILE === "1";
const verbose = process.env.BENCHMARK_VERBOSE === "1";
const outputF16Path = process.env.BENCHMARK_OUTPUT_F16
  ? resolve(process.env.BENCHMARK_OUTPUT_F16)
  : null;
const preprocessImage = process.env.BENCHMARK_PREPROCESS_IMAGE || null;
const cacheMode = process.env.BENCHMARK_CACHE_MODE || "none";
const sessionCycles = Number.parseInt(process.env.BENCHMARK_SESSION_CYCLES || "1", 10);
const repeatMessages = Number.parseInt(process.env.BENCHMARK_REPEAT_MESSAGES || "1", 10);
const modelEncoding = process.env.BENCHMARK_MODEL_ENCODING || "identity";

if (!["none", "cache-read", "cache-write", "opfs-read", "opfs-write"].includes(cacheMode)) {
  throw new Error("BENCHMARK_CACHE_MODE must be none, cache-read, cache-write, opfs-read, or opfs-write");
}
if (!Number.isInteger(sessionCycles) || sessionCycles < 1) {
  throw new Error("BENCHMARK_SESSION_CYCLES must be a positive integer");
}
if (!Number.isInteger(repeatMessages) || repeatMessages < 1) {
  throw new Error("BENCHMARK_REPEAT_MESSAGES must be a positive integer");
}
if (!["identity", "br"].includes(modelEncoding)) {
  throw new Error("BENCHMARK_MODEL_ENCODING must be identity or br");
}

if (![modelPath, fixturePath].every((path) => path.startsWith(`${artifactRoot}/`))) {
  throw new Error("Model and fixture must resolve inside artifacts/browser-webgpu");
}

async function processTreeRss(rootPid) {
  const { stdout } = await execFileAsync("ps", ["-axo", "pid=,ppid=,rss=,comm="]);
  const processes = stdout.trim().split("\n").map((line) => {
    const match = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
    return match && {
      pid: Number(match[1]),
      ppid: Number(match[2]),
      rss_bytes: Number(match[3]) * 1024,
      command: match[4],
    };
  }).filter(Boolean);
  const pids = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const process of processes) {
      if (pids.has(process.ppid) && !pids.has(process.pid)) {
        pids.add(process.pid);
        changed = true;
      }
    }
  }
  const tree = processes.filter((process) => pids.has(process.pid));
  return {
    rss_bytes: tree.reduce((sum, process) => sum + process.rss_bytes, 0),
    processes: tree,
  };
}

function summarizeOrtNodeProfile(logs) {
  const providers = {};
  const operators = {};
  let nodeEvents = 0;
  for (const { text } of logs) {
    if (!/"cat"\s*:\s*"Node"/.test(text)) continue;
    nodeEvents += 1;
    const provider = text.match(/"provider"\s*:\s*"([^"]+)"/)?.[1] || "unknown";
    const operator = text.match(/"op_name"\s*:\s*"([^"]+)"/)?.[1] || "unknown";
    providers[provider] = (providers[provider] || 0) + 1;
    operators[operator] = (operators[operator] || 0) + 1;
  }
  return { node_events: nodeEvents, providers, operators };
}

for (const path of [modelPath, fixturePath]) {
  if (!existsSync(path)) throw new Error(`Missing ${path}`);
}

const contentTypes = {
  ".f16": "application/octet-stream",
  ".html": "text/html; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
};
const mounts = [
  ["/node_modules/", join(root, "node_modules")],
  ["/artifacts/", join(workspace, "artifacts")],
  ["/", root],
];

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/favicon.ico") {
      response.writeHead(204);
      response.end();
      return;
    }
    if (url.pathname === "/cache-clear") {
      response.writeHead(204, { "Clear-Site-Data": '"cache", "storage"' });
      response.end();
      return;
    }
    const [prefix, directory] = mounts.find(([candidate]) => url.pathname.startsWith(candidate));
    const relative = decodeURIComponent(url.pathname.slice(prefix.length)) || "index.html";
    const path = normalize(join(directory, relative));
    if (!path.startsWith(directory)) throw new Error("Path traversal rejected");
    const encodedPath = modelEncoding === "br" && path === modelPath ? `${path}.br` : path;
    const details = await stat(encodedPath);
    response.writeHead(200, {
      "Content-Type": contentTypes[extname(path)] || "application/octet-stream",
      "Content-Length": details.size,
      "Cache-Control": "no-store",
      ...(encodedPath !== path ? { "Content-Encoding": "br" } : {}),
    });
    createReadStream(encodedPath).pipe(response);
  } catch (error) {
    response.writeHead(error.code === "ENOENT" ? 404 : 500);
    response.end(String(error));
  }
});

await new Promise((resolveReady) => server.listen(0, "127.0.0.1", resolveReady));
const { port } = server.address();
const logs = [];
const memorySamples = [];
let memoryTimer;
let browser;
try {
  browser = await chromium.launch({
    channel: "chromium",
    headless: true,
    args: ["--enable-unsafe-webgpu"],
  });
  const sampleMemory = async () => {
    try {
      memorySamples.push({ at_ms: performance.now(), ...await processTreeRss(process.pid) });
    } catch (error) {
      logs.push({ type: "memory-sampler", text: String(error) });
    }
  };
  await sampleMemory();
  memoryTimer = setInterval(sampleMemory, 100);
  const page = await browser.newPage();
  page.on("console", (message) => logs.push({ type: message.type(), text: message.text() }));
  page.on("pageerror", (error) => logs.push({ type: "pageerror", text: error.stack }));
  await page.goto(`http://127.0.0.1:${port}/index.html`);
  await page.evaluate(
    ({ rounds: runCount, profile: profilingEnabled, verbose: verboseLogging, modelFile: modelName, fixtureFile: fixtureName, captureOutput, preprocessImage: imageName, cacheMode: selectedCacheMode, sessionCycles: cycleCount, repeatMessages: messageCount }) => window.startBenchmark({
      modelUrl: `/artifacts/browser-webgpu/${modelName.split("/").map(encodeURIComponent).join("/")}`,
      fixtureUrl: `/artifacts/browser-webgpu/${fixtureName.split("/").map(encodeURIComponent).join("/")}`,
      imageUrl: imageName ? `/artifacts/browser-webgpu/${imageName.split("/").map(encodeURIComponent).join("/")}` : null,
      rounds: runCount,
      profile: profilingEnabled,
      verbose: verboseLogging,
      captureOutput,
      cacheMode: selectedCacheMode,
      sessionCycles: cycleCount,
      repeatMessages: messageCount,
    }),
    { rounds, profile, verbose, modelFile, fixtureFile, captureOutput: outputF16Path !== null, preprocessImage, cacheMode, sessionCycles, repeatMessages },
  );
  await page.waitForFunction(() => window.__benchmarkResult, null, { timeout: 15 * 60 * 1000 });
  const result = await page.evaluate(() => window.__benchmarkResult);
  let capturedOutput = null;
  if (outputF16Path && result.output?.f16_base64) {
    const bytes = Buffer.from(result.output.f16_base64, "base64");
    await mkdir(dirname(outputF16Path), { recursive: true });
    await writeFile(outputF16Path, bytes);
    capturedOutput = {
      path: outputF16Path,
      bytes: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
    delete result.output.f16_base64;
  }
  await page.waitForTimeout(100);
  clearInterval(memoryTimer);
  memoryTimer = undefined;
  await sampleMemory();
  const peakMemory = memorySamples.reduce(
    (peak, sample) => sample.rss_bytes > peak.rss_bytes ? sample : peak,
    memorySamples[0],
  );
  const report = {
    measured_at: new Date().toISOString(),
    platform: process.platform,
    architecture: process.arch,
    model_file: modelFile,
    fixture_file: fixtureFile,
    requested_rounds: rounds,
    profiling_enabled: profile,
    verbose_logging: verbose,
    preprocess_image: preprocessImage,
    cache_mode: cacheMode,
    session_cycles: sessionCycles,
    repeat_messages: repeatMessages,
    model_encoding: modelEncoding,
    captured_output: capturedOutput,
    ort_node_profile: profile ? summarizeOrtNodeProfile(logs) : null,
    browser_process_tree_memory: {
      metric: "resident set size; includes Chromium CPU and unified-memory processes, not a WebGPU allocation counter",
      baseline_rss_bytes: memorySamples[0].rss_bytes,
      peak_rss_bytes: peakMemory.rss_bytes,
      final_rss_bytes: memorySamples.at(-1).rss_bytes,
      peak_processes: peakMemory.processes,
      sample_count: memorySamples.length,
    },
    result,
    browser_log_count: logs.length,
    browser_logs: profile
      ? logs.filter(({ type, text }) => type !== "log" || !/"cat"\s*:\s*"Node"/.test(text)).slice(0, 100)
      : logs,
  };
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
  if (result.type !== "result") process.exitCode = 1;
} finally {
  if (memoryTimer) clearInterval(memoryTimer);
  await browser?.close();
  await new Promise((resolveClosed) => server.close(resolveClosed));
}

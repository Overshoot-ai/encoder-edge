import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const workspace = resolve(import.meta.dirname, "..");
const resultDir = resolve(workspace, "benchmark-results", "browser-webgpu", "matmul-sweep");
const rounds = process.env.BENCHMARK_ROUNDS || "5";
const model = process.env.BENCHMARK_MODEL || "gemma4-e4b-web-fp16-fused-rmsnorm-rope-fastgelu.onnx";
const dimensions = {
  linear768: "ORT_MATMUL_ROWS_LINEAR768",
  mlpExpand: "ORT_MATMUL_ROWS_MLP_EXPAND",
  mlpDown: "ORT_MATMUL_ROWS_MLP_DOWN",
  attentionScore: "ORT_MATMUL_ROWS_ATTN_SCORE",
  attentionValue: "ORT_MATMUL_ROWS_ATTN_VALUE",
};

const allRows = (value) => Object.fromEntries(
  Object.values(dimensions).map((name) => [name, String(value)]),
);
const configurations = [
  { name: "all-row4", rows: allRows(4) },
  { name: "all-row8", rows: allRows(8) },
];
for (const [label, variable] of Object.entries(dimensions)) {
  configurations.push({
    name: `${label}-row4`,
    rows: { ...allRows(8), [variable]: "4" },
  });
  configurations.push({
    name: `${label}-row16`,
    rows: { ...allRows(8), [variable]: "16" },
  });
}

const summary = [];
for (const configuration of configurations) {
  const output = resolve(resultDir, `${configuration.name}.json`);
  const run = spawnSync("npm", ["run", "benchmark"], {
    cwd: import.meta.dirname,
    encoding: "utf8",
    env: {
      ...process.env,
      ...configuration.rows,
      BENCHMARK_MODEL: model,
      BENCHMARK_ROUNDS: rounds,
      BENCHMARK_OUTPUT: output,
    },
  });
  if (run.status !== 0) {
    process.stderr.write(run.stdout);
    process.stderr.write(run.stderr);
    process.exit(run.status ?? 1);
  }
  const report = JSON.parse(await readFile(output, "utf8"));
  const result = report.result;
  summary.push({
    name: configuration.name,
    rows: configuration.rows,
    p50_ms: result.warm_inference.p50_ms,
    p90_ms: result.warm_inference.p90_ms,
    mean_ms: result.warm_inference.mean_ms,
    relative_l2: result.output.relative_l2,
    finite: result.output.finite,
  });
  process.stdout.write(`${configuration.name}: ${result.warm_inference.p50_ms.toFixed(1)} ms p50\n`);
}

const summaryPath = resolve(resultDir, "summary.json");
await writeFile(summaryPath, `${JSON.stringify({ model, rounds: Number(rounds), results: summary }, null, 2)}\n`);
process.stdout.write(`${summaryPath}\n`);

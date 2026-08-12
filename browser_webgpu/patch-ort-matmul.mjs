import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const ortPath = fileURLToPath(
  new URL("./node_modules/onnxruntime-web/dist/ort.mjs", import.meta.url),
);
const rowCount = (name, fallback) => {
  const value = Number.parseInt(process.env[name] || String(fallback), 10);
  if (![1, 2, 4, 8, 16].includes(value)) {
    throw new Error(`${name} must be one of 1, 2, 4, 8, or 16`);
  }
  return value;
};
const rows = {
  linear768: rowCount("ORT_MATMUL_ROWS_LINEAR768", 8),
  mlpExpand: rowCount("ORT_MATMUL_ROWS_MLP_EXPAND", 8),
  mlpDown: rowCount("ORT_MATMUL_ROWS_MLP_DOWN", 8),
  attentionScore: rowCount("ORT_MATMUL_ROWS_ATTN_SCORE", 8),
  attentionValue: rowCount("ORT_MATMUL_ROWS_ATTN_VALUE", 8),
  fallback: rowCount("ORT_MATMUL_ROWS_FALLBACK", 4),
};
const optimized = `const elementsPerThread = dimAOuter <= 8 ? [4, 1, 1] : [4, batchSize === 1 && dimInner === 768 && dimBOuter === 768 ? ${rows.linear768} : batchSize === 1 && dimInner === 768 && dimBOuter === 3072 ? ${rows.mlpExpand} : batchSize === 1 && dimInner === 3072 && dimBOuter === 768 ? ${rows.mlpDown} : batchSize === 12 && dimInner === 64 && dimBOuter === 2376 ? ${rows.attentionScore} : batchSize === 12 && dimInner === 2376 && dimBOuter === 64 ? ${rows.attentionValue} : ${rows.fallback}, 1];`;
let source = await readFile(ortPath, "utf8");

const pattern = /const elementsPerThread = dimAOuter <= 8 \? \[4, 1, 1\] : \[4, [^;]+, 1\];/;
if (!source.includes(optimized) && !pattern.test(source)) {
  throw new Error("Unsupported onnxruntime-web MatMul source; expected stock tile not found");
}
if (!source.includes(optimized)) {
  source = source.replace(pattern, optimized);
}

const stockMatMul = `matMul = (context) => {
      validateInputs22(context.inputs);`;
const fusedMatMul = `matMul = (context, activationAttributes = { activation: "" }) => {
      validateInputs22(context.inputs);`;
const brokenMatMul = `matMul = (context, activationAttributes = activationAttributes) => {
      validateInputs22(context.inputs);`;
if (source.includes(brokenMatMul)) {
  source = source.replace(brokenMatMul, fusedMatMul);
}
if (!source.includes(fusedMatMul)) {
  if (!source.includes(stockMatMul)) {
    throw new Error("Unsupported onnxruntime-web MatMul entry point");
  }
  const blockStart = source.indexOf(stockMatMul);
  const blockEnd = source.indexOf("\n    };", blockStart);
  const block = source.slice(blockStart, blockEnd);
  const patchedBlock = block
    .replaceAll('{ activation: "" }', "activationAttributes")
    .replace(stockMatMul, fusedMatMul);
  if (patchedBlock === block) {
    throw new Error("ONNX Runtime MatMul activation sites were not found");
  }
  source = source.slice(0, blockStart) + patchedBlock + source.slice(blockEnd);
}

const stockCreateKernel = `createKernel(kernelType, kernelId, attribute, kernelName) {
        const op = WEBGPU_OP_RESOLVE_RULES.get(kernelType);`;
const clipCreateKernel = `createKernel(kernelType, kernelId, attribute, kernelName) {
        if (kernelType === "MatMul") {
          const marker = "|clip=";
          const markerOffset = kernelName.lastIndexOf(marker);
          if (markerOffset !== -1) {
            const [clipMin, clipMax] = kernelName.slice(markerOffset + marker.length).split(",").map(Number);
            if (!Number.isFinite(clipMin) || !Number.isFinite(clipMax)) {
              throw new Error(\`Invalid fused MatMul Clip bounds in \${kernelName}\`);
            }
            attribute = { activation: "Clip", clipMin, clipMax };
          }
        }
        const op = WEBGPU_OP_RESOLVE_RULES.get(kernelType);`;
if (!source.includes('const marker = "|clip=";')) {
  if (!source.includes(stockCreateKernel)) {
    throw new Error("Unsupported onnxruntime-web kernel creation entry point");
  }
  source = source.replace(stockCreateKernel, clipCreateKernel);
}
await writeFile(ortPath, source);

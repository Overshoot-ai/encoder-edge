import {
  EMBEDDING_DIMENSIONS,
  halfToFloat,
  TOKEN_COLUMNS,
  TOKEN_COUNT,
  TOKEN_ROWS,
  VECTOR_COLUMNS,
  VECTOR_ROWS,
} from "./viewer-shared.mjs";

const HEAT_STOPS = [
  [0, [8, 5, 30]],
  [0.25, [65, 18, 93]],
  [0.5, [160, 42, 99]],
  [0.75, [238, 105, 45]],
  [1, [252, 245, 137]],
];

function heatColor(value) {
  const upper = HEAT_STOPS.findIndex(([position]) => value <= position);
  if (upper <= 0) return HEAT_STOPS[0][1];

  const [lowPosition, low] = HEAT_STOPS[upper - 1];
  const [highPosition, high] = HEAT_STOPS[upper];
  const fraction = (value - lowPosition) / (highPosition - lowPosition);
  return low.map((channel, index) => Math.round(channel + (high[index] - channel) * fraction));
}

function embeddingMean(source) {
  const mean = new Float32Array(EMBEDDING_DIMENSIONS);
  let globalSquared = 0;

  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const offset = token * EMBEDDING_DIMENSIONS;
    for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
      const value = halfToFloat(source[offset + dimension]);
      mean[dimension] += value;
      globalSquared += value * value;
    }
  }
  for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
    mean[dimension] /= TOKEN_COUNT;
  }
  return { mean, globalSquared };
}

function summarizeVectors(source, mean, globalSquared) {
  const novelty = new Float32Array(TOKEN_COUNT);
  const vectorColors = new Uint8Array(TOKEN_COUNT * EMBEDDING_DIMENSIONS);
  const vectorScale = 2 * Math.sqrt(globalSquared / (TOKEN_COUNT * EMBEDDING_DIMENSIONS));

  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const offset = token * EMBEDDING_DIMENSIONS;
    let squaredDistance = 0;
    for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
      const value = halfToFloat(source[offset + dimension]);
      const delta = value - mean[dimension];
      squaredDistance += delta * delta;
      vectorColors[offset + dimension] = Math.round(128 + Math.tanh(value / vectorScale) * 127);
    }
    novelty[token] = Math.sqrt(squaredDistance / EMBEDDING_DIMENSIONS);
  }
  return { novelty, vectorColors };
}

function renderNovelty(novelty) {
  const sorted = [...novelty].sort((a, b) => a - b);
  const low = sorted[Math.floor(TOKEN_COUNT * 0.05)];
  const high = sorted[Math.floor(TOKEN_COUNT * 0.95)];
  const pixels = new Uint8ClampedArray(TOKEN_COUNT * 4);

  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const normalized = Math.max(0, Math.min(1, (novelty[token] - low) / (high - low || 1)));
    pixels.set([...heatColor(normalized), 255], token * 4);
  }
  return pixels;
}

export function createEmbeddingPreview(source) {
  const { mean, globalSquared } = embeddingMean(source);
  const { novelty, vectorColors } = summarizeVectors(source, mean, globalSquared);
  return {
    width: TOKEN_COLUMNS,
    height: TOKEN_ROWS,
    vectorColumns: VECTOR_COLUMNS,
    vectorRows: VECTOR_ROWS,
    noveltyPixels: renderNovelty(novelty).buffer,
    vectorColors: vectorColors.buffer,
    vectorBits: new Uint16Array(source).buffer,
  };
}

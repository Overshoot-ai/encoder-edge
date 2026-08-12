import { halfToFloat } from "./fp16.mjs";
import {
  EMBEDDING_DIMENSIONS,
  TOKEN_COLUMNS,
  TOKEN_COUNT,
  TOKEN_ROWS,
} from "./viewer-config.mjs";

const VECTOR_GAP = 1;

export function decodePreview(raw) {
  return {
    ...raw,
    noveltyPixels: new Uint8ClampedArray(raw.noveltyPixels),
    vectorColors: new Uint8Array(raw.vectorColors),
    vectorBits: new Uint16Array(raw.vectorBits),
  };
}

function signedColor(value) {
  const signed = (value - 128) / 127;
  if (signed < 0) {
    const amount = -signed;
    return [9 + 28 * amount, 9 + 90 * amount, 11 + 224 * amount];
  }
  return [9 + 211 * signed, 9 + 29 * signed, 11 + 27 * signed];
}

function drawSpatial(context, canvas, preview) {
  canvas.width = preview.width;
  canvas.height = preview.height;
  const image = context.createImageData(preview.width, preview.height);
  image.data.set(preview.noveltyPixels);
  context.putImageData(image, 0, 0);
}

function drawVectors(context, canvas, preview) {
  const cellWidth = preview.vectorColumns + VECTOR_GAP;
  const cellHeight = preview.vectorRows + VECTOR_GAP;
  canvas.width = preview.width * cellWidth - VECTOR_GAP;
  canvas.height = preview.height * cellHeight - VECTOR_GAP;
  const image = context.createImageData(canvas.width, canvas.height);

  for (let token = 0; token < TOKEN_COUNT; token += 1) {
    const tokenX = token % TOKEN_COLUMNS;
    const tokenY = Math.floor(token / TOKEN_COLUMNS);
    for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
      const x = tokenX * cellWidth + dimension % preview.vectorColumns;
      const y = tokenY * cellHeight + Math.floor(dimension / preview.vectorColumns);
      const offset = (y * canvas.width + x) * 4;
      const color = signedColor(preview.vectorColors[token * EMBEDDING_DIMENSIONS + dimension]);
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

export function drawEmbedding(canvas, preview, view) {
  if (!preview) return;
  const context = canvas.getContext("2d");
  if (view === "vectors") drawVectors(context, canvas, preview);
  else drawSpatial(context, canvas, preview);
}

export function tokenAtPoint(canvas, preview, view, event) {
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canvas.height / bounds.height;

  if (view === "spatial") {
    const tokenX = Math.min(TOKEN_COLUMNS - 1, Math.floor(x));
    const tokenY = Math.min(TOKEN_ROWS - 1, Math.floor(y));
    return tokenY * TOKEN_COLUMNS + tokenX;
  }

  const cellWidth = preview.vectorColumns + VECTOR_GAP;
  const cellHeight = preview.vectorRows + VECTOR_GAP;
  const tokenX = Math.floor(x / cellWidth);
  const tokenY = Math.floor(y / cellHeight);
  const insideVector = x % cellWidth < preview.vectorColumns && y % cellHeight < preview.vectorRows;
  if (tokenX >= TOKEN_COLUMNS || tokenY >= TOKEN_ROWS || !insideVector) return null;
  return tokenY * TOKEN_COLUMNS + tokenX;
}

export function tokenValues(preview, token) {
  const offset = token * EMBEDDING_DIMENSIONS;
  return Array.from(
    preview.vectorBits.subarray(offset, offset + EMBEDDING_DIMENSIONS),
    halfToFloat,
  );
}

import assert from "node:assert/strict";
import { createEmbeddingPreview } from "./embedding-preview.mjs";
import {
  EMBEDDING_DIMENSIONS,
  floatToHalf,
  halfToFloat,
  TOKEN_COLUMNS,
  TOKEN_COUNT,
  TOKEN_ROWS,
  tokenDisplayPosition,
  VECTOR_COLUMNS,
  VECTOR_ROWS,
} from "./viewer-shared.mjs";

for (const value of [0, -0, 0.5, -2, 1000, Number.POSITIVE_INFINITY]) {
  assert.equal(halfToFloat(floatToHalf(value)), value);
}
assert.ok(Number.isNaN(halfToFloat(0x7e00)));

const bits = new Uint16Array(TOKEN_COUNT * EMBEDDING_DIMENSIONS);
for (let token = 0; token < TOKEN_COUNT; token += 1) {
  for (let dimension = 0; dimension < EMBEDDING_DIMENSIONS; dimension += 1) {
    bits[token * EMBEDDING_DIMENSIONS + dimension] = floatToHalf((token - dimension) / 64);
  }
}

const preview = createEmbeddingPreview(bits);
assert.equal(preview.width, TOKEN_COLUMNS);
assert.equal(preview.height, TOKEN_ROWS);
assert.equal(preview.vectorColumns, VECTOR_COLUMNS);
assert.equal(preview.vectorRows, VECTOR_ROWS);
assert.equal(preview.noveltyPixels.byteLength, TOKEN_COUNT * 4);
assert.equal(preview.vectorColors.byteLength, bits.length);
assert.equal(preview.vectorBits.byteLength, bits.byteLength);

const previewBits = new Uint16Array(preview.vectorBits);
assert.equal(previewBits[5 * EMBEDDING_DIMENSIONS], bits[5 * EMBEDDING_DIMENSIONS]);
assert.deepEqual(tokenDisplayPosition(0), { x: TOKEN_COLUMNS - 1, y: 0 });
assert.deepEqual(tokenDisplayPosition(TOKEN_COUNT - 1), { x: 0, y: TOKEN_ROWS - 1 });

console.log("Viewer module tests passed");

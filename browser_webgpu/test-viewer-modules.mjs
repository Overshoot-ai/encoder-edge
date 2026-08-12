import assert from "node:assert/strict";
import { createEmbeddingPreview } from "./embedding-preview.mjs";
import { tokenAtPoint, tokenValues } from "./embedding-canvas.mjs";
import { floatToHalf, halfToFloat } from "./fp16.mjs";
import {
  EMBEDDING_DIMENSIONS,
  TOKEN_COLUMNS,
  TOKEN_COUNT,
  TOKEN_ROWS,
  VECTOR_COLUMNS,
  VECTOR_ROWS,
} from "./viewer-config.mjs";

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

const canvas = {
  width: TOKEN_COLUMNS,
  height: TOKEN_ROWS,
  getBoundingClientRect: () => ({ left: 0, top: 0, width: TOKEN_COLUMNS, height: TOKEN_ROWS }),
};
assert.equal(tokenAtPoint(canvas, preview, "spatial", { clientX: 0.5, clientY: 0.5 }), 0);
assert.equal(tokenAtPoint(canvas, preview, "spatial", { clientX: 21.5, clientY: 11.5 }), TOKEN_COUNT - 1);

const values = tokenValues({ ...preview, vectorBits: bits }, 5);
assert.equal(values.length, EMBEDDING_DIMENSIONS);
assert.equal(values[0], halfToFloat(bits[5 * EMBEDDING_DIMENSIONS]));

console.log("Viewer module tests passed");

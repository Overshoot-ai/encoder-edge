export const MODEL_BYTES = 307689273;
export const MODEL_CACHE_NAME = "gemma4-e4b-webgpu-a706b29b0d586714.onnx";

export const INPUT_SHAPE = Object.freeze([1, 2376, 768]);
export const INPUT_WIDTH = 1056;
export const INPUT_HEIGHT = 576;
export const FIT_WIDTH = 854;
export const FIT_HEIGHT = 480;
export const PATCH_SIZE = 16;

export const TOKEN_COLUMNS = 22;
export const TOKEN_ROWS = 12;
export const TOKEN_COUNT = TOKEN_COLUMNS * TOKEN_ROWS;
export const EMBEDDING_DIMENSIONS = 768;
export const OUTPUT_SHAPE = Object.freeze([1, TOKEN_COUNT, EMBEDDING_DIMENSIONS]);

export const VECTOR_COLUMNS = 32;
export const VECTOR_ROWS = 24;

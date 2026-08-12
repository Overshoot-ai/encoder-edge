export const MODEL_BYTES = 307689273;
export const MODEL_CACHE_NAME = "gemma4-e4b-webgpu-a706b29b0d586714.onnx";

export const TOKEN_COLUMNS = 22;
export const TOKEN_ROWS = 12;
export const TOKEN_COUNT = TOKEN_COLUMNS * TOKEN_ROWS;
export const EMBEDDING_DIMENSIONS = 768;
export const VECTOR_COLUMNS = 32;
export const VECTOR_ROWS = 24;
export const INPUT_SHAPE = [1, 2376, EMBEDDING_DIMENSIONS];
export const OUTPUT_SHAPE = [1, TOKEN_COUNT, EMBEDDING_DIMENSIONS];

export function tokenDisplayPosition(token) {
  return {
    x: TOKEN_COLUMNS - 1 - token % TOKEN_COLUMNS,
    y: Math.floor(token / TOKEN_COLUMNS),
  };
}

const floatBuffer = new ArrayBuffer(4);
const floatValue = new Float32Array(floatBuffer);
const floatInteger = new Uint32Array(floatBuffer);

export function floatToHalf(value) {
  floatValue[0] = value;
  const bits = floatInteger[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;

  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

export function halfToFloat(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits;

  if (exponent === 0) {
    if (mantissa === 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      bits = sign | ((exponent + 112) << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  floatInteger[0] = bits >>> 0;
  return floatValue[0];
}

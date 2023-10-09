// CUDA
#ifdef __CUDACC__
  #define GLOBAL
  #define KERNEL extern "C" __global__
// OpenCL
#else
  #define GLOBAL __global
  #define KERNEL __kernel
#endif

KERNEL void add(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}
KERNEL void subtract(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = a[i] - b[i];
    }
}

KERNEL void multiply(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = a[i] * b[i];
    }
}

KERNEL void divide(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        if (b[i] != 0) { 
            result[i] = a[i] / b[i];
        } else {
            result[i] = 0; 
        }
    }
}
KERNEL void add_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] + b[i]) % MODULUS_P;
    }
}

KERNEL void subtract_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] >= b[i]) ? (a[i] - b[i]) : (MODULUS_P + a[i] - b[i]);
        result[i] %= MODULUS_P;
    }
}

KERNEL void multiply_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {
    for (uint i = 0; i < num; i++) {
        result[i] = (a[i] * b[i]) % MODULUS_P;
    }
}
/*
KERNEL void divide_field(uint num, GLOBAL uint *a, GLOBAL uint *b, GLOBAL uint *result) {

    for (uint i = 0; i < num; i++) {
        if (b[i] != 0) { 

            result[i] = a[i] / b[i];
        } else {
            result[i] = 0; 
        }
    }
}
*/
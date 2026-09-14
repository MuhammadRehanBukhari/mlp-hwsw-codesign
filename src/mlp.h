#ifndef MLP_H
#define MLP_H
#include <stdint.h>
#include <stdio.h>
#define IN_SIZE   64
#define HIDDEN    128
#define OUT_SIZE  64
void mlp_C(int8_t *in, int32_t *out);
#endif

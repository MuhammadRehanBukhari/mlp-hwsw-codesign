#include "mlp.h"
#include "activations.h"
#include "weights_w1.h"
#include "weights_w2.h"
#include "biases.h"
#include "reference_output.h"

static int8_t clamp8(int32_t x)
{
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}
static int8_t relu(int32_t x)
{
    if (x < 0) x = 0;
    return clamp8(x >> 7);
}
/*
 * MLP structure:
 * Step 1: FC1
 * Step 2: ReLU
 * Step 3: FC2
 */
void mlp_C(int8_t *in, int32_t *out)
{
    int i, j;
    int32_t acc;
    int32_t fc1[HIDDEN];
    int8_t hidden[HIDDEN];
    /*
     * Step 1: FC1
     * input -> hidden pre-activation
     */
    for (i = 0; i < HIDDEN; i++) {
        acc = b1[i];
        for (j = 0; j < IN_SIZE; j++) {
            acc += (int32_t)in[j] * (int32_t)W1[i][j];
        }
        fc1[i] = acc;
    }
    /*
     * Step 2: ReLU
     * hidden pre-activation -> hidden activation
     */
    for (i = 0; i < HIDDEN; i++) {
        hidden[i] = relu(fc1[i]);
    }
    /*
     * Step 3: FC2
     * hidden activation -> output
     */
    for (i = 0; i < OUT_SIZE; i++) {
        acc = b2[i];
        for (j = 0; j < HIDDEN; j++) {
            acc += (int32_t)hidden[j] * (int32_t)W2[i][j];
        }
        out[i] = acc;
    }
}
int main()
{
    int i;
    int32_t out_run[OUT_SIZE];
    mlp_C(input, out_run);
    for (i = 0; i < OUT_SIZE; i++) {
        if (out_run[i] != ref_out[i]) {
            printf("Mismatch at %d: got %d expected %d\n", i, out_run[i], ref_out[i]);
            printf("ERROR!\n");
            return 1;
        }
    }
    printf("Results match.\n");
    return 0;
}

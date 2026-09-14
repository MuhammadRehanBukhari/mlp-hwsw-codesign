#include "mlp.h"
#include "activations.h"
#include "weights_w1.h"
#include "weights_w2.h"
#include "biases.h"
#include "reference_output.h"
#include <xtensa/tie/Flix.h>

/*
 * MLP structure:
 * Step 1: FC1
 * Step 2: ReLU
 * Step 3: FC2
 *
 * Final (cache-based FLIX) version. Two optimizations added on top of the
 * earlier SIMD/FLIX-only design:
 *
 *   1. Sliding-window register cache: MLP_FILL_Cx loads the input/hidden
 *      vector into 8x 128-bit cache registers ONCE per layer instead of
 *      once per neuron; MLP_SEL_Cx then supplies it to MAC_SIMD from the
 *      cache with zero further memory access.
 *
 *   2. Eliminate redundant WUR_ptr_w() resets. W1[128][64] and W2[64][128]
 *      are row-major and CONTIGUOUS: after streaming a full row's worth of
 *      16-byte chunks, ptr_w has already auto-incremented to exactly the
 *      start of the NEXT row. Resetting it every iteration was therefore a
 *      no-op for every neuron except the first. Set the pointer ONCE
 *      before each layer's loop instead.
 */

void mlp_C(int8_t *in, int32_t *out)
{
    int i;
    int8_t hidden[HIDDEN];

    /* ---------------------------------------------------------------
     * Step 1: FC1  (input -> hidden pre-activation -> ReLU)
     * ------------------------------------------------------------- */
    WUR_ptr_in((unsigned)in);
    MLP_FILL_C0();
    MLP_FILL_C1();
    MLP_FILL_C2();
    MLP_FILL_C3();               /* entire input loaded into cache ONCE */

    WUR_ptr_w((unsigned)&W1[0][0]);   /* set weight pointer ONCE for the whole layer */

    for (i = 0; i < HIDDEN; i++) {
        WUR_mlp_acc(b1[i]);      /* bias still must be reloaded every neuron */

        MLP_SEL_C0(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C1(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C2(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C3(); MLP_LD_W();
        MAC_SIMD();

        hidden[i] = RELU(RUR_mlp_acc());
    }

    /* ---------------------------------------------------------------
     * Step 2: FC2  (hidden activation -> output)
     * ------------------------------------------------------------- */
    WUR_ptr_in((unsigned)hidden);
    MLP_FILL_C0(); MLP_FILL_C1(); MLP_FILL_C2(); MLP_FILL_C3();
    MLP_FILL_C4(); MLP_FILL_C5(); MLP_FILL_C6(); MLP_FILL_C7();

    WUR_ptr_w((unsigned)&W2[0][0]);   /* set weight pointer ONCE for the whole layer */

    for (i = 0; i < OUT_SIZE; i++) {
        WUR_mlp_acc(b2[i]);

        MLP_SEL_C0(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C1(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C2(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C3(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C4(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C5(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C6(); MLP_LD_W();
        MAC_SIMD();   MLP_SEL_C7(); MLP_LD_W();
        MAC_SIMD();

        out[i] = RUR_mlp_acc();   /* raw accumulator, no ReLU on the output layer */
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

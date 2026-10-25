// The RTCD entries cnn.c and ml.c call, bound to their C versions as a generic build binds them.
#ifndef AV1_RTCD_H_
#define AV1_RTCD_H_
#include <stdbool.h>

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"

struct NN_CONFIG;
typedef struct NN_CONFIG NN_CONFIG;
struct CNN_CONFIG;
typedef struct CNN_CONFIG CNN_CONFIG;
struct CNN_LAYER_CONFIG;
typedef struct CNN_LAYER_CONFIG CNN_LAYER_CONFIG;
struct CNN_THREAD_DATA;
typedef struct CNN_THREAD_DATA CNN_THREAD_DATA;
struct CNN_BRANCH_CONFIG;
typedef struct CNN_BRANCH_CONFIG CNN_BRANCH_CONFIG;
struct CNN_MULTI_OUT;
typedef struct CNN_MULTI_OUT CNN_MULTI_OUT;
enum { NONE, RELU, SOFTSIGN, SIGMOID } UENUM1BYTE(ACTIVATION);

void av1_nn_predict_c(const float *input_nodes, const NN_CONFIG *const nn_config, int reduce_prec, float *const output);
#define av1_nn_predict av1_nn_predict_c
void av1_nn_fast_softmax_16_c(const float *input_nodes, float *output);
#define av1_nn_fast_softmax_16 av1_nn_fast_softmax_16_c
bool av1_cnn_predict_c(const float **input, int in_width, int in_height, int in_stride, const CNN_CONFIG *cnn_config, const CNN_THREAD_DATA *thread_data, CNN_MULTI_OUT *output_struct);
#define av1_cnn_predict av1_cnn_predict_c
#define av1_cnn_activate av1_cnn_activate_c
#define av1_cnn_add av1_cnn_add_c
#define av1_cnn_convolve_no_maxpool_padding_valid av1_cnn_convolve_no_maxpool_padding_valid_c
#define av1_cnn_deconvolve av1_cnn_deconvolve_c
#define av1_cnn_batchnorm av1_cnn_batchnorm_c
#endif

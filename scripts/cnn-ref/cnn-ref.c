// The reference for src/enc/partition_cnn.rs: libaom's own av1_cnn_predict_c, av1_nn_predict_c
// and intra CNN model, with intra_mode_cnn_partition's features and thresholds copied from
// av1/encoder/partition_strategy.c (libaom v3.15.1). Reads records of a qindex byte and 65x65
// luma bytes on stdin; writes per record the CNN outputs, log_q and the logit of every block by
// quad_tree_idx as f32, then per block and resolution (low, mid, hd) a byte: 1 split only,
// 2 no split.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "av1/encoder/cnn.h"
#include "av1/encoder/ml.h"
#include "av1/encoder/partition_cnn_weights.h"
#include "dc_qlookup.h"

static float cnn_buffer[CNN_OUT_BUF_SIZE];
static float log_q;

static float logit_of(int bsize_idx, int quad_tree_idx) {
  const NN_CONFIG *dnn_configs[5] = {
    NULL,
    &av1_intra_mode_cnn_partition_branch_0_dnn_config,
    &av1_intra_mode_cnn_partition_branch_1_dnn_config,
    &av1_intra_mode_cnn_partition_branch_2_dnn_config,
    &av1_intra_mode_cnn_partition_branch_3_dnn_config,
  };
  const NN_CONFIG *dnn_config = dnn_configs[bsize_idx];
  float dnn_features[100];
  float logits[4] = { 0.0f };
  const float *branch_0 = cnn_buffer;
  const float *branch_1 = branch_0 + CNN_BRANCH_0_OUT_SIZE;
  const float *branch_2 = branch_1 + CNN_BRANCH_1_OUT_SIZE;
  const float *branch_3 = branch_2 + CNN_BRANCH_2_OUT_SIZE;
  if (bsize_idx == 1) {
    int f_idx = 0;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_0_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_0[ch_idx];
    }
    const int spa_stride = 2 * 2;
    for (int lin_idx = 0; lin_idx < spa_stride; lin_idx++) {
      for (int ch_idx = 0; ch_idx < CNN_BRANCH_1_OUT_CH; ch_idx++) {
        dnn_features[f_idx++] = branch_1[lin_idx + ch_idx * spa_stride];
      }
    }
    dnn_features[f_idx++] = log_q;
  } else if (bsize_idx == 2) {
    int f_idx = 0;
    for (int idx = 0; idx < CNN_BRANCH_0_OUT_CH; idx++) {
      dnn_features[f_idx++] = branch_0[idx];
    }
    const int curr_lin_idx = quad_to_linear_1[quad_tree_idx - 1];
    const int spa_stride = 2 * 2;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_1_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_1[curr_lin_idx + ch_idx * spa_stride];
    }
    dnn_features[f_idx++] = log_q;
  } else if (bsize_idx == 3) {
    int f_idx = 0;
    const int prev_quad_idx = (quad_tree_idx - 1) / 4;
    const int prev_lin_idx = quad_to_linear_1[prev_quad_idx - 1];
    const int prev_spa_stride = 2 * 2;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_1_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_1[prev_lin_idx + ch_idx * prev_spa_stride];
    }
    const int curr_lin_idx = quad_to_linear_2[quad_tree_idx - 5];
    const int spa_stride = 4 * 4;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_2_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_2[curr_lin_idx + ch_idx * spa_stride];
    }
    dnn_features[f_idx++] = log_q;
  } else {
    int f_idx = 0;
    const int prev_quad_idx = (quad_tree_idx - 1) / 4;
    const int prev_lin_idx = quad_to_linear_2[prev_quad_idx - 5];
    const int prev_spa_stride = 4 * 4;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_2_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_2[prev_lin_idx + ch_idx * prev_spa_stride];
    }
    const int curr_lin_idx = quad_to_linear_3[quad_tree_idx - 21];
    const int spa_stride = 8 * 8;
    for (int ch_idx = 0; ch_idx < CNN_BRANCH_3_OUT_CH; ch_idx++) {
      dnn_features[f_idx++] = branch_3[curr_lin_idx + ch_idx * spa_stride];
    }
    dnn_features[f_idx++] = log_q;
  }
  av1_nn_predict(dnn_features, dnn_config, 1, logits);
  return logits[0];
}

static float logits[85];

static void walk(int bsize_idx, int quad_tree_idx) {
  logits[quad_tree_idx] = logit_of(bsize_idx, quad_tree_idx);
  if (bsize_idx == 4) return;
  for (int idx = 0; idx < 4; idx++) walk(bsize_idx + 1, 4 * quad_tree_idx + idx + 1);
}

int main(void) {
  static uint8_t record[1 + 65 * 65];
  const CNN_CONFIG *cnn_config = &av1_intra_mode_cnn_partition_cnn_config;
  const CNN_THREAD_DATA thread_data = { .num_workers = 1, .workers = NULL };
  while (fread(record, 1, sizeof(record), stdin) == sizeof(record)) {
    const int num_outputs = 4;
    const int output_dims[4] = { 1, 2, 4, 8 };
    const int out_chs[4] = { CNN_BRANCH_0_OUT_CH, CNN_BRANCH_1_OUT_CH,
                             CNN_BRANCH_2_OUT_CH, CNN_BRANCH_3_OUT_CH };
    float *output_buffer[CNN_TOT_OUT_CH];
    float **cur_output_buf = output_buffer;
    float *curr_buf_ptr = cnn_buffer;
    for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
      const int num_chs = out_chs[output_idx];
      const int ch_size = output_dims[output_idx] * output_dims[output_idx];
      for (int ch = 0; ch < num_chs; ch++) {
        cur_output_buf[ch] = curr_buf_ptr;
        curr_buf_ptr += ch_size;
      }
      cur_output_buf += num_chs;
    }
    CNN_MULTI_OUT output = {
      .num_outputs = 4,
      .output_channels = out_chs,
      .output_strides = output_dims,
      .output_buffer = output_buffer,
    };
    const int dc_q = dc_qlookup_QTX[record[0]];
    log_q = log1pf((float)(dc_q * dc_q) / 256.0f);
    log_q = (log_q - av1_intra_mode_cnn_partition_mean[0]) /
            av1_intra_mode_cnn_partition_std[0];
    uint8_t *image[1] = { record + 1 };
    if (!av1_cnn_predict_img_multi_out(image, 65, 65, 65, cnn_config,
                                       &thread_data, &output)) {
      return 1;
    }
    walk(1, 0);
    fwrite(cnn_buffer, sizeof(float), CNN_OUT_BUF_SIZE, stdout);
    fwrite(&log_q, sizeof(float), 1, stdout);
    fwrite(logits, sizeof(float), 85, stdout);
    for (int q = 0; q < 85; q++) {
      const int bsize_idx = q == 0 ? 1 : q < 5 ? 2 : q < 21 ? 3 : 4;
      const float *split[3] = { av1_intra_mode_cnn_partition_split_thresh_lowres,
                                av1_intra_mode_cnn_partition_split_thresh_midres,
                                av1_intra_mode_cnn_partition_split_thresh_hdres };
      const float *no_split[3] = { av1_intra_mode_cnn_partition_no_split_thresh_lowres,
                                   av1_intra_mode_cnn_partition_no_split_thresh_midres,
                                   av1_intra_mode_cnn_partition_no_split_thresh_hdres };
      for (int r = 0; r < 3; r++) {
        const uint8_t flags = (logits[q] > split[r][bsize_idx]) |
                              (logits[q] < no_split[r][bsize_idx]) << 1;
        fwrite(&flags, 1, 1, stdout);
      }
    }
  }
  return 0;
}

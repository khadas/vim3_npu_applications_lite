#ifndef __DENSENET_CTC_PROCESS__
#define __DENSENET_CTC_PROCESS__

#include "vnn_global.h"
#include "nn_detect_common.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned char   uint8_t;
typedef unsigned int   uint32_t;

void densenet_ctc_preprocess(input_image_t imageData, vsi_nn_graph_t *g_graph, int nn_width, int nn_height, int channels, vsi_nn_tensor_t *tensor);
void densenet_ctc_postprocess(vsi_nn_graph_t *graph, char* result, int* result_len);

#ifdef __cplusplus
}
#endif

#endif

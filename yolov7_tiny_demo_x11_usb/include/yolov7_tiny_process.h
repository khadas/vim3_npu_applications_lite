#ifndef __YOLOV7_TINY_PROCESS__
#define __YOLOV7_TINY_PROCESS__

#include "vnn_global.h"
#include "nn_detect_common.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned char   uint8_t;
typedef unsigned int   uint32_t;

void yolov7_tiny_preprocess(input_image_t imageData, vsi_nn_graph_t *g_graph, int nn_width, int nn_height, int channels, vsi_nn_tensor_t *tensor);
void yolov7_tiny_postprocess(vsi_nn_graph_t *graph, pDetResult resultData);

#ifdef __cplusplus
}
#endif

#endif

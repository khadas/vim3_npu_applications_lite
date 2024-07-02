#ifndef __YOLOV3_PROCESS__
#define __YOLOV3_PROCESS__

#include "vnn_global.h"
#include "nn_detect_common.h"

#ifdef __cplusplus
extern "C"{
#endif

typedef unsigned char   uint8_t;
typedef unsigned int   uint32_t;

void yolov8n_preprocess(input_image_t imageData, uint8_t *ptr, int nn_width, int nn_height, int channels, vsi_size_t stride, vsi_nn_tensor_t *tensor);
void yolov8n_postprocess(vsi_nn_graph_t *graph, pDetResult resultData);

#ifdef __cplusplus
}
#endif


#endif

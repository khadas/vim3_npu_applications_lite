#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "densenet_ctc_process.h"

#define NN_TENSOR_MAX_DIMENSION_NUMBER 4

/*Preprocess*/
void densenet_ctc_preprocess(input_image_t imageData, uint8_t *ptr, int nn_width, int nn_height, int channels, vsi_size_t stride, vsi_nn_tensor_t *tensor)
{
    int i, j, k;
    float *src = (float *)imageData.data;

    memset(ptr, 0, stride * nn_width * nn_height * channels * sizeof(uint8_t));

    for (i = 0; i < channels; i++) {
        for (j = 0; j < nn_width; j++) {
        	for (k = 0; k < nn_height; k++) {
    			vsi_nn_Float32ToDtype(src[channels * nn_width * k + channels * j + i], &ptr[stride * (nn_width * nn_height * i + nn_width * k + j)], &tensor->attr.dtype);
    		}
    	}
    }
    return;
}

/* Postprocess */

static char *names[] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", ",", ".", "[", "]", "{", "}", "|", "~", "@", "#", "$", "%", "^", "&", "(", ")", "<", ">", "?", ":", ";", "a", "b", "c", "d", "e", "f", "g", "h", "i", "g", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};


static vx_float32 Float16ToFloat32(const vx_int16* src, float* dst, int length)
{
	vx_int32 t1;
	vx_int32 t2;
	vx_int32 t3;
	vx_float32 out;
	
	for (int i = 0; i < length; i++)
	{
		t1 = src[i] & 0x7fff;
		t2 = src[i] & 0x8000;
		t3 = src[i] & 0x7c00;
		
		t1 <<= 13;
		t2 <<= 16;
		
		t1 += 0x38000000;
		t1 = (t3 == 0 ? 0 : t1);
		t1 |= t2;
		*((uint32_t *)&out) = t1;
		dst[i] = out;
	}
	return out;
}

void densenet_ctc_postprocess(vsi_nn_graph_t *graph, char* result, int* result_len)
{
    vsi_nn_tensor_t *tensor = NULL;
    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
    float *predictions = NULL;
    int output_len = 0;
    int output_cnt = 0;
    int i, j, max_index, stride;
    int box = 35;
    int class_num = 88;
    int sz[10];
    uint8_t *tensor_data = NULL;
    float threshold = 0.25;
    float max_conf, conf;
    vsi_status status = VSI_FAILURE;
    
    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        sz[i] = 1;
        for (j = 0; j < tensor->attr.dim_num; j++) {
            sz[i] *= tensor->attr.size[j];
        }
        output_len += sz[i];
    }
    predictions = (float *)malloc(sizeof(float) * output_len);
    
    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);

        stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
        tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
        
        for (j = 0; j < sz[i]; j++)
        {
        	vsi_nn_DtypeToFloat32(&tensor_data[stride * j], &predictions[output_cnt], &tensor->attr.dtype);
        	output_cnt++;
        }
        vsi_nn_Free(tensor_data);
    }
    
    int last_index = class_num - 1;
    for (i = 0; i < box; ++i)
    {
    	max_conf = 0;
    	max_index = class_num - 1;
    	for (j = 0; j < class_num; ++j)
    	{
    		conf = predictions[i * class_num + j];
    		if (conf > threshold && conf > max_conf)
    		{
    			max_conf = conf;
    			max_index = j;
    		}
    	}
    	if (max_index != class_num - 1 && max_index != last_index)
    	{
    		result[*result_len] = *names[max_index];
    		(*result_len)++;
    	}
    	last_index = max_index;
    }
    
    if (predictions) free(predictions);
    return;
}

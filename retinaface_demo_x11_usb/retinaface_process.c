#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "retinaface_process.h"

#define NN_TENSOR_MAX_DIMENSION_NUMBER 4

/*Preprocess*/
void retinaface_preprocess(input_image_t imageData, vsi_nn_graph_t *g_graph, int nn_width, int nn_height, int channels, float* mean, float var, vsi_nn_tensor_t *tensor)
{
    int i, j, k;
    float *src = (float *)imageData.data;
    vsi_status status = VSI_FAILURE;
    int pixel_size = nn_width * nn_height;

    if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8) {
    	vsi_size_t stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    	int8_t* ptr = (int8_t*)malloc(stride * pixel_size * channels * sizeof(int8_t));
		
		float fl = powf(2., tensor->attr.dtype.fl) / var;

		#pragma omp parallel for collapse(2)
		for (k = 0; k < nn_height; k++) {
		    for (j = 0; j < nn_width; j++) {
		    	float* src_pix = &src[(k * nn_width + j) * channels];
		    	for (i = 0; i < channels; i++) {
					ptr[stride * (pixel_size * i + nn_width * k + j)] = (src_pix[i] - mean[i]) * fl;
				}
			}
		}
		status = vsi_nn_CopyDataToTensor(g_graph, tensor, ptr);
		free(ptr);
    }
    else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT16) {
    	int16_t* ptr = (int16_t*)malloc(pixel_size * channels * sizeof(int16_t));
		
		float fl = powf(2., tensor->attr.dtype.fl) / var;

		#pragma omp parallel for collapse(2)
		for (k = 0; k < nn_height; k++) {
		    for (j = 0; j < nn_width; j++) {
		    	float* src_pix = &src[(k * nn_width + j) * channels];
		    	for (i = 0; i < channels; i++) {
					ptr[pixel_size * i + nn_width * k + j] = (src_pix[i] - mean[i]) * fl;
				}
			}
		}
		status = vsi_nn_CopyDataToTensor(g_graph, tensor, ptr);
		free(ptr);
    }
    else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
    	vsi_size_t stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    	uint8_t* ptr = (uint8_t*)malloc(stride * pixel_size * channels * sizeof(uint8_t));
		
		float scale = tensor->attr.dtype.scale * var;
		int zero_point = tensor->attr.dtype.zero_point;

		#pragma omp parallel for collapse(2)
		for (k = 0; k < nn_height; k++) {
		    for (j = 0; j < nn_width; j++) {
		    	float* src_pix = &src[(k * nn_width + j) * channels];
		    	for (i = 0; i < channels; i++) {
					ptr[stride * (pixel_size * i + nn_width * k + j)] = (src_pix[i] - mean[i]) / scale + zero_point;
				}
			}
		}
		status = vsi_nn_CopyDataToTensor(g_graph, tensor, ptr);
		free(ptr);
    }
    else {
    	vsi_size_t stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    	uint8_t* ptr = (uint8_t*)malloc(stride * pixel_size * channels * sizeof(uint8_t));
    	
    	#pragma omp parallel for collapse(2)
    	for (k = 0; k < nn_height; k++) {
		    for (j = 0; j < nn_width; j++) {
		    	float* src_pix = &src[(k * nn_width + j) * channels];
		    	for (i = 0; i < channels; i++) {
					src_pix[i] = (src_pix[i] - mean[i]) / var;
					vsi_nn_Float32ToDtype(src_pix[i], &ptr[stride * (pixel_size * i + nn_width * k + j)], &tensor->attr.dtype);
				}
			}
		}
		status = vsi_nn_CopyDataToTensor(g_graph, tensor, ptr);
		free(ptr);
    }
    return;
}

/* Postprocess */

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

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(box a, box b)
{
    float area = 0;
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    area = w*h;
    return area;
}

static float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void retinaface_postprocess(vsi_nn_graph_t *graph, pDetResult resultData)
{
    vsi_nn_tensor_t *tensor = NULL;

    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);

    float min_sizes[3][2] = {{16, 32}, {64, 128}, {256, 512}};
    float variance[2] = {0.1, 0.2};
    int grid[3] = {80, 40, 20};
    int model_width = 640;
    int model_height = 640;
    
    int sz[10];
    int i, j, k, stride;
    float threshold = 0.5;
    float iou_threshold = 0.6;
    float *box_predictions = NULL;
    float *conf_predictions = NULL;
    float *point_predictions = NULL;

    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        sz[i] = 1;
        for (j = 0; j < tensor->attr.dim_num; j++) {
            sz[i] *= tensor->attr.size[j];
        }
        if (i == 0)
        {
        	box_predictions = (float *)malloc(sizeof(float) * sz[i]);
        }
        else if (i == 1)
        {
        	conf_predictions = (float *)malloc(sizeof(float) * sz[i]);
        }
        else
        {
        	point_predictions = (float *)malloc(sizeof(float) * sz[i]);
        }
    }

    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        
        if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8) {
		    int8_t *tensor_data = NULL;
		    float fl = pow(2., -tensor->attr.dtype.fl);

		    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
		    tensor_data = (int8_t *)vsi_nn_ConvertTensorToData(graph, tensor);

		    for (j = 0; j < sz[i]; j++)
		    {
		    	if (i == 0) {
		    		box_predictions[j] = tensor_data[stride * j] * fl;
		    	}
		    	else if (i == 1) {
		    		conf_predictions[j] = tensor_data[stride * j] * fl;
		    	}
		    	else {
		    		point_predictions[j] = tensor_data[stride * j] * fl;
		    	}
		    }
		    vsi_nn_Free(tensor_data);
		}
		else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT16) {
		    int16_t *tensor_data = NULL;
		    float fl = pow(2., -tensor->attr.dtype.fl);

		    tensor_data = (int16_t *)vsi_nn_ConvertTensorToData(graph, tensor);

		    for (j = 0; j < sz[i]; j++)
		    {
		    	if (i == 0) {
		    		box_predictions[j] = tensor_data[j] * fl;
		    	}
		    	else if (i == 1) {
		    		conf_predictions[j] = tensor_data[j] * fl;
		    	}
		    	else {
		    		point_predictions[j] = tensor_data[j] * fl;
		    	}
		    }
		    vsi_nn_Free(tensor_data);
		}
		else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16) {
			uint8_t *tensor_data = NULL;
			
			tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
			
			vx_int16 *data_ptr_32 = (vx_int16 *)tensor_data;
			
			if (i == 0) {
	    		Float16ToFloat32(data_ptr_32, box_predictions, sz[i]);
	    	}
	    	else if (i == 1) {
	    		Float16ToFloat32(data_ptr_32, conf_predictions, sz[i]);
	    	}
	    	else {
	    		Float16ToFloat32(data_ptr_32, point_predictions, sz[i]);
	    	}
			vsi_nn_Free(tensor_data);
		}
		else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
		    uint8_t *tensor_data = NULL;
		    float scale = tensor->attr.dtype.scale;
			int zero_point = tensor->attr.dtype.zero_point;

		    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
		    tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);

		    for (j = 0; j < sz[i]; j++)
		    {
		    	if (i == 0) {
		    		box_predictions[j] = (tensor_data[stride * j] - zero_point) * scale;
		    	}
		    	else if (i == 1) {
		    		conf_predictions[j] = (tensor_data[stride * j] - zero_point) * scale;
		    	}
		    	else {
		    		point_predictions[j] = (tensor_data[stride * j] - zero_point) * scale;
		    	}
		    }
		    vsi_nn_Free(tensor_data);
		}
		else {
			uint8_t *tensor_data = NULL;
			
			stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
			tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
			
			for (j = 0; j < sz[i]; j++) 
		    {
				if (i == 0) {
		        	vsi_nn_DtypeToFloat32(&tensor_data[stride * j], &box_predictions[j], &tensor->attr.dtype);
		    	}
				else if (i == 1) {
				    vsi_nn_DtypeToFloat32(&tensor_data[stride * j], &conf_predictions[j], &tensor->attr.dtype);
				}
				else {
					vsi_nn_DtypeToFloat32(&tensor_data[stride * j], &point_predictions[j], &tensor->attr.dtype);
				}
			}
			vsi_nn_Free(tensor_data);
		}
    }

    box *boxes = (box *)calloc(sz[0], sizeof(box));
    float *probs = (float *)calloc(sz[1], sizeof(float));
    float **points = (float **)calloc(sz[2] / 10, sizeof(float *));
    
    for (j = 0; j < sz[0] / 10; ++j)
    {
        points[j] = (float *)calloc(10, sizeof(float *));
    }
    
    int initial = 0;
    int result_len = 0;
    int index;
    float conf;
    
    for (int n = 0; n < 3; ++n)
    {
    	if (n == 1)
    	{
    		initial = pow(grid[0], 2) * 2;
    	}
    	else if (n == 2)
    	{
    		initial = (pow(grid[0], 2) + pow(grid[1], 2)) * 2;
    	}
    	for (i = 0; i < grid[n]; ++i)
    	{
    		for (j = 0; j < grid[n]; ++j)
    		{
    			for (k = 0; k < 2; ++k)
    			{
    				index = initial + (i * grid[n] + j) * 2 + k;
    				conf = conf_predictions[index * 2 + 1];
    				if (conf >= threshold)
    				{
    					boxes[result_len].x = (j + 0.5) / grid[n] + box_predictions[index * 4 + 0] * variance[0] * min_sizes[n][k] / model_width;
    					boxes[result_len].y = (i + 0.5) / grid[n] + box_predictions[index * 4 + 1] * variance[0] * min_sizes[n][k] / model_height;
    					boxes[result_len].w = min_sizes[n][k] / model_width * exp(box_predictions[index * 4 + 2] * variance[1]);
    					boxes[result_len].h = min_sizes[n][k] / model_height * exp(box_predictions[index * 4 + 3] * variance[1]);
    					
    					points[result_len][0] = (j + 0.5) / grid[n] + point_predictions[index * 10 + 0] * variance[0] * min_sizes[n][k] / model_width;
    					points[result_len][1] = (i + 0.5) / grid[n] + point_predictions[index * 10 + 1] * variance[0] * min_sizes[n][k] / model_height;
    					points[result_len][2] = (j + 0.5) / grid[n] + point_predictions[index * 10 + 2] * variance[0] * min_sizes[n][k] / model_width;
    					points[result_len][3] = (i + 0.5) / grid[n] + point_predictions[index * 10 + 3] * variance[0] * min_sizes[n][k] / model_height;
    					points[result_len][4] = (j + 0.5) / grid[n] + point_predictions[index * 10 + 4] * variance[0] * min_sizes[n][k] / model_width;
    					points[result_len][5] = (i + 0.5) / grid[n] + point_predictions[index * 10 + 5] * variance[0] * min_sizes[n][k] / model_height;
    					points[result_len][6] = (j + 0.5) / grid[n] + point_predictions[index * 10 + 6] * variance[0] * min_sizes[n][k] / model_width;
    					points[result_len][7] = (i + 0.5) / grid[n] + point_predictions[index * 10 + 7] * variance[0] * min_sizes[n][k] / model_height;
    					points[result_len][8] = (j + 0.5) / grid[n] + point_predictions[index * 10 + 8] * variance[0] * min_sizes[n][k] / model_width;
    					points[result_len][9] = (i + 0.5) / grid[n] + point_predictions[index * 10 + 9] * variance[0] * min_sizes[n][k] / model_height;
    					
    					boxes[result_len].prob_obj = conf;
    					
    					result_len++;
    				}
    			}
    		}
    	}
    }
    
    index = 0;
    for (i = 0; i < result_len; ++i)
    {
    	if (boxes[i].prob_obj != -1)
    	{
    		for (j = i + 1; j < result_len; j++)
    		{
    			if (boxes[j].prob_obj != -1)
    			{
    				float iou = box_iou(boxes[i], boxes[j]);
    				
    				if (iou > iou_threshold)
    				{
    					if (boxes[i].prob_obj >= boxes[j].prob_obj)
    					{
    						boxes[j].prob_obj = -1;
    					}
    					else
    					{
    						boxes[i].prob_obj = -1;
    						break;
    					}
    				}
    			}
    		}
    	}
    	if (boxes[i].prob_obj != -1)
    	{
    		resultData->point[index].type = DET_RECTANGLE_TYPE;
    		resultData->point[index].point.rectPoint.left = boxes[i].x - boxes[i].w / 2;
            	resultData->point[index].point.rectPoint.top = boxes[i].y - boxes[i].h / 2;
            	resultData->point[index].point.rectPoint.right = boxes[i].x + boxes[i].w / 2;
            	resultData->point[index].point.rectPoint.bottom = boxes[i].y + boxes[i].h / 2;
           	            	
            	resultData->point[index].tpts.floatX[0] = points[i][0];
            	resultData->point[index].tpts.floatY[0] = points[i][1];
            	resultData->point[index].tpts.floatX[1] = points[i][2];
            	resultData->point[index].tpts.floatY[1] = points[i][3];
            	resultData->point[index].tpts.floatX[2] = points[i][4];
            	resultData->point[index].tpts.floatY[2] = points[i][5];
            	resultData->point[index].tpts.floatX[3] = points[i][6];
            	resultData->point[index].tpts.floatY[3] = points[i][7];
            	resultData->point[index].tpts.floatX[4] = points[i][8];
            	resultData->point[index].tpts.floatY[4] = points[i][9];
            	
            	sprintf(resultData->result_name[index].lable_name, "face: %.0f%% ", boxes[i].prob_obj*100);
            	
            	index++;
    	}
    }
    resultData->detect_num = index;

    free(boxes);
    boxes = NULL;
    free(probs);
    probs = NULL;

    if (point_predictions) free(point_predictions);
    if (conf_predictions) free(conf_predictions);
    if (box_predictions) free(box_predictions);

    for (j = 0; j < sz[2] / 10; ++j) {
        free(points[j]);
    }
    free(points);
    return;
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "yolov8n_process.h"

#define NN_TENSOR_MAX_DIMENSION_NUMBER 4

/*Preprocess*/
void yolov8n_preprocess(input_image_t imageData, vsi_nn_graph_t *g_graph, int nn_width, int nn_height, int channels, float* mean, float var, vsi_nn_tensor_t *tensor)
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

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
static char *coco_names[] = {"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};


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

static int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.classId] - b.probs[b.index][b.classId];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

static void do_nms_sort(box *boxes, float **probs, int total_in, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *)calloc(total_in, sizeof(sortable_bbox));

    int total = 0;
    for (i = 0; i < total_in; ++i) {
        if (boxes[i].prob_obj>0) {
            s[total].index = i;
            s[total].classId = 0;
            s[total].probs = probs;
            total++;
        }
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].classId = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);

        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0)
                continue;
            for (j = i+1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (probs[s[j].index][k]>0) {
                    if (box_iou(boxes[s[i].index], b) > thresh) {
                        probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    }
    free(s);
}

static int max_index(float *a, int n)
{
    int i, max_i = 0;
    float max = a[0];

    if (n <= 0)
        return -1;

    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    float r = 0;
    ratio -= i;
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

static void get_detections_result(pDetResult resultData, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
    int i,detect_num = 0;
    float left = 0, right = 0, top = 0, bot=0;

    memset(resultData, 0, sizeof(*resultData));
    for (i = 0; i < num; ++i) {
        if (boxes[i].prob_obj <= thresh) continue;
        int classId = max_index(probs[i], classes);
        float prob = probs[i][classId];
        if (prob > thresh) {
            left  = boxes[i].x-boxes[i].w/2.;
            right = boxes[i].x+boxes[i].w/2.;
            top   = boxes[i].y-boxes[i].h/2.;
            bot   = boxes[i].y+boxes[i].h/2.;

            if (left < 0) left = 0;
            if (right > 1) right = 1.0;
            if (top < 0) top = 0;
            if (bot > 1) bot = 1.0;

            if (detect_num >= MAX_DETECT_NUM) {
                break;
            }

            resultData->point[detect_num].type = DET_RECTANGLE_TYPE;
            resultData->point[detect_num].point.rectPoint.left = left;
            resultData->point[detect_num].point.rectPoint.top = top;
            resultData->point[detect_num].point.rectPoint.right = right;
            resultData->point[detect_num].point.rectPoint.bottom = bot;
            resultData->result_name[detect_num].lable_id = classId;
            sprintf(resultData->result_name[detect_num].lable_name, "%s: %.0f%% ", names[classId], prob*100);

            detect_num ++;
        }
    }
    resultData->detect_num= detect_num;
}

static float logistic_activate(float x){return 1./(1. + expf(-x));}

static float unsigmoid(float x){return -1. * logf((1. / x)) - 1.;}

static box get_region_box(float *x, int i, int j, int w, int h)
{
    box b;
    float tmp[4] = {0};
    for (int k = 0; k < 4; k++)
    {
    	float sum = 0;
    	for (int m = 0; m < 16; m++)
    	{
    		x[k * 16 + m] = exp(x[k * 16 + m]);
    		sum += x[k * 16 + m];
    	}
    	for (int m = 0; m < 16; m++)
    	{
    		tmp[k] += m * x[k * 16 + m] / sum;
    	}
    }
    b.x = (j + 0.5 - tmp[0]) / w;
    b.y = (i + 0.5 - tmp[1]) / h;
    b.w = (j + 0.5 + tmp[2]) / w;
    b.h = (i + 0.5 + tmp[3]) / h;
    b.w = b.w - b.x;
    b.h = b.h - b.y;
    b.x = b.x + b.w / 2;
    b.y = b.y + b.h / 2;
    return b;
}

static void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float*)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for (b = 0; b < batch; ++b) {
        for (c = 0; c < layers; ++c) {
            for (i = 0; i < size; ++i) {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

float dequantize(void* data, int vx_type, float fl, float scale, int zero_point)
{
	if (vx_type == VSI_NN_TYPE_INT8) {
	    int8_t ptr = *(int8_t*)data;
	    return ptr * fl;
	}
	else if (vx_type == VSI_NN_TYPE_INT16) {
	    int16_t ptr = *(int16_t*)data;
	    return ptr * fl;
	}
	else if (vx_type == VSI_NN_TYPE_UINT8) {
	    uint8_t ptr = *(uint8_t*)data;
	    return (ptr - zero_point) * scale;
	}
	else {
		uint8_t ptr = *(uint8_t*)data;
		
	    float result;
	    vsi_nn_DtypeToFloat32(&result, &ptr, vx_type);
	    return result;
	}
}

int yolo_v3_post_process_onescale(void *tensor_data, int input_size[3] , box *boxes, float **probs, float threshold_in, vsi_nn_tensor_t *tensor)
{
    int i,j,k,m,index;
    int num_class = 80;
    int coords = 64;
    int bb_size = coords + num_class;
    int modelWidth = input_size[0];
    int modelHeight = input_size[1];
    
    float fl = 1;
    float scale = 1;
    int zero_point = 0;
    
    int tmp = 0;
    
    float threshold = unsigmoid(threshold_in);
    
    for (j = 0; j < modelWidth*modelHeight; ++j)
        probs[j] = (float *)calloc(num_class, sizeof(float *));

    if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT8) {
    	fl = powf(2., -tensor->attr.dtype.fl);
    	float converted_predictions[coords];
    	int8_t *tmp_data = (int8_t *)tensor_data;
    	
    	int8_t thre_tmp = (int8_t)(threshold / fl);
    	int8_t max_prob;
    	int max_id = -1;
    	
    	#pragma omp parallel for collapse(2)
		for (i = 0; i < modelHeight; ++i)
		{
			for (j = 0; j < modelWidth; ++j)
			{
				index = i * modelHeight + j;
				max_prob = 0;
				for (k = 0; k < num_class; ++k)
				{
					if (tmp_data[index * bb_size + k] > thre_tmp && tmp_data[index * bb_size + k] > max_prob) {
						max_prob = tmp_data[index * bb_size + k];
						max_id = k;
					}
				}
				if (max_id >= 0) {
					probs[index][max_id] = logistic_activate(dequantize(&tmp_data[index * bb_size + max_id], tensor->attr.dtype.vx_type, fl, scale, zero_point));
					int box_index = index * bb_size + num_class;
					for (m = 0; m < coords; ++m) {
						converted_predictions[m] = dequantize(&tmp_data[box_index + m], tensor->attr.dtype.vx_type, fl, scale, zero_point);
					}
					boxes[index] = get_region_box(converted_predictions, i, j, modelWidth, modelHeight);
					boxes[index].prob_obj = max_prob;
				}
			}
		}
    }
    else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_INT16) {
    	fl = powf(2., -tensor->attr.dtype.fl);
    	float converted_predictions[coords];
    	int16_t *tmp_data = (int16_t *)tensor_data;
    	
    	int16_t thre_tmp = (int16_t)(threshold / fl);
    	int16_t max_prob;
    	int max_id = -1;
    	
    	#pragma omp parallel for collapse(2)
		for (i = 0; i < modelHeight; ++i)
		{
			for (j = 0; j < modelWidth; ++j)
			{
				index = i * modelHeight + j;
				max_prob = 0;
				for (k = 0; k < num_class; ++k)
				{
					if (tmp_data[index * bb_size + k] > thre_tmp && tmp_data[index * bb_size + k] > max_prob) {
						max_prob = tmp_data[index * bb_size + k];
						max_id = k;
					}
				}
				if (max_id >= 0) {
					probs[index][max_id] = logistic_activate(dequantize(&tmp_data[index * bb_size + max_id], tensor->attr.dtype.vx_type, fl, scale, zero_point));
					int box_index = index * bb_size + num_class;
					for (m = 0; m < coords; ++m) {
						converted_predictions[m] = dequantize(&tmp_data[box_index + m], tensor->attr.dtype.vx_type, fl, scale, zero_point);
					}
					boxes[index] = get_region_box(converted_predictions, i, j, modelWidth, modelHeight);
					boxes[index].prob_obj = max_prob;
				}
			}
		}
    }
    else if (tensor->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
    	scale = tensor->attr.dtype.scale;
		zero_point = tensor->attr.dtype.zero_point;
		float converted_predictions[coords];
    	uint8_t *tmp_data = (uint8_t *)tensor_data;
    	
    	uint8_t thre_tmp = (uint8_t)(threshold / zero_point + zero_point);
    	uint8_t max_prob;
    	int max_id = -1;
    	#pragma omp parallel for collapse(2)
		for (i = 0; i < modelHeight; ++i)
		{
			for (j = 0; j < modelWidth; ++j)
			{
				index = i * modelHeight + j;
				max_prob = 0;
				for (k = 0; k < num_class; ++k)
				{
					if (tmp_data[index * bb_size + k] > thre_tmp && tmp_data[index * bb_size + k] > max_prob) {
						max_prob = tmp_data[index * bb_size + k];
						max_id = k;
					}
				}
				if (max_id >= 0) {
					probs[index][max_id] = logistic_activate(dequantize(&tmp_data[index * bb_size + max_id], tensor->attr.dtype.vx_type, fl, scale, zero_point));
					int box_index = index * bb_size + num_class;
					for (m = 0; m < coords; ++m) {
						converted_predictions[m] = dequantize(&tmp_data[box_index + m], tensor->attr.dtype.vx_type, fl, scale, zero_point);
					}
					boxes[index] = get_region_box(converted_predictions, i, j, modelWidth, modelHeight);
					boxes[index].prob_obj = max_prob;
				}
			}
		}
    }
    else {
    	scale = tensor->attr.dtype.scale;
		zero_point = tensor->attr.dtype.zero_point;
		float converted_predictions[coords];
    	uint8_t *tmp_data = (uint8_t *)tensor_data;
    	
    	uint8_t thre_tmp;
    	vsi_nn_Float32ToDtype(threshold, &thre_tmp, &tensor->attr.dtype);
    	uint8_t max_prob;
    	int max_id = -1;
    	#pragma omp parallel for collapse(2)
		for (i = 0; i < modelHeight; ++i)
		{
			for (j = 0; j < modelWidth; ++j)
			{
				index = i * modelHeight + j;
				max_prob = 0;
				for (k = 0; k < num_class; ++k)
				{
					if (tmp_data[index * bb_size + k] > thre_tmp && tmp_data[index * bb_size + k] > max_prob) {
						max_prob = tmp_data[index * bb_size + k];
						max_id = k;
					}
				}
				if (max_id >= 0) {
					probs[index][max_id] = logistic_activate(dequantize(&tmp_data[index * bb_size + max_id], tensor->attr.dtype.vx_type, fl, scale, zero_point));
					int box_index = index * bb_size + num_class;
					for (m = 0; m < coords; ++m) {
						converted_predictions[m] = dequantize(&tmp_data[box_index + m], tensor->attr.dtype.vx_type, fl, scale, zero_point);
					}
					boxes[index] = get_region_box(converted_predictions, i, j, modelWidth, modelHeight);
					boxes[index].prob_obj = max_prob;
				}
			}
		}
    }
    return 0;
}

void yolov8n_postprocess(vsi_nn_graph_t *graph, pDetResult resultData)
{
    int nn_width,nn_height, nn_channel;
    vsi_nn_tensor_t *tensor = NULL;

    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
    nn_width = tensor->attr.size[0];
    nn_height = tensor->attr.size[1];
    nn_channel = tensor->attr.size[2];
    (void)nn_channel;

    int i, j, stride;
    int num_class = 80;
    float threshold = 0.3;
    float iou_threshold = 0.4;
    
    int size[3]={nn_width/32, nn_height/32, num_class + 64};

    int size2[3] = {size[0]*2,size[1]*2,size[2]};
    int size4[3] = {size[0]*4,size[1]*4,size[2]};
    int len1 = size[0]*size[1]*size[2];
    int box1 = len1/(num_class+64);

    box *boxes = (box *)calloc(box1*(1+4+16), sizeof(box));
    float **probs = (float **)calloc(box1*(1+4+16), sizeof(float *));
    
    for (i = 0; i < graph->output.num; i++) {
    	tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
    	int vx_type = tensor->attr.dtype.vx_type;
    	if (vx_type == VSI_NN_TYPE_INT8) {
    		int8_t *tensor_data = NULL;
    		tensor_data = (int8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    		if (i == 0) {
    			yolo_v3_post_process_onescale(tensor_data, size4, boxes, &probs[0], threshold, tensor);
    		}
    		else if (i == 1) {
    			yolo_v3_post_process_onescale(tensor_data, size2, &boxes[box1*16], &probs[box1*16], threshold, tensor);
    		}
    		else if (i == 2) {
    			yolo_v3_post_process_onescale(tensor_data, size, &boxes[box1*(16+4)], &probs[box1*(16+4)], threshold, tensor);
    		}
    		vsi_nn_Free(tensor_data);
    	}
    	else if (vx_type == VSI_NN_TYPE_INT16) {
    		int16_t *tensor_data = NULL;
    		tensor_data = (int16_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    		if (i == 0) {
    			yolo_v3_post_process_onescale(tensor_data, size4, boxes, &probs[0], threshold, tensor);
    		}
    		else if (i == 1) {
    			yolo_v3_post_process_onescale(tensor_data, size2, &boxes[box1*16], &probs[box1*16], threshold, tensor);
    		}
    		else if (i == 2) {
    			yolo_v3_post_process_onescale(tensor_data, size, &boxes[box1*(16+4)], &probs[box1*(16+4)], threshold, tensor);
    		}
    		vsi_nn_Free(tensor_data);
    	}
    	else {
    		uint8_t *tensor_data = NULL;
    		tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    		if (i == 0) {
    			yolo_v3_post_process_onescale(tensor_data, size4, boxes, &probs[0], threshold, tensor);
    		}
    		else if (i == 1) {
    			yolo_v3_post_process_onescale(tensor_data, size2, &boxes[box1*16], &probs[box1*16], threshold, tensor);
    		}
    		else if (i == 2) {
    			yolo_v3_post_process_onescale(tensor_data, size, &boxes[box1*(16+4)], &probs[box1*(16+4)], threshold, tensor);
    		}
    		vsi_nn_Free(tensor_data);
    	}
    }
    do_nms_sort(boxes, probs, box1*21, num_class, iou_threshold);
    get_detections_result(resultData, box1*21, threshold, boxes, probs, coco_names, num_class);

    free(boxes);
    boxes = NULL;
    
    for (j = 0; j < box1*(1+4+16); ++j) {
        free(probs[j]);
    }
    free(probs);
    return;
}

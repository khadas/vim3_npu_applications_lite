#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <semaphore.h>
#include <sys/time.h>
#include <sched.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <semaphore.h>
#include <getopt.h>
#include <queue>
#include <thread>
#include <mutex>

#include "nn_detect_utils.h"
#include "yolov8n_process.h"
#include "vnn_yolov8n.h"
#include "camera_util.h"

using namespace std;
using namespace cv;

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640
#define DEFAULT_DEVICE "/dev/video0"
#define MESON_BUFFER_SIZE 4
#define DEFAULT_OUTPUT "default.h264"
#define ION_DEVICE_NODE "/dev/ion"
#define FB_DEVICE_NODE "/dev/fb0"


struct option longopts[] = {
	{ "device",             required_argument,  NULL,   'd' },
	{ "width",              required_argument,  NULL,   'w' },
	{ "height",             required_argument,  NULL,   'h' },
	{ "model",              required_argument,  NULL,   'm' },
	{ "type",               required_argument,  NULL,   't' },
	{ "multithreading_num", required_argument,  NULL,   'n' },
	{ "help",               no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

const char *device = DEFAULT_DEVICE;
const char *model_path;
std::string type;
int thread_num;

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

const static vsi_nn_postprocess_map_element_t* postprocess_map = NULL;
const static vsi_nn_preprocess_map_element_t* preprocess_map = NULL;

int width = MAX_WIDTH;
int height = MAX_HEIGHT;

#define DEFAULT_FRAME_RATE  30

struct  Frame
{   
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;

int g_nn_height, g_nn_width, g_nn_channel;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
		cout << "_CHECK_STATUS_ File" << __FUNCTION__ << __LINE__ <<endl; \
	}\
	goto lbl; \
}while(0)


int minmax(int min, int v, int max)
{
	return (v < min) ? min : (max < v) ? max : v;
}

uint8_t* yuyv2rgb(uint8_t* yuyv, uint32_t width, uint32_t height)
{
  	uint8_t* rgb = (uint8_t *)calloc(width * height * 3, sizeof (uint8_t));
  	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j += 2) {
	  		size_t index = i * width + j;
	  		int y0 = yuyv[index * 2 + 0] << 8;
	  		int u = yuyv[index * 2 + 1] - 128;
	  		int y1 = yuyv[index * 2 + 2] << 8;
	  		int v = yuyv[index * 2 + 3] - 128;
	  		rgb[index * 3 + 0] = minmax(0, (y0 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 1] = minmax(0, (y0 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 2] = minmax(0, (y0 + 454 * u) >> 8, 255);
	  		rgb[index * 3 + 3] = minmax(0, (y1 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 4] = minmax(0, (y1 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 5] = minmax(0, (y1 + 454 * u) >> 8, 255);
		}
  	}
  	return rgb;
}

const vsi_nn_preprocess_map_element_t * vnn_GetPrePorcessMap()
{
	return preprocess_map;
}

uint32_t vnn_GetPrePorcessMapCount()
{
	if (preprocess_map == NULL)
		return 0;
	else
		return sizeof(preprocess_map) / sizeof(vsi_nn_preprocess_map_element_t);
}

const vsi_nn_postprocess_map_element_t * vnn_GetPostPorcessMap()
{
	return postprocess_map;
}

uint32_t vnn_GetPostPorcessMapCount()
{
	if (postprocess_map == NULL)
		return 0;
	else
		return sizeof(postprocess_map) / sizeof(vsi_nn_postprocess_map_element_t);
}

std::mutex input_mtx, output_mtx;
std::queue<cv::Mat> input_queue;
std::queue<cv::Mat> output_queue;
int max_input_queue_size = 5;
int input_queue_num = 0;
int output_queue_num = 0;

void read_image(std::string camera_type, std::string device_number)
{
	int ret;
	
	if (camera_type == "usb") {
		ret = load_usb_camera(device_number, width, height);
	}
	else if (camera_type == "mipi") {
		ret = load_mipi_camera(device_number, width, height);
	}
	
	while (true) {
		cv::Mat image;
		read_usb_frame(&image);
		
		input_mtx.lock();
		if (input_queue.size() < max_input_queue_size) {
			input_queue.push(image.clone());
		}
		else {
			input_queue.pop();
			input_queue.push(image.clone());
		}
		input_mtx.unlock();
	}
}

void show_image()
{
	struct timeval time_start, time_end;
	gettimeofday(&time_start, NULL);
	int n = 0;
	float total_time = 0;
	while (true) {
		output_mtx.lock();
		if (output_queue.empty()) {
			output_mtx.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		cv::Mat image = output_queue.front();
		output_queue.pop();
		cv::imshow("Image Window", image);
		cv::waitKey(1);
		gettimeofday(&time_end, NULL);
		total_time += (float)((time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000.0f / 1000.0f);
		n++;
		if (n == 10)
		{
			printf("average time : %f ms\n", (total_time / 10));
			total_time = 0;
			n = 0;
		}
		gettimeofday(&time_start, NULL);
		output_mtx.unlock();
	}
}

static cv::Scalar obj_id_to_color(int obj_id) {

	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}


static void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height){

	int i = 0;
	float left, right, top, bottom;

	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        	right = resultData.point[i].point.rectPoint.right*img_width;
        	top = resultData.point[i].point.rectPoint.top*img_height;
        	bottom = resultData.point[i].point.rectPoint.bottom*img_height;
		
//		cout << "i:" <<resultData.detect_num <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom <<endl;

		cv::Rect rect(left, top, right-left, bottom-top);
		cv::rectangle(frame,rect,obj_id_to_color(resultData.result_name[i].lable_id),1,8,0);
		int baseline;
		cv::Size text_size = cv::getTextSize(resultData.result_name[i].lable_name, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
		cv::Rect rect1(left, top-20, text_size.width+10, 20);
		cv::rectangle(frame,rect1,obj_id_to_color(resultData.result_name[i].lable_id),-1);
		cv::putText(frame,resultData.result_name[i].lable_name,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
	}
}

int run_detect_model(int thread_id, int total_thread_num){
	int nn_height, nn_width, nn_channel;

	//prepare model
	vsi_nn_graph_t * g_graph = NULL;
	g_graph = vnn_CreateYolov8n(model_path, NULL,
			vnn_GetPrePorcessMap(), vnn_GetPrePorcessMapCount(),
			vnn_GetPostPorcessMap(), vnn_GetPostPorcessMapCount());
	//cout << "det_set_model success!!" << endl;

	vsi_nn_tensor_t *tensor = NULL;
	tensor = vsi_nn_GetTensor(g_graph, g_graph->input.tensors[0]);

	nn_width = tensor->attr.size[0];
	nn_height = tensor->attr.size[1];
	nn_channel = tensor->attr.size[2];

	/*cout << "\nmodel.width:" << nn_width <<endl;
	cout << "model.height:" << nn_height <<endl;
	cout << "model.channel:" << nn_channel << "\n" <<endl;*/

	g_nn_width = nn_width;
	g_nn_height = nn_height;
	g_nn_channel = nn_channel;

	DetectResult resultData;
	cv::Mat tmp_image(g_nn_width, g_nn_height, CV_8UC3);
	cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));
	int frames = 0;
	float total_time = 0;
	vsi_status status = VSI_FAILURE;
	
	int tmp;

	while (true) {
		input_mtx.lock();
		if (input_queue.empty()) {
			input_mtx.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		img = input_queue.front();
		input_queue.pop();
		tmp = input_queue_num;
		if (input_queue_num == thread_id - 1) {
			input_queue_num = 0;
		}
		else {
			input_queue_num++;
		}
		input_mtx.unlock();
		
		cv::resize(img, tmp_image, tmp_image.size());
		cv::cvtColor(tmp_image, tmp_image, cv::COLOR_BGR2RGB);
		tmp_image.convertTo(tmp_image, CV_32FC3);
		float mean[3] = {0, 0, 0};
		float var = 255.0;

		input_image_t image;
		image.data      = tmp_image.data;
		image.width     = tmp_image.cols;
		image.height    = tmp_image.rows;
		image.channel   = tmp_image.channels();
		image.pixel_format = PIX_FMT_RGB888;
		
		yolov8n_preprocess(image, g_graph, g_nn_width, g_nn_height, g_nn_channel, mean, var, tensor);

		status = vsi_nn_RunGraph(g_graph);
		yolov8n_postprocess(g_graph, &resultData);
		
		draw_results(img, resultData, width, height);
		
		while (true) {
			output_mtx.lock();
			if (tmp == output_queue_num) {
				output_queue.push(img.clone());
				if (output_queue_num == thread_id - 1) {
					output_queue_num = 0;
				}
				else {
					output_queue_num++;
				}
				output_mtx.unlock();
				break;
			}
			output_mtx.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
    }
	vnn_ReleaseYolov8n(g_graph, TRUE);
	g_graph = NULL;
	
	return 0;
}

int main(int argc, char** argv){
	int c;
	while ((c = getopt_long(argc, argv, "d:w:h:m:t:n:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'd':
				device = optarg;
				break;

			case 'w':
				width = atoi(optarg);
				break;

			case 'h':
				height = atoi(optarg);
				break;

			case 'm':
				model_path = optarg;
				break;
			
			case 't':
				type = optarg;
				break;
			
			case 'n':
				thread_num = atoi(optarg);
				break;

			default:
				printf("%s [-d device] [-w width] [-h height] [-m model] [-t type] [-H]\n", argv[0]);
				exit(1);
		}
	}
	
	if (type.empty()) {
		std::cout << "Please choose camera type -t !!!" << std:: endl;
		return 0;
	}
	
	std::thread read_image_thread(read_image, type, device);
	std::thread show_image_thread(show_image);
	
	std::vector<std::thread> threads;
	for (int i = 0; i < thread_num; i++)
	{
		threads.emplace_back(run_detect_model, i, thread_num);
	}
	
	for (auto& t : threads)
	{
		t.join();
	}
	
	read_image_thread.join();
	show_image_thread.join();

	return 0;
}

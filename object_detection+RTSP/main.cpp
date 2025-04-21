#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <queue>
#include <sched.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <pthread.h>
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

#include "nn_detect_utils.h"
#include "yolov8n_process.h"
#include "vnn_yolov8n.h"

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
	{ "device",         required_argument,  NULL,   'd' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "model",          required_argument,  NULL,   'm' },
	{ "rtsp",           required_argument,  NULL,   'r' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

const char *device = DEFAULT_DEVICE;
const char *model_path;
std::string rtsp_url = "";

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

vsi_nn_graph_t * g_graph = NULL;

const static vsi_nn_postprocess_map_element_t* postprocess_map = NULL;
const static vsi_nn_preprocess_map_element_t* preprocess_map = NULL;

int width = MAX_WIDTH;
int height = MAX_HEIGHT;

#define DEFAULT_FRAME_RATE  30
int frame_rate = DEFAULT_FRAME_RATE;

struct  Frame
{
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;

int g_nn_height, g_nn_width, g_nn_channel;
pthread_mutex_t mutex4q;

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
			return (preprocess_map ? 1 : 0);
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
		return (postprocess_map ? 1 : 0);
}

static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(std::min(255.0, (double)colors[offset][2] * color_scale),
						std::min(255.0, (double)colors[offset][1] * color_scale),
						std::min(255.0, (double)colors[offset][0] * color_scale));
	return color;
}


static void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height){

	int i = 0;
	float left, right, top, bottom;

	for (i = 0; i < resultData.detect_num; i++) {
		left =  std::max(0.0f, resultData.point[i].point.rectPoint.left * img_width);
		right = std::min((float)img_width - 1, resultData.point[i].point.rectPoint.right * img_width);
		top = std::max(0.0f, resultData.point[i].point.rectPoint.top * img_height);
		bottom = std::min((float)img_height - 1, resultData.point[i].point.rectPoint.bottom * img_height);

		if (right <= left || bottom <= top) {
			continue;
		}

		cv::Rect rect(left, top, right-left, bottom-top);
		cv::Scalar color = obj_id_to_color(resultData.result_name[i].label_id);
		cv::rectangle(frame,rect, color ,1,8,0);

		int baseline;
		std::string label_text = resultData.result_name[i].label_name[0] != '\0'
						? resultData.result_name[i].label_name
						: "unknown";
		if (label_text.empty()) label_text = "unknown";

		cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);

		int text_rect_y = std::max(0, (int)(top - 20));
		int text_rect_height = 20;
		if (text_rect_y == 0) {
				text_rect_y = top;
				text_rect_height = std::min(20, (int)(bottom - top));
		}

		cv::Rect rect1(left, text_rect_y, std::min((int)(text_size.width+10), img_width - (int)left) , text_rect_height);
		cv::rectangle(frame,rect1, color ,-1);

		cv::Point text_org(std::max(0, (int)(left+5)), std::max(text_size.height, (int)(top-5)));
		if (text_org.y > text_rect_y + text_rect_height - baseline) {
			text_org.y = text_rect_y + text_rect_height - baseline;
		}
		if(text_org.y < text_size.height) text_org.y = text_size.height;

		cv::putText(frame,label_text,text_org,cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
	}
}

int run_detect_model(){
	int nn_height, nn_width, nn_channel;
	FILE* rtsp_pipe = nullptr;

	if (!rtsp_url.empty()) {
		std::string command =
							"gst-launch-1.0 fdsrc ! "
							"videoparse width=" + std::to_string(width) +
							" height=" + std::to_string(height) +
							" framerate=" + std::to_string(frame_rate) + "/1 format=bgr ! "
							"videoconvert ! "
							"amlvenc ! "
							"h264parse ! "
							"rtspclientsink location=" + rtsp_url + " latency=0 protocols=tcp";

		rtsp_pipe = popen(command.c_str(), "w");
		if (!rtsp_pipe) {
			std::cerr << "Failed to start gst!" << std::endl;
			return -1;
		}
		if (!rtsp_pipe) {
			perror("popen(gst-launch-1.0) failed");
			cerr << "Failed to open pipe to GStreamer. RTSP streaming disabled." << endl;
		} else {
			cout << "GStreamer pipeline launched for RTSP streaming to: " << rtsp_url << endl;
		}
	}

	uint32_t pre_count = vnn_GetPrePorcessMapCount();
	uint32_t post_count = vnn_GetPostPorcessMapCount();
	g_graph = vnn_CreateYolov8n(model_path, NULL,
			vnn_GetPrePorcessMap(), pre_count,
			vnn_GetPostPorcessMap(), post_count);

	if (!g_graph) {
			cerr << "Failed to create VNN graph!" << endl;
			if (rtsp_pipe) pclose(rtsp_pipe);
			return -1;
	}

	cout << "det_set_model success!!" << endl;

	vsi_nn_tensor_t *tensor = NULL;
	tensor = vsi_nn_GetTensor(g_graph, g_graph->input.tensors[0]);
	if (!tensor) {
		cerr << "Failed to get input tensor!" << endl;
		vnn_ReleaseYolov8n(g_graph, TRUE);
		if (rtsp_pipe) pclose(rtsp_pipe);
		return -1;
	}


	nn_width = tensor->attr.size[0];
	nn_height = tensor->attr.size[1];
	nn_channel = tensor->attr.size[2];

	cout << "\nmodel.width:" << nn_width <<endl;
	cout << "model.height:" << nn_height <<endl;
	cout << "model.channel:" << nn_channel << "\n" <<endl;

	g_nn_width = nn_width;
	g_nn_height = nn_height;
	g_nn_channel = nn_channel;

	DetectResult resultData;
	cv::Mat tmp_image(g_nn_width, g_nn_height, CV_8UC3);
	cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));
	struct timeval time_start, time_end;
	vsi_status status = VSI_FAILURE;

	string str = device;
	string res = str.substr(10);
	int capture_index = -1;
	try {
		capture_index = stoi(res);
	} catch (const std::invalid_argument& ia) {
		cerr << "Invalid device index: " << res << endl;
		vnn_ReleaseYolov8n(g_graph, TRUE);
		if (rtsp_pipe) pclose(rtsp_pipe);
		return -1;
	} catch (const std::out_of_range& oor) {
			cerr << "Device index out of range: " << res << endl;
			vnn_ReleaseYolov8n(g_graph, TRUE);
			if (rtsp_pipe) pclose(rtsp_pipe);
			return -1;
	}

	cv::VideoCapture cap(capture_index);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	if (!cap.isOpened()) {
		cout << "capture device failed to open!" << endl;
		cap.release();
		exit(-1);
	}

	while (cap.read(img)) {

		cv::Mat process_img;
		cv::resize(img, process_img, cv::Size(g_nn_width, g_nn_height));
		cv::cvtColor(process_img, process_img, cv::COLOR_BGR2RGB);
		process_img.convertTo(process_img, CV_32FC3);
		process_img = process_img / 255.0;

		input_image_t image;
		image.data      = process_img.data;
		image.width     = process_img.cols;
		image.height    = process_img.rows;
		image.channel   = process_img.channels();
		image.pixel_format = PIX_FMT_RGB888;

		gettimeofday(&time_start, 0);
		yolov8n_preprocess(image, g_graph, g_nn_width, g_nn_height, g_nn_channel, tensor);

		status = vsi_nn_RunGraph(g_graph);
		if (status != VSI_SUCCESS) {
			std::cerr << "vsi_nn_RunGraph error" << std::endl;
			break;
		}
		yolov8n_postprocess(g_graph, &resultData);

		gettimeofday(&time_end, 0);
		draw_results(img, resultData, width, height);
		fwrite(img.data, 1, img.total() * img.elemSize(), rtsp_pipe);
	}

	cout << "Releasing resources..." << endl;
	if (rtsp_pipe) {
		cout << "Closing GStreamer pipe..." << endl;
		pclose(rtsp_pipe);
		rtsp_pipe = nullptr;
	}

	cap.release();
	cv::destroyAllWindows();

	if (g_graph) {
			cout << "Releasing VNN graph..." << endl;
			vnn_ReleaseYolov8n(g_graph, TRUE);
			g_graph = NULL;
	}
		cout << "Cleanup complete." << endl;

	return 0;
}

int main(int argc, char** argv){
	int c;

	model_path = NULL;

	while ((c = getopt_long(argc, argv, "d:w:h:m:r:H", longopts, NULL)) != -1) {
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
				model_path  = optarg;
				break;

			case 'r':
				rtsp_url = optarg;
				break;

			case 'H':
			default:
				printf("Usage: %s [-d device] [-w width] [-h height] -m model [-r rtsp_url] [-H]\n", argv[0]);
				printf("Options:\n");
				printf("  -d, --device  Video device path (e.g., /dev/video0) [Default: %s]\n", DEFAULT_DEVICE);
				printf("  -w, --width   Capture width [Default: %d]\n", MAX_WIDTH);
				printf("  -h, --height  Capture height [Default: %d]\n", MAX_HEIGHT);
				printf("  -m, --model   Path to the YOLOv8n model file (Required)\n");
				printf("  -r, --rtsp    Output RTSP stream URL (e.g., rtsp://localhost:8554/mystream)\n");
				printf("  -H, --help    Show this help message\n");
				exit( (c == 'H' || c == '?') ? 0 : 1);
		}
	}

	if (model_path == NULL) {
		cerr << "Error: Model path (-m) is required." << endl;
		printf("Usage: %s [-d device] [-w width] [-h height] -m model [-r rtsp_url] [-H]\n", argv[0]);
		exit(1);
	}

	if (width <= 0 || height <= 0 || width > MAX_WIDTH || height > MAX_HEIGHT) {
			cerr << "Error: Invalid width or height specified. Max allowed: " << MAX_WIDTH << "x" << MAX_HEIGHT << endl;
			exit(1);
		}

	cout << "Configuration:" << endl;
	cout << "  Device: " << device << endl;
	cout << "  Resolution: " << width << "x" << height << endl;
	cout << "  Model Path: " << model_path << endl;
	if (!rtsp_url.empty()) {
		cout << "  RTSP Output URL: " << rtsp_url << endl;
	} else {
		cout << "  RTSP Output: Disabled" << endl;
	}
	cout << "---------------------" << endl;


	run_detect_model();

	return 0;
}

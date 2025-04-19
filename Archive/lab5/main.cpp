/*********************************************
* File: main.cpp
*
* Description: Lab 4 uses threads to make the Lab 3
				sobel filter faster
*
* Author: Nicolas Alvarez
*
* Version: 0.1
*
* Reference: 
**********************************************/
#include <stdio.h>
#include <pthread.h>
#include <cmath>
#include <arm_neon.h>
#include "main.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/***********************************************/
// Function Definitions
void* threadFrameSplit(void* threadSplitArgs);
void* threadSobel(void* inputThreadArgs);

pthread_t splitThread;
pthread_t sobelThread[3];

struct threadSplitArgs {
	Mat* input;
	Mat* grayScale;
	Mat* output;
	int rows;
	int cols;
	int stop;
};

struct threadArgs {
	Mat* input;
	Mat* grayScale;
	Mat* output;
	int start;
	int end;
	int stop;
};

pthread_barrier_t barrierSobel, barrierGrayScale, barrierStart, barrierContinue, barrierEnd;
/*********************************************/


/*********************************************
* Function: threadFrameSplit
*
* Description: Converts input grayscale Mat frame to sobel
* Input: Input full mat frame reference
* Output: Output sobel mat frame reference
*
**********************************************/
void* threadFrameSplit(void* threadSplitArgs) {

	threadArgs defaultThreadArgs = {
		.input = NULL,
		.grayScale = NULL,
		.output = NULL,
		.start = 0,
		.end = 0,
		.stop = 0
	};

	// Have to assign the threadArgs to a default to prevent a seg fault 
	// when creating the threads for the first time
	threadArgs thread1Args = defaultThreadArgs;
	threadArgs thread2Args = defaultThreadArgs;
	threadArgs thread3Args = defaultThreadArgs;
	threadArgs thread4Args = defaultThreadArgs;
	
	// 4 threads for 4 horizontal sections of the frame
	pthread_create(&sobelThread[0], NULL, threadSobel, (void *)&thread1Args);
	pthread_create(&sobelThread[1], NULL, threadSobel, (void *)&thread2Args);
	pthread_create(&sobelThread[2], NULL, threadSobel, (void *)&thread3Args);
	pthread_create(&sobelThread[3], NULL, threadSobel, (void *)&thread4Args);

	while(true) {

		pthread_barrier_wait(&barrierStart);

		struct threadSplitArgs *pStruct = (struct threadSplitArgs*)threadSplitArgs;

		// Unpacking threadSplitArgs to their respective var types
		Mat* inputFrame = (pStruct->input);
		Mat* grayScaleFrame = (pStruct->grayScale);
		Mat* outputFrame = (pStruct->output);
		int rows = (pStruct->rows);
		int cols = (pStruct->cols);
		int stopProcess = (pStruct->stop);

		// Breaks out of while loop if stop is set to '1' by main while
		if(stopProcess == 1) {
			pthread_barrier_wait(&barrierContinue);
			break;
		}

		thread1Args.input = inputFrame;
		thread1Args.grayScale = grayScaleFrame;
		thread1Args.output = outputFrame;
		thread1Args.start = 1;
		thread1Args.end = rows / 4;
		thread1Args.stop = stopProcess;

		thread2Args.input = inputFrame;
		thread2Args.grayScale = grayScaleFrame;
		thread2Args.output = outputFrame;
		thread2Args.start = (rows / 4) + 1; 
		thread2Args.end = rows / 2;
		thread1Args.stop = stopProcess;

		thread3Args.input = inputFrame;
		thread3Args.grayScale = grayScaleFrame;
		thread3Args.output = outputFrame;
		thread3Args.start = (rows / 2) + 1;
		thread3Args.end = (rows / 2) + (rows / 4);
		thread1Args.stop = stopProcess;

		thread4Args.input = inputFrame;
		thread4Args.grayScale = grayScaleFrame;
		thread4Args.output = outputFrame;
		thread4Args.start = (rows / 2) + (rows / 4) + 1;
		thread4Args.end = rows - 1;
		thread1Args.stop = stopProcess;
		
		pthread_barrier_wait(&barrierContinue);

		//Pad top and bottom border pixels as zero
		for(int i = 0; i < cols; ++i) {
			//First row
			grayScaleFrame->at<uchar>(0, i) = 0;
			//Last row
			grayScaleFrame->at<uchar>(rows - 1, i) = 0;
		}

		//Pad left and right border pixels as zero
		for(int j = 0; j < rows; ++j) {
			//First column
			grayScaleFrame->at<uchar>(j, 0) = 0;
			//Last column
			grayScaleFrame->at<uchar>(j, cols - 1) = 0;
		}

		// Wait for threads to reach barrier
		pthread_barrier_wait(&barrierGrayScale);
		

		// Wait for threads to reach barrier
		pthread_barrier_wait(&barrierSobel);
	}

	pthread_barrier_wait(&barrierEnd);
	return 0;
}

/*********************************************
* Function: threadSobel
*
* Description: Threading sobel into 4 threads
* Input: Input arbitrary mat frame size 
* Output: Output sobel mat frame reference
*
**********************************************/
void* threadSobel(void* inputThreadArgs) {
	while(true) {

		pthread_barrier_wait(&barrierContinue);

		struct threadArgs *sobelStruct = (struct threadArgs*)inputThreadArgs;
		
		// Unpacking inputThreadArgs back to their respective var types
		Mat* inputFrame = (sobelStruct->input);
		Mat* grayScaleFrame = (sobelStruct->grayScale);
		Mat* outputFrame = (sobelStruct->output);
		int start = (sobelStruct->start);
		int end = (sobelStruct->end);
		int stopProcess = (sobelStruct->stop);

		// Breaks out of while loop if stop is set to '1' by main while loop
		if(stopProcess == 1) break;

		// might need to increase size of pointer
		uint16_t* rgb_pixel_pointer = inputFrame;
		uint16_t* grayscale_pointer = grayScaleFrame;

		// red_weight = 0.2126
		// green_weight = 0.7152
		// blue_weight = 0.0722

		// 2167, 4683, and 472 are scaling factors in Q14 format. Rather than using floating point
		// numbers for the weights. The 14-bits that are allocated for the decimal point can be
		// represented as a whole number to perform non-floating point operations. As long as the
		// resulting number of the computation is scaled back to the appropriate range of values,
		// the estimate using fixed-point arithmetic "Q14" will be suitable.
		const uint16x8_t red_weight = {2167, 2167, 2167, 2167, 2167, 2167, 2167, 2167};
		const uint16x8_t green_weight = {4683, 4683, 4683, 4683, 4683, 4683, 4683, 4683};
		const uint16x8_t blue_weight = {472, 472, 472, 472, 472, 472, 472, 472};

		for(int row = 0, row <= end; ++row) { 							// Rows loop

			// Pointer for the beginning of each row
			rgb_pixel_pointer = inputFrame->ptr<uint8_t>(row);
			grayscale_pointer = grayScaleFrame->ptr<uint8_t>(row);

			for(int col = 0; col < inputFrame->cols; col+=8) { 			// Columns loop
				// Load 16 bits, 8 of them, 3 packed vectors
				uint16x8x3_t rgb_pixels = vld3_u8(rgb_pixel_pointer);

				// Access each packed vector of rgb_pixels for all 3 colors
				uint16x8_t red_channel = rgb_pixels.val[0];
				uint16x8_t green_channel = rgb_pixels.val[1];
				uint16x8_t blue_channel = rgb_pixels.val[2];
				
				// Apply grayscale conversion weights, and shift result right by 14 bits to
				// map back to appropriately sized values for the grayscale intensity
				uint16x8_t red_weighted = vshrq_n_u16(vmulq_u16(red_channel, red_weight), 14);
				uint16x8_t green_weighted = vshrq_n_u16(vmulq_u16(green_channel, green_weight), 14);
				uint16x8_t blue_weighted = vshrq_n_u16(vmulq_u16(blue_channel, blue_weight), 14);

				// Sum weighted color channels together for grayscale intensity
				uint16x8_t grayscale_pixels = vaddq_u16(vaddq_u16(red_weighted, green_weighted), blue_weighted);

				// Stores the resulting grayscale pixels at the specified pointer
				vst1q_u16(grayscale_pointer, grayscale_pixels);

				// Increment pointers for next iteration
				rgb_pixel_pointer += 8; // move to next 8 rgb pixels
				grayscale_pointer += 8; // moves to next empty position for 8 more grayscale pixels
			}															// End columns loop

			for(int col = (inputFrame->cols - (inputFrame->cols % 8)); col < inputFrame->cols; col++) {
				// Do remaining pixels
			}



		}															// End rows loop

		pthread_barrier_wait(&barrierGrayScale);

		if(stopProcess == 1) break;	

		const int16x8_t sobel_x_above = {-1, 0, 1, -1, 0, 1, 0, 0};
		const int16x8_t sobel_x_current = {-2, 0, 2, -2, 0, 2, 0, 0};
		const int16x8_t sobel_x_below = {-1, 0, 1, -1, 0, 1, 0, 0};

		const int16x8_t sobel_y_above = {1, 2, 1, 1, 2, 1, 0, 0};
		const int16x8_t sobel_y_current = {0, 0, 0, 0, 0, 0, 0, 0};
		const int16x8_t sobel_y_below = {-1, -2, -1, -1, -2, -1, 0, 0};

		const int width = outputFrame->cols;
		//uint16_t* output_pointer = outputFrame;

		for (int row = start; row <= end; ++row) {					// Rows loop

			// Pointer for the beginning of each row
			uint16_t* row_above = grayScaleFrame->ptr<uint8_t>(row - 1);
			uint16_t* row_current = grayScaleFrame->ptr<uint8_t>(row);
			uint16_t* row_below = grayScaleFrame->ptr<uint8_t>(row + 1);
			//output_pointer = outputFrame->ptr<uint8_t>(row);

			for (int col = 1; col < width; col+=2) {				// Columns loop

				//int 8x16 vld1q_s8
				//vectorize per pixel operation
				int16x8_t above_channel = vld1q_s16(row_above);
				int16x8_t current_channel = vld1q_s16(row_current);
				int16x8_t below_channel = vld1q_s16(row_below);

				int16x8_t g_x_vec = vaddq_s16(vaddq_s16(vmulq_s16(sobel_x_above, above_channel),
													    vmulq_s16(sobel_x_current, current_channel)),
													    vmulq_s16(sobel_x_below, below_channel));
				// add 0 and 2 for first pixel
				int16_t g_x_result_1 = vgetq_lane_s16(g_x_vec, 0) + vgetq_lane_s16(g_x_vec, 2);

				// add 3 and 5 for second pixel
				int16_t g_x_result_2 = vgetq_lane_s16(g_x_vec, 3) + vgetq_lane_s16(g_x_vec, 5);

				int16x8_t g_y_vec = vaddq_s16(vaddq_s16(vmulq_s16(sobel_y_above, above_channel),
													    vmulq_s16(sobel_y_current, current_channel)),
													    vmulq_s16(sobel_y_below, below_channel));
				// add 0, 1, and 2 for first pixel
				int16_t g_y_result_1 = vgetq_lane_s16(g_y_vec, 0) + vgetq_lane_s16(g_y_vec, 1) 
																  + vgetq_lane_s16(g_y_vec, 2);

				// add 3, 4, and 5 for second pixel
				int16_t g_y_result_2 = vgetq_lane_s16(g_y_vec, 3) + vgetq_lane_s16(g_y_vec, 4) 
																  + vgetq_lane_s16(g_y_vec, 5);
				
				//might be an issue with the way I am just setting int16_t to the result
				//might have to use a vstore somehow
				//maybe not, the vget does just return a int16_t value

				uint16_t gradient_mag_1 = saturate_cast<uchar>(std::abs(g_x_result_1) + std::abs(g_y_result_1));
				uint16_t gradient_mag_2 = saturate_cast<uchar>(std::abs(g_x_result_2) + std::abs(g_y_result_2));

				// store result								
				outputFrame->at<uchar>(row, col) = gradient_mag_1;
				outputFrame->at<uchar>(row, col + 1) = gradient_mag_2;

				// // Obtain a pointer to the beginning of a specific row in the matrix
				// // Row index should be within the range of the matrix
				// // This gives you a pointer to the first element of the row
				// uchar* rowPtr = frameMat->ptr<uchar>(row);

				// // Access individual elements within the row by indexing into the pointer
				// // You can access elements using column indices
				// uchar pixelValue = rowPtr[col];
			}														// End columns loop

			for(int col = (inputFrame->cols - (inputFrame->cols % 2)); col < inputFrame->cols; col++) {
				// Do remaining pixels
			}

			// Increment current row pointer by the width of the
			row_current += width;

		}															// End rows loop
		pthread_barrier_wait(&barrierSobel);

	}

	pthread_barrier_wait(&barrierEnd);
	return 0;
}


//int main(int argc, char *argv[]) {
int main(int argc, char** argv) {
	// Checks whether the number of command-line 
	// arguments is sufficient (Should be ./main <videofile>)
	if (argc != 2) {
        printf("Usage: %s <video_file>\n", argv[0]);
        return -1;
    }

	// Open the video file using the 2nd command-line argument
    VideoCapture cap(argv[1]);

	// Throws error if video does not open
    if (!cap.isOpened()) {
        printf("Error opening video\n");
        return -1;
    }

	cout << "Video conversion has begun. " 
		 << "Press 'x' to terminate" << endl;

    Mat inputFrame;
	Mat grayScaleFrame;
	Mat outputFrame;

	//printf("1\n");
	// Read first frame from the video and creates an output frame of same size.
	// The output frame is same size but single channel (1 unsigned char per pixel)
	cap.read(inputFrame);
	outputFrame.create(inputFrame.size(), CV_8UC1);
	grayScaleFrame.create(inputFrame.size(), CV_8UC1);

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	// Initialize pthread number of barriers before threads can continue
	pthread_barrier_init(&barrierGrayScale, NULL, 6);
	pthread_barrier_init(&barrierSobel, NULL, 6);
	pthread_barrier_init(&barrierStart, NULL, 2);
	pthread_barrier_init(&barrierContinue, NULL, 5); // Makes sobel threads wait for split thread before beginning grayscale
	pthread_barrier_init(&barrierEnd, NULL, 6);

	// Initial set of struct for frame splitting thread before it is created
	threadSplitArgs threadSplitArgs;

	threadSplitArgs.input = NULL;
	threadSplitArgs.grayScale = NULL;
	threadSplitArgs.output = NULL;
	threadSplitArgs.rows = 0;
	threadSplitArgs.cols = 0;
	threadSplitArgs.stop = 0;

	// 1 thread for splitting the input frame and feeding start and end rows to other 4 threads
	pthread_create(&splitThread, NULL, threadFrameSplit, (void *)&threadSplitArgs);
	//printf("3\n");
	//printf("splitter thread created\n");

    while (true) {
		// Continuous setting of struct for every new frame
		// maybe I can pass in one int variable that just holds the number of rows each thread will do
		threadSplitArgs.input = &inputFrame;
		threadSplitArgs.grayScale = &grayScaleFrame;
		threadSplitArgs.output = &outputFrame;
		threadSplitArgs.rows = inputFrame.rows;
		threadSplitArgs.cols = inputFrame.cols;

		//printf("splitter args set\n");

		// Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
		// Break the loop end if no more frames to grab
        if (waitKey(10) == 'x' || inputFrame.empty()) {
			threadSplitArgs.stop = 1;
			pthread_barrier_wait(&barrierStart);
			break;
		}

		//printf("1P\n");
		//printf("main thread reaches start barrier\n");
		pthread_barrier_wait(&barrierStart);
		//printf("parent thread passed start barrier\n");
	
		// // Wait for grayScale to finish
		//printf("3P\n");
		pthread_barrier_wait(&barrierGrayScale);

		// printf("Parent passed graybarrier\n");
		//printf("parent reaches sobelbarrier\n");
		// Wait for sobel to finish
		//printf("4P\n");
		pthread_barrier_wait(&barrierSobel);

		//printf("Parent passed sobelbarrier\n");

		// Display Sobel frame
		imshow("Lab 5 Sobel Frame", outputFrame);

		//printf("Parent displayed frame\n");

		// Read next frame from the video
		cap.read(inputFrame);
    }

	//printf("5P\n");
	pthread_barrier_wait(&barrierEnd);
    // Release the VideoCapture and close the window
    cap.release();
    //destroyAllWindows();
	pthread_barrier_destroy(&barrierGrayScale);
	pthread_barrier_destroy(&barrierSobel);
	pthread_barrier_destroy(&barrierStart);
	pthread_barrier_destroy(&barrierContinue);
	pthread_barrier_destroy(&barrierEnd);

	// Join threads
	for (int i = 0; i < 4; ++i) {
        pthread_join(sobelThread[i], NULL);
    }

	pthread_join(splitThread, NULL);

    return 0;
}

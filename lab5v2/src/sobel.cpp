/*********************************************************************
* File: sobel.cpp
*
* Description: Lab 5 uses intrinsics to complete vector operations
*			   to optimize grayscale and sobel operations
*
* Author: Nicolas Alvarez
*
* Version: 0.2
**********************************************************************/

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "sobel.hpp"

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>

using namespace cv;
using namespace std;

/*********************************************************************/

void* threadSobel(void* inputThreadArgs);
void* thread1Status = NULL;
void* thread2Status = NULL;
void* thread3Status = NULL;
void* thread4Status = NULL;

pthread_t sobelThread[4];


struct threadArgs {
	Mat* input;
	Mat* grayScale;
	Mat* output;
	int start;
	int end;
};

pthread_barrier_t barrierSobel, barrierGrayScale;

/************************************************************************/

		

/************************************************************************
* Function: threadSobel
*
* Description: Threading sobel into 4 threads
* Input: Input arbitrary mat frame size 
* Output: Output sobel mat frame reference
*
*************************************************************************/
void* threadSobel(void* inputThreadArgs) {

		struct threadArgs *sobelStruct = (struct threadArgs*)inputThreadArgs;
		
		// Unpacking inputThreadArgs back to their respective var types
		Mat* inputFrame = (sobelStruct->input);
		Mat* grayScaleFrame = (sobelStruct->grayScale);
		Mat* outputFrame = (sobelStruct->output);
		int start = (sobelStruct->start);
		int end = (sobelStruct->end);

		// might need to increase size of pointer
		uchar* rgb_pixel_pointer = NULL;
		uchar* grayscale_pointer = NULL;

		//printf("Non-intrinsic initialization within threadSobel function");

		/****************************Grayscale Computation***********************************/

		// 2167, 4683, and 472 are scaling factors in Q14 format. Rather than using floating point
		// numbers for the weights. The 14-bits that are allocated for the decimal point can be
		// represented as a whole number to perform non-floating point operations. As long as the
		// resulting number of the computation is scaled back to the appropriate range of values,
		// the estimate using fixed-point arithmetic "Q14" will be suitable.
		uint16x8_t red_weight = {2167, 2167, 2167, 2167, 2167, 2167, 2167, 2167};
		uint16x8_t green_weight = {4683, 4683, 4683, 4683, 4683, 4683, 4683, 4683};
		uint16x8_t blue_weight = {472, 472, 472, 472, 472, 472, 472, 472};

		//printf("Begin computations");

		for (int i = start; i < end; ++i) {					//ROWS

			// Pointer for the beginning of each row
			rgb_pixel_pointer = inputFrame->ptr(i);
			grayscale_pointer = grayScaleFrame->ptr(i);
			printf("rgb pntr: %p\n", (void*)((const uint16_t*)(rgb_pixel_pointer)));
			//printf("gray pntr: %p\n", (void*)grayscale_pointer);
			
			printf("\n");
			
			//printf("Pointers created");
			int hi = ((int)(inputFrame->cols) / 8) * 8; 
			printf("All cols: %d\n", inputFrame->cols);
			printf("div by 8: %d\n", hi);
			
			//might be an issue wiht the way I am iterating over the input frame!!!!!!!!!!!!!!!!!!!
			
			// Operates up to the number cols that is divisible by 8
			for (int j = 0; j < ((int)(inputFrame->cols) / 8) * 8; ++j) {	//COLS
				printf("%d --- rgb %p\n", j, rgb_pixel_pointer);
				//printf("gray: %p\n", grayscale_pointer);
				
				// Load 16 bits, 8 of them, 3 packed vectors
				uint16x8x3_t rgb_pixels = vld3q_u16((const uint16_t*)rgb_pixel_pointer);

				// Access each packed vector of rgb_pixels for all 3 colors
				uint16x8_t blue_channel = rgb_pixels.val[0];
				uint16x8_t green_channel = rgb_pixels.val[1];
				uint16x8_t red_channel = rgb_pixels.val[2];

				// Apply grayscale conversion weights, and shift result right by 14 bits to
				// map back to appropriately sized values for the grayscale intensity
				uint16x8_t red_weighted = vmulq_u16(red_channel, red_weight);
				uint16x8_t green_weighted = vmulq_u16(green_channel, green_weight);
				uint16x8_t blue_weighted = vmulq_u16(blue_channel, blue_weight);

				// Sum weighted color channels together for grayscale intensity
				uint16x8_t grayscale_pixels = vaddq_u16(red_weighted, green_weighted);
				grayscale_pixels = vaddq_u16(grayscale_pixels, blue_weighted);
				grayscale_pixels = vshrq_n_u16(grayscale_pixels, 14);

				// Stores the resulting grayscale pixels at the specified pointer
				vst1q_u16((uint16_t*)grayscale_pointer, grayscale_pixels);

				// Increment pointers for next iteration
				rgb_pixel_pointer += 24; // move to next 8 rgb pixels (3 channels * 8 pixels = 24)
				grayscale_pointer += 8; // moves to next empty position for 8 more grayscale pixels
				//printf("%d\n", j);
			}	

			//printf("Start of remaining grayscale pixels");

			//Compute remaining number of cols that was not divisible by 8 using traditional 
			for (int k = ((int)(inputFrame->cols) - ((int)(inputFrame->cols) % 8)); k < inputFrame->cols; ++k) {
				Vec3b pixel = inputFrame->at<Vec3b>(i, k);

				//Red = pixel[2];
				//Green = pixel[1];
				//Blue = pixel[0];

				//ITU-R (BT.709) recommended algorithm for grayscale
				uchar grayPixel = (0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]);
				//All pixels now represent 1 'intensity' value that will be used in the sobel
				grayScaleFrame->at<uchar>(i, k) = grayPixel;
			}
		}
		
		//printf("Grayscale barrier");
		
		//All pixels now represent 1 'intensity' value that will be used in the sobel
		// Wait for all threads to complete the grayScaleFrame
		pthread_barrier_wait(&barrierGrayScale);

		/****************************Sobel Compuations***********************************/

		// const int16x8_t sobel_x_above = {-1, 0, 1, -1, 0, 1, 0, 0};
		// const int16x8_t sobel_x_current = {-2, 0, 2, -2, 0, 2, 0, 0};
		// const int16x8_t sobel_x_below = {-1, 0, 1, -1, 0, 1, 0, 0};

		// const int16x8_t sobel_y_above = {1, 2, 1, 1, 2, 1, 0, 0};
		// const int16x8_t sobel_y_current = {0, 0, 0, 0, 0, 0, 0, 0};
		// const int16x8_t sobel_y_below = {-1, -2, -1, -1, -2, -1, 0, 0};

		// const int width = outputFrame->cols;

		// At this point, the section of the frame alotted for this thread is now grayscale
		// Next is to pass the grayscale through the sobel filter
		for (int i = start; i < end; ++i) {

			// Pointer for the beginning of each row
			// uint16_t* row_above = grayScaleFrame->ptr<uint8_t>(row - 1);
			// uint16_t* row_current = grayScaleFrame->ptr<uint8_t>(row);
			// uint16_t* row_below = grayScaleFrame->ptr<uint8_t>(row + 1);


			for (int j = 1; j < outputFrame->cols - 1; ++j) {

				//X and Y filter operations on surrounding intensity pixels
				//Had to upgrade the variable type from uchar to int to prevent overflow
				int g_x = (-1*grayScaleFrame->at<uchar>(i-1,j-1)) + grayScaleFrame->at<uchar>(i-1,j+1) +
						  (-2*grayScaleFrame->at<uchar>(i,j-1)) + 2*grayScaleFrame->at<uchar>(i,j+1) +
					      (-1*grayScaleFrame->at<uchar>(i+1,j-1)) + grayScaleFrame->at<uchar>(i+1,j+1);
				int g_y = (-1*grayScaleFrame->at<uchar>(i-1,j-1)) + -1*grayScaleFrame->at<uchar>(i-1,j+1) +
						  (-2*grayScaleFrame->at<uchar>(i-1,j)) + 2*grayScaleFrame->at<uchar>(i+1,j) +
						  (1*grayScaleFrame->at<uchar>(i+1,j-1)) + 1*grayScaleFrame->at<uchar>(i+1,j+1);
				

				//Approximation of Sobel without using pow or sqrt
				//A saturate cast of uchar is used to cut off the size of the computation if 
				//It is bigger than a uchar can hold.
				outputFrame->at<uchar>(i, j) = saturate_cast<uchar>(std::abs(g_x) + std::abs(g_y));
			}
		}
		// Wait for threads to finish Sobel frame before moving to the next frame
		pthread_barrier_wait(&barrierSobel);
		return 0;
}


int main(int argc, char** argv) {
	/****************************Process Command-Line Input***********************************/

	// Checks whether the number of command-line 
	// arguments is sufficient (Should be "./main <videofile>")
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

	/*****************************Barrier Initialization******************************************/

	// Initialize pthread number of barriers before threads can continue
	pthread_barrier_init(&barrierGrayScale, NULL, 5);
	pthread_barrier_init(&barrierSobel, NULL, 5);

	/**********************Creating Mats and Setting pThread Attributes******************************************/

	// Create input, grayscale, and output sobel mats
    Mat inputFrame, grayScaleFrame, outputFrame;
	threadArgs thread1Args, thread2Args, thread3Args, thread4Args;

	// Read first frame from the video and creates an output frame of same size.
	// The output and grayscale frame is same size but single channel (1 unsigned char per pixel)
	cap.read(inputFrame);
	outputFrame.create(inputFrame.size(), CV_8UC1);
	grayScaleFrame.create(inputFrame.size(), CV_8UC1);

	// Rows and columns
	uint num_rows = inputFrame.rows;
	uint num_cols = inputFrame.cols;

	uint thread_rows = num_rows / 4;

	// Set joinable attribute for all pthreads that are created
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	/**********************************Assigning pthread Arguments********************************************/

	// Set video capture at the first frame
	cap.set(CAP_PROP_POS_FRAMES, 0);
	double start_time = (double)cv::getTickCount();

	thread1Args.input = &inputFrame;
	thread1Args.grayScale = &grayScaleFrame;
	thread1Args.output = &outputFrame;

	thread2Args.input = &inputFrame;
	thread2Args.grayScale = &grayScaleFrame;
	thread2Args.output = &outputFrame;

	thread3Args.input = &inputFrame;
	thread3Args.grayScale = &grayScaleFrame;
	thread3Args.output = &outputFrame;

	thread4Args.input = &inputFrame;
	thread4Args.grayScale = &grayScaleFrame;
	thread4Args.output = &outputFrame;


	thread1Args.start = 1;
	thread1Args.end = thread_rows;

	thread2Args.start = thread_rows;
	thread2Args.end = thread_rows + thread_rows;

	thread3Args.start = thread_rows + thread_rows;
	thread3Args.end = thread_rows + thread_rows + thread_rows;

	thread4Args.start = thread_rows + thread_rows + thread_rows;
	thread4Args.end = num_rows - 1;

/***********************************Main While Loop*********************************************/

    while (true) {
		
		// Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
		// Break the loop end if no more frames to grab
        if (waitKey(50) == 'x' || inputFrame.empty()) {
			//threadSplitArgs.stop = 1;
			//pthread_barrier_wait(&barrierStart);
			break;
		}

		// Read next frame from the video
		cap.read(inputFrame);
		//printf("Read\n");
		

		// 4 threads for 4 horizontal sections of the frame
		pthread_create(&sobelThread[0], NULL, threadSobel, (void *)&thread1Args);
		pthread_create(&sobelThread[1], NULL, threadSobel, (void *)&thread2Args);
		pthread_create(&sobelThread[2], NULL, threadSobel, (void *)&thread3Args);
		pthread_create(&sobelThread[3], NULL, threadSobel, (void *)&thread4Args);

		// Wait for grayScale to finish
		pthread_barrier_wait(&barrierGrayScale);
		//printf("G\n");

		// Wait for sobel to finish
		pthread_barrier_wait(&barrierSobel);
		//printf("S\n");
		
		//Pad top and bottom border pixels as zero
		for(int i = 0; i < (int)num_cols; ++i) {
			//First row
			outputFrame.at<uchar>(0, i) = 0;
			//Last row
			outputFrame.at<uchar>(num_rows - 1, i) = 0;
		}

		//Pad left and right border pixels as zero
		for(int j = 0; j < (int)num_rows; ++j) {
			//First column
			outputFrame.at<uchar>(j, 0) = 0;
			//Last column
			outputFrame.at<uchar>(j, num_cols - 1) = 0;
		}

		// Display Sobel frame
		imshow("Lab 4 Sobel Frame", outputFrame);

		// Join threads
		int retJoinVal1 = pthread_join(sobelThread[0], &thread1Status);
		int retJoinVal2 = pthread_join(sobelThread[1], &thread2Status);
		int retJoinVal3 = pthread_join(sobelThread[2], &thread3Status);
		int retJoinVal4 = pthread_join(sobelThread[3], &thread4Status);

		//printf("Joined\n");
    }

	// Calculate elapsed time
    double end_time = (double)cv::getTickCount();
    double elapsed_time = (end_time - start_time) / cv::getTickFrequency();
    cout << "Total processing time: " << elapsed_time << " seconds" << endl;

    // Release the VideoCapture and close the window
    cap.release();

	pthread_barrier_destroy(&barrierGrayScale);
	pthread_barrier_destroy(&barrierSobel);

    return 0;
}

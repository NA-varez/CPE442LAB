/*********************************************************************
* File: main.cpp
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

#define FIXED_POINT_BITS 8
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)

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
*********************************************************************************/
void* threadSobel(void* inputThreadArgs) {

		struct threadArgs *sobelStruct = (struct threadArgs*)inputThreadArgs;
		
		// Unpacking inputThreadArgs back to their respective var types
		Mat* inputFrame = (sobelStruct->input);
		Mat* grayScaleFrame = (sobelStruct->grayScale);
		Mat* outputFrame = (sobelStruct->output);
		int start = (sobelStruct->start);
		int end = (sobelStruct->end);


		for (int i = start; i < end; ++i) {					//ROWS
			for (int j = 0; j < inputFrame->cols; ++j) {	//COLS
				Vec3b pixel = inputFrame->at<Vec3b>(i, j);

				//Red = pixel[2];
				//Green = pixel[1];
				//Blue = pixel[0];

				//ITU-R (BT.709) recommended algorithm for grayscale
				uchar grayPixel = (0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]);
				//All pixels now represent 1 'intensity' value that will be used in the sobel
				grayScaleFrame->at<uchar>(i, j) = grayPixel;
			}	
		}
		
		
		// All pixels now represent 1 'intensity' value that will be used in the sobel
		// Wait for all threads to complete the grayScaleFrame
		pthread_barrier_wait(&barrierGrayScale);
		
		uchar* first_row = NULL;
		uchar* second_row = NULL;
		uchar* third_row = NULL;
		
		uchar* sobel_frame_ptr = NULL;

		/****************************Sobel Compuations***********************************/

		// At this point, the section of the frame alotted for this thread is now grayscale
		// Next is to pass the grayscale through the sobel filter
		for (int i = start; i < end; ++i) {

			first_row = grayScaleFrame->ptr(i - 1);
			second_row = grayScaleFrame->ptr(i);
			third_row = grayScaleFrame->ptr(i + 1);
			
			sobel_frame_ptr = outputFrame->ptr(i);


			for (int col = 0; col < (inputFrame->cols - (inputFrame->cols % 8)); col+=6) {
		
				// Get 3, 8 element vectors loaded with 16-bit types of the next 3 rows
				uint16x8_t utop = vld1q_u16((const uint16_t*)first_row);
				uint16x8_t umid = vld1q_u16((const uint16_t*)second_row);
				uint16x8_t ubot = vld1q_u16((const uint16_t*)third_row);
				
				// Reinterpret vectors as signed before vector multiplies
				int16x8_t top = vreinterpretq_s16_u16(utop);
				int16x8_t mid = vreinterpretq_s16_u16(umid);
				int16x8_t bot = vreinterpretq_s16_u16(ubot);
				
				int16x8_t x_top_left = vmulq_n_s16(top, -1);
				//int16x8_t x_top_mid = vmulq_n_s16(top, 0);
				int16x8_t x_top_right = vmulq_n_s16(top, 1);
				
				int16x8_t x_mid_left = vmulq_n_s16(mid, -2);
				//int16x8_t x_mid_mid = vmulq_n_s16(mid, 0);
				int16x8_t x_mid_right = vmulq_n_s16(mid, 2);
				
				int16x8_t x_bot_left = vmulq_n_s16(bot, -1);
				//int16x8_t x_bot_mid = vmulq_n_s16(bot, 0);
				int16x8_t x_bot_right = vmulq_n_s16(bot, 1);
				
				
				// Add all G_x vectors
				int16x8_t g_x = vaddq_s16(vaddq_s16(vaddq_s16(x_top_left, x_top_right), 
				vaddq_s16(x_mid_left, x_mid_right)), 
				vaddq_s16(x_bot_left, x_bot_right));
				
				// Absolute value of G_x sum
				int16x8_t g_x_abs = vabsq_s16(g_x);
				
				int16x8_t y_top_left = vmulq_n_s16(top, 1);
				int16x8_t y_top_mid = vmulq_n_s16(top, 2);
				int16x8_t y_top_right = vmulq_n_s16(top, 1);
				
				//int16x8_t y_mid_left = vmulq_n_s16(mid, 0);
				//int16x8_t y_mid_mid = vmulq_n_s16(mid, 0);
				//int16x8_t y_mid_right = vmulq_n_s16(mid, 0);
				
				int16x8_t y_bot_left = vmulq_n_s16(bot, -1);
				int16x8_t y_bot_mid = vmulq_n_s16(bot, -2);
				int16x8_t y_bot_right = vmulq_n_s16(bot, -1);
				
				// Add all G_x vectors
				int16x8_t g_y = vaddq_s16(vaddq_s16(vaddq_s16(y_top_left, y_top_right), 
				vaddq_s16(y_top_mid, y_bot_mid)), 
				vaddq_s16(y_bot_left, y_bot_right));
				
				// Absolute value of G_y sum
				int16x8_t g_y_abs = vabsq_s16(g_y);
				
				// Add G_x and G_y sums
				int16x8_t final_sum = vaddq_s16(g_x_abs, g_y_abs);
				
				// Reinterpret as unsigned
				uint16x8_t sobel_pixels = vreinterpretq_u16_s16(final_sum);
				
				// Vector store 
				vst1q_u16((uint16_t*)sobel_frame_ptr, sobel_pixels);
				
				// Increment pointer for next 6 pixels pixels
				// Loop will move to next row if not enough pixels for a full vector 
				sobel_frame_ptr += 6;
				
				first_row += 6;
				second_row += 6;
				third_row += 6;
			}
		
		
		
		/**************************Remaining Sobel Computations**************************/

		// Operate on remaining pixles using normal un-vectorized operations
		for(int j = (inputFrame->cols - (inputFrame->cols % 8)); j < inputFrame->cols - 1; j++) {

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


		// Wait for threads to finish Sobel frame before outputting the frame and beginning the next frame
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
	//pthread_barrier_init(&barrierJoined, NULL, 1);

	/**********************Creating Mats and Setting pThread Attributes****************************/

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

    while (true) {
		
		// Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
		// Break the loop end if no more frames to grab
        if (waitKey(20) == 'x' || inputFrame.empty()) {
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
		for(uint i = 0; i < num_cols; ++i) {
			//First row
			outputFrame.at<uchar>(0, i) = 0;
			//Last row
			outputFrame.at<uchar>(num_rows - 1, i) = 0;
		}

		//Pad left and right border pixels as zero
		for(uint j = 0; j < num_rows; ++j) {
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

		//pthread_barrier_wait(&barrierJoined);
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
	//pthread_barrier_destroy(&barrierJoined);

    return 0;
}


//int 8x16 vld1q_s8
				//vectorize per pixel operation
				//int16x8_t above_channel = vld1q_s16(row_above);
				//int16x8_t current_channel = vld1q_s16(row_current);
				//int16x8_t below_channel = vld1q_s16(row_below);

				//int16x8_t g_x_vec = vaddq_s16(vaddq_s16(vmulq_s16(sobel_x_above, above_channel),
				//									    vmulq_s16(sobel_x_current, current_channel)),
				//									    vmulq_s16(sobel_x_below, below_channel));
				// add 0 and 2 for first pixel
				//int16_t g_x_result_1 = vgetq_lane_s16(g_x_vec, 0) + vgetq_lane_s16(g_x_vec, 2);

				// add 3 and 5 for second pixel
				//int16_t g_x_result_2 = vgetq_lane_s16(g_x_vec, 3) + vgetq_lane_s16(g_x_vec, 5);

				//int16x8_t g_y_vec = vaddq_s16(vaddq_s16(vmulq_s16(sobel_y_above, above_channel),
				//									    vmulq_s16(sobel_y_current, current_channel)),
				//									    vmulq_s16(sobel_y_below, below_channel));
				// add 0, 1, and 2 for first pixel
				//int16_t g_y_result_1 = vgetq_lane_s16(g_y_vec, 0) + vgetq_lane_s16(g_y_vec, 1) 
				//												  + vgetq_lane_s16(g_y_vec, 2);
//
				// add 3, 4, and 5 for second pixel
				//int16_t g_y_result_2 = vgetq_lane_s16(g_y_vec, 3) + vgetq_lane_s16(g_y_vec, 4) 
				//												  + vgetq_lane_s16(g_y_vec, 5);
				
				//might be an issue with the way I am just setting int16_t to the result
				//might have to use a vstore somehow
				//maybe not, the vget does just return a int16_t value

				//uint16_t gradient_mag_1 = saturate_cast<uchar>(std::abs(g_x_result_1) + std::abs(g_y_result_1));
				//uint16_t gradient_mag_2 = saturate_cast<uchar>(std::abs(g_x_result_2) + std::abs(g_y_result_2));

				// store result								
				//outputFrame->at<uchar>(row, col) = gradient_mag_1;
				//outputFrame->at<uchar>(row, col + 1) = gradient_mag_2;

				// Increment row pointers by n2 pixel locations
				//row_above += 2;
				//row_current += 2;
				//row_below += 2;

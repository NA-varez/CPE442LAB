/*********************************************************************
* File: main.cpp
*
* Description: Lab 6 uses persisting pthreads and more compiler flags
* 			   to optimize the sobel filter from lab5v3
*
* Author: Nicolas Alvarez
*
* Version: 0.1
**********************************************************************/

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "sobel.hpp"

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>

#define FIXED_POINT_BITS 7
#define FIXED_POINT_SCALE (1 << FIXED_POINT_BITS)

using namespace cv;
using namespace std;

/*********************************************************************/

void* threadSobel(void* inputThreadArgs);
void* thread1Status = NULL;
void* thread2Status = NULL;
void* thread3Status = NULL;
void* thread4Status = NULL;

pthread_t sobelThread[3];


struct threadArgs {
	Mat* input;
	Mat* grayScale;
	Mat* output;
	int start;
	int end;
	bool stop = 0;
};

pthread_barrier_t barrierSobel, barrierGrayScale, barrierFrameRead;

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
	while(true)
	{
		// Wait for next frame to be read by main
		pthread_barrier_wait(&barrierFrameRead);
		
		struct threadArgs *sobelStruct = (struct threadArgs*)inputThreadArgs;
		bool stop = (sobelStruct->stop);
		
		// If an end flag is recieved (no more frames) from main 
		// then break out of while loop
		if(stop) {
			break;
		}
		
		// Unpacking inputThreadArgs back to their respective var types
		Mat* inputFrame = (sobelStruct->input);
		Mat* grayScaleFrame = (sobelStruct->grayScale);
		Mat* outputFrame = (sobelStruct->output);
		int start = (sobelStruct->start);
		int end = (sobelStruct->end);
		int frame_columns = inputFrame->cols;
		int frame_columns_minus_one = frame_columns - 1;
		
		for (int i = start; i < end; ++i) {					//ROWS
			for (int j = 0; j < frame_columns; ++j) {	//COLS
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

				for (int col = 0; col < (frame_columns - (frame_columns % 8)); col+=6) {
					
					// Load grayscale values and reinterpret them as signed integers before operations
					int16x8_t top = vreinterpretq_s16_u16(vld1q_u16((const uint16_t*)first_row));
					int16x8_t mid = vreinterpretq_s16_u16(vld1q_u16((const uint16_t*)second_row));
					int16x8_t bot = vreinterpretq_s16_u16(vld1q_u16((const uint16_t*)third_row));
					
					// Shift grayscale values to the appropriate precision to scale for fixed-point format
					top = vshlq_n_s16(top, FIXED_POINT_BITS);
					mid = vshlq_n_s16(mid, FIXED_POINT_BITS);
					bot = vshlq_n_s16(bot, FIXED_POINT_BITS);

					// Sobel computation for X direction
					int16x8_t g_x = vaddq_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(top, -1), vmulq_n_s16(mid, -2)),
														vaddq_s16(vmulq_n_s16(bot, -1), vmulq_n_s16(top, 1))), 
											  vaddq_s16(vmulq_n_s16(mid, 2), vmulq_n_s16(bot, 1)));
					
					// Sobel computation for Y direction
					int16x8_t g_y = vaddq_s16(vaddq_s16(vaddq_s16(vmulq_n_s16(top, 1), vmulq_n_s16(bot, -1)),
														vaddq_s16(vmulq_n_s16(top, 2), vmulq_n_s16(bot, -2))),
											  vaddq_s16(vmulq_n_s16(top, 1), vmulq_n_s16(bot, -1)));

					// Combine gradients
					int16x8_t final_sum = vaddq_s16(vabsq_s16(g_x), vabsq_s16(g_y));

					// Convert back to integer format and scale to original range
					uint16x8_t sobel_pixels = vreinterpretq_u16_s16(vshrq_n_s16(final_sum, FIXED_POINT_BITS));

					// Vector store 
					vst1q_u16((uint16_t*)sobel_frame_ptr, sobel_pixels);

					// Increment pointers for next 6 pixels
					sobel_frame_ptr += 6;
					first_row += 6;
					second_row += 6;
					third_row += 6;
				}
				
		
		/**************************Remaining Sobel Computations**************************/
			
			// Operate on remaining pixles using normal un-vectorized operations
			for(int j = (frame_columns - (frame_columns % 8)); j < frame_columns_minus_one; j++) {
					
					int im1 = i - 1;
					int ip1 = i + 1;
					int jm1 = j - 1;
					int jp1 = j + 1;
					
					//X and Y filter operations on surrounding intensity pixels
					//Had to upgrade the variable type from uchar to int to prevent overflow
					int g_x = (-1*grayScaleFrame->at<uchar>(im1,jm1)) + grayScaleFrame->at<uchar>(im1,jp1) +
							  (-2*grayScaleFrame->at<uchar>(i,jm1)) + 2*grayScaleFrame->at<uchar>(i,jp1) +
							  (-1*grayScaleFrame->at<uchar>(ip1,jm1)) + grayScaleFrame->at<uchar>(ip1,jp1);
					int g_y = (-1*grayScaleFrame->at<uchar>(im1,jm1)) + -1*grayScaleFrame->at<uchar>(im1,jp1) +
							  (-2*grayScaleFrame->at<uchar>(im1,j)) + 2*grayScaleFrame->at<uchar>(ip1,jp1) +
							  (1*grayScaleFrame->at<uchar>(ip1,jm1)) + 1*grayScaleFrame->at<uchar>(ip1,jp1);
					

					//Approximation of Sobel without using pow or sqrt
					//A saturate cast of uchar is used to cut off the size of the computation if 
					//It is bigger than a uchar can hold.
					outputFrame->at<uchar>(i, j) = saturate_cast<uchar>(std::abs(g_x) + std::abs(g_y));
			}
		
		}


		// Wait for threads to finish Sobel frame before outputting the frame and beginning the next frame
		pthread_barrier_wait(&barrierSobel);
		
	}
	
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
	pthread_barrier_init(&barrierFrameRead, NULL, 5);

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
	uint num_rows_minus_one = num_rows - 1;
	uint num_cols_minus_one = num_cols - 1;

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
	
	
	// 4 threads for 4 horizontal sections of the frame
	pthread_create(&sobelThread[0], NULL, threadSobel, (void *)&thread1Args);
	pthread_create(&sobelThread[1], NULL, threadSobel, (void *)&thread2Args);
	pthread_create(&sobelThread[2], NULL, threadSobel, (void *)&thread3Args);
	pthread_create(&sobelThread[3], NULL, threadSobel, (void *)&thread4Args);


    while (true) {
		
		// Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
		// Break the loop end if no more frames to grab
        if (waitKey(20) == 'x' || inputFrame.empty()) {
			
			thread1Args.stop = 1;
			thread2Args.stop = 1;
			thread3Args.stop = 1;
			thread4Args.stop = 1;
			
			pthread_barrier_wait(&barrierFrameRead);
			break;
		}

		// Read next frame from the video
		cap.read(inputFrame);
		
		
		// Prevents threads from moving on before the next frame is read
		pthread_barrier_wait(&barrierFrameRead);
	
		// Wait for grayScale to finish
		pthread_barrier_wait(&barrierGrayScale);


		// Wait for sobel to finish
		pthread_barrier_wait(&barrierSobel);

		
		// Pad top and bottom border pixels as zero
		for(uint i = 0; i < num_cols; ++i) {
			outputFrame.at<uchar>(0, i) = 0;			//First row
			outputFrame.at<uchar>(num_rows_minus_one, i) = 0;	//Last row
		}

		// Pad left and right border pixels as zero
		for(uint j = 0; j < num_rows; ++j) {
			outputFrame.at<uchar>(j, 0) = 0;			//First column
			outputFrame.at<uchar>(j, num_cols_minus_one) = 0;	//Last column
		}

		// Display Sobel frame
		imshow("Lab 4 Sobel Frame", outputFrame);

    }
    
	// Join threads
	int retJoinVal1 = pthread_join(sobelThread[0], &thread1Status);
	int retJoinVal2 = pthread_join(sobelThread[1], &thread2Status);
	int retJoinVal3 = pthread_join(sobelThread[2], &thread3Status);
	int retJoinVal4 = pthread_join(sobelThread[3], &thread4Status);

	// Calculate elapsed time
    double end_time = (double)cv::getTickCount();
    double elapsed_time = (end_time - start_time) / cv::getTickFrequency();
    cout << "Total processing time: " << elapsed_time << " seconds" << endl;

    // Release the VideoCapture and close the window
    cap.release();
	
	// Destroy barriers
	pthread_barrier_destroy(&barrierGrayScale);
	pthread_barrier_destroy(&barrierSobel);
	pthread_barrier_destroy(&barrierFrameRead);

    return 0;
}



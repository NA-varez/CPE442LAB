/*********************************************************************
* File: main.cpp
*
* Description: Lab 4 uses multi-threading to 
*			   make the Lab 3 sobel filter faster
*			   For each frame a new thread is created and destroyed
*			   Two barriers are used to synchronize the grayscale and
*			   sobel computations.
*
* Author: Nicolas Alvarez
*
* Version: 0.1
**********************************************************************/

#include <stdio.h>
#include <pthread.h>
#include "sobel.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

/*********************************************************************/

void* threadSobel(void* inputThreadArgs);

pthread_t sobelThread[3];


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

		// Wait for threads to complete the grayScaleFrame
		pthread_barrier_wait(&barrierGrayScale);

		// At this point, the section of the frame alotted for this thread is now grayscale
		// Next is to pass the grayscale through the sobel filter
		for (int i = start; i < end; ++i) {
			for (int j = 1; j < outputFrame->cols; ++j) {

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

	// Set video capture at the first frame
	cap.set(CAP_PROP_POS_FRAMES, 0);
	double start_time = (double)cv::getTickCount();

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
		
		thread1Args.input = &inputFrame;
		thread1Args.grayScale = &grayScaleFrame;
		thread1Args.output = &outputFrame;
		thread1Args.start = 0;
		thread1Args.end = thread_rows;

		thread2Args.input = &inputFrame;
		thread2Args.grayScale = &grayScaleFrame;
		thread2Args.output = &outputFrame;
		thread2Args.start = thread_rows;
		thread2Args.end = thread_rows * 2;

		thread3Args.input = &inputFrame;
		thread3Args.grayScale = &grayScaleFrame;
		thread3Args.output = &outputFrame;
		thread3Args.start = (thread_rows * 2);
		thread3Args.end = thread_rows * 3;

		thread4Args.input = &inputFrame;
		thread4Args.grayScale = &grayScaleFrame;
		thread4Args.output = &outputFrame;
		thread4Args.start = (thread_rows * 3);
		thread4Args.end = num_rows;

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
		for(int i = 0; i <= inputFrame.cols; ++i) {
			//First row
			outputFrame.at<uchar>(0, i) = 0;
			//Last row
			outputFrame.at<uchar>(inputFrame.rows - 1, i) = 0;
		}

		//Pad left and right border pixels as zero
		for(int j = 0; j <= inputFrame.rows; ++j) {
			//First column
			outputFrame.at<uchar>(j, 0) = 0;
			//Last column
			outputFrame.at<uchar>(j, inputFrame.cols - 1) = 0;
		}

		// Display Sobel frame
		imshow("Lab 4 Sobel Frame", outputFrame);

		// Join threads
		for (int i = 0; i < 4; ++i) {
        	pthread_join(sobelThread[i], NULL);
    	}
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
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
#include "sobel.hpp"
#include "main.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>


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
	int start; // I dont know whether to make these pointers or not , same goies for the rows and cols variables in the above struct
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
	//printf("sobel threads created\n");

	while(true) {
		printf("1\n");
		pthread_barrier_wait(&barrierStart);
		//printf("splitter has passed start barrier\n");
		struct threadSplitArgs *pStruct = (struct threadSplitArgs*)threadSplitArgs;
		//threadSplitArgs* pStruct = new threadSplitArgs();

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
		thread2Args.start = (rows / 4) + 1; // might have to get rid of +1
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
		
		printf("2\n");
		//printf("splitter thread reaches CONTINUE BARRIER\n");
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
		printf("3\n");
		pthread_barrier_wait(&barrierGrayScale);
		//printf("split thread passed graybarrier\n");
		
		//printf("split thread reaches sobelbarrier\n");
		// Wait for threads to reach barrier
		printf("4\n");
		pthread_barrier_wait(&barrierSobel);
		//printf("split thread passed sobelbarrier\n");
	}
	printf("5\n");
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
		//printf("sobel thread reaches CONTINUE BARRIER\n");
		printf("2S\n");
		pthread_barrier_wait(&barrierContinue);
		//printf("sobel thread has passed CONTINUE BARRIER\n");
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

		for (int i = 0; i <= end; ++i) {						//ROWS
			for (int j = 0; j < inputFrame->cols; ++j) {		//COLS
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
		printf("3S\n");
		pthread_barrier_wait(&barrierGrayScale);
		//if(stopProcess == 1) break;	
		//printf("sobel thread passed graybarrier\n");

		// At this point, the section of the frame alotted for this thread is now grayscale
		// Next is to pass the grayscale through the sobel filter
		for (int i = start; i <= end; ++i) {
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
		//printf("sobel thread reaches sobelbarrier\n");
		printf("4S\n");
		// Wait for threads to finish Sobel frame before moving to the next frame
		pthread_barrier_wait(&barrierSobel);
		//printf("sobel thread passed sobel barrier\n");
	}
	printf("5S\n");
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

		printf("1P\n");
		//printf("main thread reaches start barrier\n");
		pthread_barrier_wait(&barrierStart);
		//printf("parent thread passed start barrier\n");
	
		// // Wait for grayScale to finish
		printf("3P\n");
		pthread_barrier_wait(&barrierGrayScale);

		// printf("Parent passed graybarrier\n");
		//printf("parent reaches sobelbarrier\n");
		// Wait for sobel to finish
		printf("4P\n");
		pthread_barrier_wait(&barrierSobel);

		//printf("Parent passed sobelbarrier\n");

		// Display Sobel frame
		imshow("Sobel Frame", outputFrame);

		//printf("Parent displayed frame\n");

		// Read next frame from the video
		cap.read(inputFrame);
    }

	printf("5P\n");
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
/*********************************************
*
* File: main.cpp
*
* Description: Lab 4 uses threads to make the Lab 3
				sobel filter faster
*
* Author: Nicolas Alvarez
*
* Version: 0.1
*
* Reference: Richard Rios: Helped fix uchar overflow in sobel function
*			 that was reducing quality of sobel output.
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


/*********************************************
* Function: to442_grayscale
*
* Description: Converts input Mat frame to grayscale
* Input: Input mat frame reference
* Output: Output grayscale mat frame reference
*
**********************************************/
void to442_grayscale(Mat& input, Mat& output) {
	// Check if input and output have the same size
    if (input.size() != output.size()) {
        cerr << "Input and output Mat sizes do not match!" << endl;
        return;
    }

    // Check if input is empty
    if (input.empty()) {
        cerr << "Input Mat is empty!" << endl;
        return;
    }
	
	//For each pixel in Mat
	for (int i = 0; i < input.rows - 1; ++i) {
		for (int j = 0; j < input.cols - 1; ++j) {
			Vec3b pixel = input.at<Vec3b>(i, j);

			//Red = pixel[2];
			//Green = pixel[1];
			//Blue = pixel[0];

			//ITU-R (BT.709) recommended algorithm for grayscale
			uchar grayPixel = (0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]);
			//All pixels now represent 1 'intensity' value that will be used in the sobel
			output.at<uchar>(i, j) = grayPixel;
		}	
	}	
}

/*********************************************
* Function: to442_sobel
*
* Description: Converts input grayscale Mat frame to sobel
* Input: Input grayscale mat frame reference
* Output: Output sobel mat frame reference
*
**********************************************/
void to442_sobel(Mat& input, Mat& output) {
	//For each pixel in Mat (except on border)
	//Starts sobel in row 1 column 1 (inclusive 0)
	for (int i = 1; i < input.rows - 1; ++i) {
		for (int j = 1; j < input.cols - 1; ++j) {

			//X and Y filter operations on surrounding intensity pixels
            //Had to upgrade the variable type from uchar to int to prevent overflow
			int g_x = (-1*input.at<uchar>(i-1,j-1)) + input.at<uchar>(i-1,j+1) +
					(-2*input.at<uchar>(i,j-1)) + 2*input.at<uchar>(i,j+1) +
					(-1*input.at<uchar>(i+1,j-1)) + input.at<uchar>(i+1,j+1);
			int g_y = (-1*input.at<uchar>(i-1,j-1)) + -1*input.at<uchar>(i-1,j+1) +
					(-2*input.at<uchar>(i-1,j)) + 2*input.at<uchar>(i+1,j) +
					(1*input.at<uchar>(i+1,j-1)) + 1*input.at<uchar>(i+1,j+1);
			

			//Approximation of Sobel without using pow or sqrt
			//A saturate cast of uchar is used to cut off the size of the computation if 
			//It is bigger than a uchar can hold.
			output.at<uchar>(i, j) = saturate_cast<uchar>(std::abs(g_x) + std::abs(g_y));
		}
	}

	//Pad top and bottom border pixels as zero
	for(int i = 0; i < input.cols; ++i) {
		//First row
		input.at<uchar>(0, i) = 0;
		//Last row
		input.at<uchar>(input.rows - 1, i) = 0;
	}

	//Pad left and right border pixels as zero
	for(int j = 0; j < input.rows; ++j) {
		//First column
		input.at<uchar>(j, 0) = 0;
		//Last column
		input.at<uchar>(j, input.cols - 1) = 0;
	}
}

/*********************************************
* Function: threadFrameSplit
*
* Description: Converts input grayscale Mat frame to sobel
* Input: Input full mat frame reference
* Output: Output full sobel mat frame reference
*
**********************************************/
void* threadFrameSplit(Mat& input, Mat& output) {
	//might need to leave the size() function out of this becuase Mat& is just a reference
	// the refereence cant support .size() ??????


	//add barrier for the read running this function
	//Feed Mat& references into threads computational functions
		
	//Stitch everything back together into full sobel frame
	
	//might not need Mat& output
}

/*********************************************
* Function: threadSobel
*
* Description: Threading sobel into 4 threads
* Input: Input arbitrary mat frame size 
* Output: Output sobel mat frame reference
*
**********************************************/
void* threadSobel(Mat& input, Mat& output) {



	//Barrier 
}


//int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
//*(*start_routine) (void *), void *arg);

//thread[4]???????????????
pthread_t thread[3];

struct threadArgs {
	int a;
	int b;
};

void* fnForThread1();
void* fnForThread2(void* threadArgs);
void* thread1Status;
void* thread2Status;

pthread_barrier_t barrier;

void setupPthreadBarrier(__uint16_t numThreads) {
	pthread_barrier_init(&barrier, NULL, numThreads)
}

//int main(int argc, char *argv[]) {
int main(int argc, char** argv) {
    
	//might need this to be 5 if the thread that splits the stuff needs to be included
	__uint16_t numThreads = 4;
	setupPthreadBarrier(numThreads);

	// Initialize attributes for new threads to be joinable
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	//Create new threads running assigned void* functions
	struct threadArgs = {.a = 0, .b = 1};
	int retVal1 = pthread_create(&thread[0], NULL, fnForThread1, NULL);
	int retVal2 = pthread_create(&thread[1], NULL, fnForThread2, (void *)&threadArgs);

	// Parent thread will continue 
	// until we reach a point when we wait for the child threads to finish
	int retVal3 = pthread_join(thread[0], &thread1Status);
	int retVal4 = pthread_join(thread[1], &thread2Status);
	//Parent collects child thread's result


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

	cout << "Video conversion has begun" 
		 << "Press 'x' to terminate" << endl;

	// Mat object for original frame,
    Mat inputFrame;

	// Read first frame from the video
	cap.read(inputFrame);

	// Create grayscale frame, and sobel frame with same size of input frame
	Mat grayscale, sobel;
	grayscale.create(inputFrame.size(), CV_8UC1);
	sobel.create(inputFrame.size(), CV_8UC1);

    while (true) {

        if (inputFrame.empty()) {
			break; // Break the loop end if no more frames to grab
        }

        // Convert to grayscale
        to442_grayscale(inputFrame, grayscale);

        //Apply Sobel filter
        to442_sobel(grayscale, sobel);

        // Display the result
        //imshow("Sobel Frame", grayscale);
		imshow("Sobel Frame", sobel);

        // Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
        if (waitKey(10) == 'x') {
            break;
        }

		// Read next frame from the video
		cap.read(inputFrame);
    }

    // Release the VideoCapture and close the window
    cap.release();
    //destroyAllWindows();

    return 0;
}
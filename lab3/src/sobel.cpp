/*********************************************
*
* File: sobel.cpp
*
* Description: Lab 3 main for sobel filter
*				creates an executable that takes
*				a video frame as input, converts the frame to grayscale
				then converts the grayscale frame to sobel frames.
				Outputs the frames in a new window.
*
* Author: Nicolas Alvarez
*
* Version: 0.1
*
* Reference: Richard Rios: Helped fix uchar overflow in sobel function
*			 that was reducing quality of sobel output.
**********************************************/
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "sobel.hpp"

#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

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
	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
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

	// Set video capture at the first frame
	cap.set(CAP_PROP_POS_FRAMES, 0);
	double start_time = (double)cv::getTickCount();

    while (true) {
		// Read frame from the video
		cap.read(inputFrame);

        if (inputFrame.empty()) {
			cout << "No more frames found" << endl;
			break; // Break the loop end if no more frames to grab
        }

        // Convert to grayscale
        to442_grayscale(inputFrame, grayscale);

        //Apply Sobel filter
        to442_sobel(grayscale, sobel);

        // Display the result
        //imshow("Lab 3 Sobel Frame", grayscale);
		imshow("Lab 3 Sobel Frame", sobel);

        // Stop processing if 'x' key is pressed within 10 ms
		// of the last sobel frame is shown
        if (waitKey(20) == 'x') {
			cout << "Output stopped by user" << endl;
            break;
        }
    }

	// Calculate elapsed time
    double end_time = (double)cv::getTickCount();
    double elapsed_time = (end_time - start_time) / cv::getTickFrequency();
    cout << "Total processing time: " << elapsed_time << " seconds" << endl;

    // Release the VideoCapture and close the window
    cap.release();

    return 0;
}

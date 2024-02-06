/*********************************************
*
* File: sobel.cpp
*
* Description: Contains functions for Sobel
*				filter functions
*
* Author: Nicolas Alvarez
*
* Version: 0.1
*
**********************************************/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "sobel.hpp"

using namespace cv;

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








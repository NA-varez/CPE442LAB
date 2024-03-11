/*********************************************
*
* File: DisplayTimage.cpp
*
* Description: Displays an image using the newly
* 		installed OpenCV
*
* Author: Nicolas Alvarez and Richard Rios
*
* Version: 0.1
*
**********************************************/

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "DisplayImage.hpp"

using namespace cv;
using namespace std;

int main(int num, char** argv)
{
	Mat image;
	image = imread(argv[1], IMREAD_COLOR);

	if (image.empty()) {
    	cout << "Error: Unable to read image '" << argv[1] << "'" << endl;
    	return 1;
   	}

	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", image);

	waitKey(0);
	return 0;
}


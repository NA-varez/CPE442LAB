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
#include <opencv2/opencv.hpp>
#include "DisplayImage.h"

using namespace cv;

int main(int num, char** argv)
{
	Mat image;
	image = imread(argv[1], IMREAD_COLOR);

	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", image);

	waitKey(0);
	return 0;
}


/*********************************************
*
* File: main.cpp
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
#include "sobel.hpp"
#include "main.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>



using namespace cv;
using namespace std;


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
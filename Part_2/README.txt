Hello! This is the README.txt for Lab 3! 

SOBEL IMPLEMENTATION

Within the lab3/ directory type the command:


make


Once the 'sobel' executable has been created follow up with:


./sobel video1.mov 		// Shorter video (Ocean Dive)
./sobel video1.mp4		// Longer video (Chameleon)



My code is heavily commented so please take a look at sobel.cpp in /src
It implements a sobel filter using opencv4 by first applying a Grayscale
and then applying the X and Y sobel gradients (and adding them up) for 
each pixel by grabbing nearby grayscale pixel data.

From the 5 times I ran it I got:

19.3006 seconds
19.3511 seconds
20.0628 seconds
19.5441 seconds
19.5384 seconds
________________
19.5594 seconds average

Without any sobel (just running the video normally on Windows) the video
is 5 seconds long

No optimization (yet!)

Hello! This is the README.txt for Lab 5! 

PTHREAD OPTIMIZATION USING NEON INTRINSICS (ASSEMBLY)

Within the lab5v3/ directory type the command:


make


Once the 'sobel' executable has been created follow up with:


./sobel video1.mov 		// Shorter video (Ocean Dive)
./sobel video1.mp4		// Longer video (Chameleon)


My code is heavily commented so please take a look at sobel.cpp in /src
Using the same computations for sobel from lab4v3, I used NEON intrinsics
to perform vector operations for the sobel filter. However, due to lack of time, I 
was unable to get to vectoring the grayscale computation.

Cool enough though, ive managed to get some descent precision using only
integer vector operations (no floating point)!

From the 5 times I ran it I got:

Previous results from lab4v3:

11.9018 seconds
11.8910 seconds
12.1033 seconds
12.0354 seconds
12.0936 seconds
________________
12.0050 seconds average

Results from lab5v3:

8.4138 seconds
8.9755 seconds
8.3846 seconds
8.1284 seconds
8.2969 seconds
________________
8.3498 seconds average


Average speedup = old time / new time = 1.4378 !



Without any sobel (just running the video normally on Windows) the video
is 5 seconds long


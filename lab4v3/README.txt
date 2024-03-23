Hello! This is the README.txt for Lab 4! 

PTHREAD OPTIMIZATION

Within the lab3/ directory type the command:


make


Once the 'sobel' executable has been created follow up with:


./sobel video1.mov 		// Shorter video (Ocean Dive)
./sobel video1.mp4		// Longer video (Chameleon)


My code is heavily commented so please take a look at sobel.cpp in /src
Using the same computations for sobel from lab3, it uses 4 pthreads and
a threadSobel function to split up the sobel frame into 4 strips.

Each strip has a range of rows to iterate over. 2 barriers were needed
to synchronize the pthreads. 1 barrier to wait for every thread to 
finish the grayscale frame, and 1 barrier to wait for every thread to
complete the sobel computations before displaying the thread.

From the 5 times I ran it I got:

Previous results from lab3:

19.3006 seconds
19.3511 seconds
20.0628 seconds
19.5441 seconds
19.5384 seconds
________________
19.5594 seconds average

Results from lab4:

11.9018 seconds
11.8910 seconds
12.1033 seconds
12.0354 seconds
12.0936 seconds
________________
12.0050 seconds average


Average speedup = old time / new time = 1.629



Without any sobel (just running the video normally on Windows) the video
is 5 seconds long


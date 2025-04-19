# README!!!
Check out the dropdowns for each lab!
Explains some neat things for each folder of code.

<details open>
<summary><h2>PART 1 - Makefile and Compiling with GCC</h2></summary>

### What is the point of a Makefile?
A Makefile is a way of ***automating*** the process of compiling code. There are 3 advantages to a Makefile that are at the core of its use:

1. Automatically determine what files have been edited and need to be rebuilt (recompiled).
2. If working in a team. Everyone can stay consistent with the same build process.
3. Saves time by rebuilding only what is necessary.


### What is an object file and why is it created for C/C++ files?
An object file is an intermediate compilation product that is not yet linked into an executable. It has the machine code, the source code's symbol table, relocation information, and debug information. Multiple object files can be created and linked into an executable. It can be converted to its assembly.

### What is in the executable that is created?
This has all of the machine code from all object files, linked together. At this point, all of the functions have their real addresses, program entry point, and any static data is defined and ready to go.

### O.K. Here is the Code
```make
CC = g++  # GCC Compiler

# The CFLAGS variable is used a bit later in the object creation line
# In short, tells the compiler to look into the 'include' directory for header files then,
# pkg-config is asked for the needed compiler flags (additional directories) for OpenCV4
CFLAGS = -I$(IDIR) $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

# File directory names
ODIR = obj
IDIR = include
SRCDIR = src
```
'-I' = "add the following directory to the include search path"
'$(___)' = expand the value of / compute whatever is inside the parenthesis
'pkg-config' is a helper tool that provides information about installed libraries

NOTE: '-I' only means something to the g++ compiler, not interpreted by Make

```make
# Adds path prefixes to each file name in _DEPS
_DEPS = DisplayImage.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# Adds path prefix to each object file in _OBJ
_OBJ = DisplayImage.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
```
'patsubst' substitues paths for each occurance of the pattern described by the first argument. 
  Second argmument is the format for each substitution you want
  Third argument is the text you are iterating over in the 'for each' loop

```make
# For all .o files in ODIR use all of the .cpp files from SRCDIR as input and consider the dependencies
$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)  # Compile command

# Target is the executable 'DisplayImage'
# OBJ is the list of object files required
# -o $@ sets the output to the target name
# CFLAGS are all the compilation flags
# LIBS are all of the library flags for linking
DisplayImage: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

# Removes the possibility that Make sees a file named 'clean' and when
# You run 'make clean' it assumes the target file is already satisfied
.PHONY: clean

# Command for force removing all files in the output directory with the file extension .o
clean:
	rm -f $(ODIR)/*.o
```


The format for a rule is:
Target (the file you want to create): Prerequisites (files needed to create target)
	Commands (actions to execute to create target)


### The CODE Code

The actual CODE code of this first part is pretty self explanatory. Explanation not really needed.

To run the code:
```c
./DisplayImage <imagehere.jpg>
```

This'll just pop up a window displaying the image you specified as long as that image is in the project directory.


![part1](https://github.com/user-attachments/assets/87c57d82-d03d-41c3-b87e-f381add23e67)
</details>


<details open>
<summary><h2>PART 2 - Sobel Filtered Image and Video</h2></summary>

### OpenCV what?

Open Source Computer Vision Library. Useful. Very helpful. Can use it to do operations on pixels!
That is important for the Sobel Filter.

### What is A Sobel Filter?

A Sobel Filter makes the edges/features of an image white, and everything else black (ideally).
Used for edge detection. The Sobel filter convolves an image's pixel data with 2 (one horizontal and one vertical)
kernels that create the approximate horizontal and vertical intensitiy deriviatve components of an image.

The horizontal derivative is the X component of the Z intensity gradient, G_X: dZ/dX.
Vertical derivative is the Y component of the Z intensity gradient, G_Y: dZ/dY.

#### Refresher on Gradients! (Hint: It's a 3-Dimensional derivative)

Gradients come alive in mathematics when we start considering a 3rd dimension for derivatives. A gradient has a magnitude and a direction. The magnitude tells you how steep the slope is at that point in 3D space. The direction tells you the direction of the steepest ascent. In this case we are only interested in the magnitude (how steep the change in intensity is) at an X, Y position of each pixel. X and Y provide information for what pixel we are interested in. The Z is the intensity axis.

The two 3x3 kernels are applied at each pixel position. As the program is iterating through every pixel, you can think of the kernels as sliding over to the next pixel position in the row (by incrementing the column variable). 


### A Grayscale?

A Sobel Fiter can be made on a full color RGB image. But this is uneccessary.
From what I can tell, the Sobel Filter really just needs to operate on the *intensity* of each pixel. So before filtering, each frame's pixel data is reduced to 1 intensity value.

### How does that Look in Code?

*The following are code snippets of things that are non-trivial! Just explaining the code that makes the magic happen.*

### The Grayscale Operation 

```c
	Vec3b pixel = input.at<Vec3b>(i, j);

	//Red = pixel[2];
	//Green = pixel[1];
	//Blue = pixel[0];

	//ITU-R (BT.709) recommended algorithm for grayscale
	uchar grayPixel = (0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]);
	//All pixels now represent 1 'intensity' value that will be used in the sobel
	output.at<uchar>(i, j) = grayPixel;
```

Iteration over every pixel's RGB values to create the grayscale frame.
Uses some algo for grayscaling from RGB values. You can read about it if you want.

### The Sobel Operation


```c
	int g_x = (-1*input.at<uchar>(i-1,j-1)) + input.at<uchar>(i-1,j+1) +
			(-2*input.at<uchar>(i,j-1)) + 2*input.at<uchar>(i,j+1) +
			(-1*input.at<uchar>(i+1,j-1)) + input.at<uchar>(i+1,j+1);
	int g_y = (-1*input.at<uchar>(i-1,j-1)) + -1*input.at<uchar>(i-1,j+1) +
			(-2*input.at<uchar>(i-1,j)) + 2*input.at<uchar>(i+1,j) +
			(1*input.at<uchar>(i+1,j-1)) + 1*input.at<uchar>(i+1,j+1);

	output.at<uchar>(i, j) = saturate_cast<uchar>(std::abs(g_x) + std::abs(g_y));
```

The above code is within a typical double for-loop for iterating over a matrix.
The loop does not iterate over the bordering pixels since the 3x3 kernel requires an area of 3x3 pixels to do an operation. Each multiplication is done on a pixel that is offset from the position (i, j) in a 3x3 area. For example (i-1, j-1) is the top left corner pixel that is diagonal to pixel (i, j).

If this were not an approximation the last line there for the output pixel would be sqrt(G_X^2 + G_Y^2), the 'Euclidean Norm' Pythagorean Theorem, Straight-line distance, *Style* magnitude. Instead the Manhatten Disatnce, G_X + G_Y is used.

After ALL THAT, to keep the same size as the input frame, a border of black pixels is added.



### Images!

![image](https://github.com/user-attachments/assets/0a39da2b-4cf6-4101-b584-45b242f54732)

![image](https://github.com/user-attachments/assets/e1c24e35-d06c-442e-8439-c37eaded7169)

### Video!

https://github.com/user-attachments/assets/3dc04aa4-f92e-4427-afe0-3cbc9722236d



</details>


<details open>
<summary><h2>PART 3 - Threading with pThreads and Barriers</h2></summary>

Going to be honest, this is where things start to get busy.


### Threading and Barriers

Threading is one of the first things you begin to consider to increase speed of processing. Threading allows you to parallelize your program. The number of threads you are capable of running at one time, depends on the number of CPU cores you have on your hardware. For the Raspberry Pi 3, I've got 4 to work with.

### Threading Initialization and Void Star
In order for each thread to be successful, they first need to know what they are going to work on. To get them started we do this:

```c
void* threadSobel(void* inputThreadArgs);
void* thread1Status = NULL;
void* thread2Status = NULL;
void* thread3Status = NULL;
void* thread4Status = NULL;

pthread_t sobelThread[3];

struct threadArgs {
	Mat* input;
	Mat* grayScale;
	Mat* output;
	int start;
	int end;
	bool stop = 0;
};
```
First the 'thread function' prototype and each variable to hold thread statuses.

A struct will package up all the information each thread needs to complete their process: Adresses of Mat frames, starting and ending row, and bool for the ability to stop processing before completion of all frames.

Inside the 'threadSobel' function (the function that each thread will run),
we have void* unpacking, grayscaling, sobelling, outputting.

Unpacking was weird when I first learned this C++ phenomenon, but makes sense 
the more I look understand whats happening.

```c
void* threadSobel(void* inputThreadArgs) {
	while(true)
	{
		// Wait for next frame to be read by main
		pthread_barrier_wait(&barrierFrameRead);
		
		struct threadArgs* sobelStruct = (struct threadArgs*)inputThreadArgs;
		bool stop = (sobelStruct->stop);
		
		// If an end flag is recieved (no more frames) from main 
		// then break out of while loop
		if(stop) {
			break;
		}
		
		// Unpacking inputThreadArgs back to their respective var types
		Mat* inputFrame = (sobelStruct->input);
		Mat* grayScaleFrame = (sobelStruct->grayScale);
		Mat* outputFrame = (sobelStruct->output);
		int start = (sobelStruct->start);
		int end = (sobelStruct->end);
		int frame_columns = inputFrame->cols;
```

Without looking it up myself, but trying to remember instead, a void* is an address to
no particular type. A different way.. a void* is a pointer that does not return its type. That allows you to do some neat stuff. Let's look at this line:

```c
	struct threadArgs* sobelStruct = (struct threadArgs*)inputThreadArgs;
```

WOW. A mouthful.

So, this function takes in a void* type variable. Much like how you can dereference a pointer to get the value, you can dereference the void*. BUT.. you have to cast that derefernce back into its type so you can make use of it. Otherwise its some weird void* type thing.

Here is that line of code in words:
Create a struct threadArgs pointer (this was defined earlier) named 'sobelStruct'
Assign the sobelStruct pointer to the struct-threadArgs-pointer-re-casted inputThreadArgs void*.

After that, the variables are then unpacked from the sobelStruct pointer.


### Barriers

To outfit your program with threads you will inevitably need barriers to prevent your threads from proceeding before some process is done. In our case, the frame is split horizontally into 4 parts for each thread. This is the jist of how the threadSobel function is organized with barriers.


```c
	// Wait for next frame to be read by main
	pthread_barrier_wait(&barrierFrameRead);

	... threadArgs unpacking and grayscale ...

	// All pixels now represent 1 'intensity' value that will be used in the sobel
	// Wait for all threads to complete the grayScaleFrame
	pthread_barrier_wait(&barrierGrayScale);

	... sobel processing ...

	// Wait for threads to finish Sobel frame before outputting the frame and beginning the next frame
	pthread_barrier_wait(&barrierSobel);
```

### Main

In the main function its a bit self-explanatory whats happening.
I just need to initialize each thread and fill up each of their respective threadArgs structs with the appropriate values, dynamically (without knowing the resolution (size of each frame) of the video).

</details>


<details open>
<summary><h2>PART 3B - Vector Operations for Optimization</h2></summary>

Whats the vector operations for?

It's always best to reduce the number of loops your program does. The operations in the loop inevitably fetch memory over and over again. Every memory fetch and counter increment you do for each loop is time consuming. So what if you could do multiple math operations at once?

This is where vector operations come in. SIMD, ISA, ARM Architecture, SIMD, NEON, ISA... What??

Abbreviations are a curse for learning! Okay, so SIMD stands for (Single Instruction Multiple Data). SIMD is the name for a particular way of doing parallelization the same way we might say TSA to desribe that line you have to stand in before you get into the airport terminal.

Every processor can support a particular ISA (Instruction Set Architecture). But not all implement SIMD technology. In this case, for the Raspberry Pi 3, it uses a Broadcom BCM2837 SoC which includes an ARM Cortex-A53 quad-core processor. That processor implements the ARMv8-A architecture which includes NEON SIMD. There we go, we've it down to the silicon.

How?

Using ARM NEON Intrinsics of course! These are available because the Raspberry Pi 3 has a 

NEON registers made available by the ARM-Cortex-A53 on the Raspberry pi 3 are 128 bits wide.

So the most we could parallelize with a single NEON register is 8 16-bit elements OR any other combination offered by [NEON](https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[Neon]) that multiplies to 128 bits.

We start by using 3 int16x8_t to load the 8 top, middle, and bottom pixel portions required for 6 full sobel pixel calculations. Lets break this down. Here our kernels again for the X and Y gradients:

|-1|0|+1|
|-2|0|+2|
|-1|0|+1|


|+1|+2|+1|
|0|0|0|
|-1|-2|-1|




```c



```






</details>


<details open>
<summary><h2>PART 3C - Compiler Optimizations for Optimization</h2></summary>







</details>


<details open>
<summary><h2>PART 4 - Optimizing Optimization!</h2></summary>






</details>





<details open>
<summary><h2>Part 5 - The Speedup Results</h2></summary>







</details>




<details open>
<summary><h2>PART ? - Questions and Where it can be Improved</h2></summary>





</details>


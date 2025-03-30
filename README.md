# README!!!
Check out the dropdowns for each lab!
Explains some neat things.


<h2>PART 1 - Makefile and Compiling with GCC</h2>

(Lab 2 folder)

### What is the point of a Makefile?
A Makefile is a way of ***automating*** the process of compiling code. There are 3 advantages to a Makefile that are at the core of its use:

1. Automatically determine what files have been edited and need to be rebuilt (recompiled).
2. If working in a team. Everyone can stay consistent with the same build process.
3. Saves time by rebuilding only what is necessary.


### What is an object file and why is it created for C/C++ files?
An object file is an intermediate compilation product that is not yet linked into an executable. It has the machine code, the source code's symbol table, relocation information, and debug information. Multiple object files can be created and linked into an executable. It can be converted to its assembly.

### What is in the executable that is created?
This has all of the machine code from all object files, linked together. At this point, all of the functions have their real addresses, program entry point, and any static data is defined and ready to go.

## O.K. Here is the Code
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


## The CODE code

The actual CODE code of this first part is pretty self explanatory. Explanation not really needed.

To run the code:
```c
./DisplayImage <imagehere.jpg>
```

This'll just pop up a window displaying the image you specified as long as that image is in the project directory.


![part1](https://github.com/user-attachments/assets/87c57d82-d03d-41c3-b87e-f381add23e67)




<h2>PART 2 - Sobel Filtered Video</h2>


### OpenCV what?

Open Source Computer Vision Library. Useful. Very helpful. Can use it to do operations on pixels!
That is important for the Sobel Filter.

### What is A Sobel Filter?

A Sobel Filter makes the edges/features of an image white, and everything else black (ideally).
Used for edge detection. The Sobel filter convolves an image's pixels with 2 (one horizontal and one vertical)
kernels that create the approximate horizontal and vertical intensitiy deriviatve components of an image.


### A Grayscale?



## How does that Look in Code?

###The following are code snippets of things that are non-trivial! Just explaining the code that makes the magic happen.


```c



```



## Images!

![image](https://github.com/user-attachments/assets/0a39da2b-4cf6-4101-b584-45b242f54732)

![image](https://github.com/user-attachments/assets/e1c24e35-d06c-442e-8439-c37eaded7169)



```c

```






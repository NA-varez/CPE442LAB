# README!!!
Check out the dropdowns for each lab!
Explains some neat things.

<details>
<summary>LAB 1</summary>
<br>

## Makefile and Compiling with GCC

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
```c
CC = g++
CFLAGS = -I$(IDIR) $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

ODIR = obj
IDIR = include
SRCDIR = src
```
The GCC compiler selected then,
CFLAGS (used later) is set to the include directory



Explanation here


</details>



<details>
<summary>LAB 1</summary>
<br>

## Hi

```c

```

Explanation here


</details>


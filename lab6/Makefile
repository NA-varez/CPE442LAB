CC = g++
CFLAGS =-mfpu=neon -std=c++11 -g -O3 -I$(IDIR) $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4) -lpthread

ODIR = obj
IDIR = include
SRCDIR = src

_DEPS = sobel.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = sobel.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -g -c -o $@ $< $(CFLAGS)

sobel: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o

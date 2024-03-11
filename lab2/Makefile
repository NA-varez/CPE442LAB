CC = g++
CFLAGS = -I$(IDIR) $(shell pkg-config --cflags opencv4)
LIBS = $(shell pkg-config --libs opencv4)

ODIR = obj
IDIR = include
SRCDIR = src

_DEPS = DisplayImage.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = DisplayImage.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

DisplayImage: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o


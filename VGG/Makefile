# Makefile for building hello.cpp with SystemC

# Compiler
CXX = g++

# Directories
INCLUDE_DIR = /root/systemC/systemc-2.3.4/include
LIB_DIR = /root/systemC/systemc-2.3.4/lib-linux64

# Flags
DEBUG_FLAGS = -g -O2 -Wall -Wextra -DDEBUG -Wno-unused-variable
CXXFLAGS = -I. -I$(INCLUDE_DIR)  
LDFLAGS = -L. -L$(LIB_DIR) -Wl,-rpath=$(LIB_DIR)
LIBS = -lsystemc -lm 

# Target and source files
TARGET = main
SRC = compute_VGG.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -O3 $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# Clean up
clean:
	rm -f $(TARGET)

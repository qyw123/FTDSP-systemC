# Makefile for building hello.cpp with SystemC

# Compiler
CXX = g++

# Directories
INCLUDE_DIR = /root/systemC/systemc-2.3.4/include
LIB_DIR = /root/systemC/systemc-2.3.4/lib-linux64

# Flags
CXXFLAGS = -I. -I$(INCLUDE_DIR)
LDFLAGS = -L. -L$(LIB_DIR) -Wl,-rpath=$(LIB_DIR)
LIBS = -lsystemc -lm

# Target and source files
TARGET = main
SRC = testbench.cpp 

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# Clean up
clean:
	rm -f $(TARGET)

CXX = g++
CXXFLAGS = -O2 -Wall -std=c++17
TARGET = fft
SRC = radix2.cpp

INPUT = ./data/fft_16777216_input.txt
OUTPUT = ./data/fft_output.txt

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lm -lstdc++fs -lcnpy

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O3 -fopenmp -Iinclude

# Target executable
TARGET = main

# Source files
SRC = src/activation.cpp src/base.cpp src/layers.cpp src/loss.cpp src/matrix.cpp src/optimizer.cpp main.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove compiled files
clean:
	rm -f $(OBJ) $(TARGET)

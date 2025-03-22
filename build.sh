#!/bin/bash

# Set colors for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Building Parallel Matrix Multiplication Project${NC}"
echo -e "${GREEN}==================================================${NC}"

# Find CMake command - try system installation first
CMAKE_CMD="cmake"
if ! command -v $CMAKE_CMD &> /dev/null; then
    echo -e "${YELLOW}System-wide CMake not found, checking alternatives...${NC}"
    
    # Try common alternative locations
    if command -v /usr/bin/cmake &> /dev/null; then
        CMAKE_CMD="/usr/bin/cmake"
    elif command -v /usr/local/bin/cmake &> /dev/null; then
        CMAKE_CMD="/usr/local/bin/cmake"
    else
        echo -e "${RED}Error: CMake not found. Please install CMake and try again.${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}Using CMake at: $(which $CMAKE_CMD)${NC}"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p build
fi

# Enter build directory
cd build

# Run CMake
echo -e "${YELLOW}Configuring project with CMake...${NC}"
$CMAKE_CMD .. -DCMAKE_BUILD_TYPE=Release

# Check if CMake succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CMake configuration failed.${NC}"
    exit 1
fi

# Determine the number of CPU cores for parallel build
# NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
NUM_CORES=20
if [ $NUM_CORES -gt 1 ]; then
    # Use N-1 cores for compilation to avoid system lockup
    NUM_CORES=$((NUM_CORES - 1))
fi

# Build the project
echo -e "${YELLOW}Building project with ${NUM_CORES} parallel jobs...${NC}"
make -j${NUM_CORES}

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "You can find all executables in the 'build' directory."
echo -e "\nAvailable executables:"
find . -maxdepth 1 -type f -executable -not -path "*/\.*" | sort | while read file; do
    echo -e "  - ${YELLOW}$(basename $file)${NC}"
done

echo -e "\nTo run benchmarks, execute:"
echo -e "  ${YELLOW}cd build && ./benchmark_all${NC}"
echo -e "  ${YELLOW}./scripts/run_benchmarks.sh${NC}"

echo -e "\nTo run tests, execute:"
echo -e "  ${YELLOW}cd build && ./test_correctness${NC}"

echo -e "${GREEN}==================================================${NC}"

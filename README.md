<img width="953" alt="image" src="https://github.com/user-attachments/assets/1cbf9844-e5ee-45ef-8088-f66ef167fa78" /># Edge Detection Application

A C++ application implementing the Canny edge detection algorithm with support for both grayscale and color images. Built using OpenCV and modern C++17 features.

<img width="953" alt="image" src="https://github.com/user-attachments/assets/0a77d277-a4de-4f8a-bfd0-eccb7c302874" />


⚠️ **Note:** This application is currently optimized for macOS and uses macOS-specific file dialog. It may not work correctly on other operating systems.

## Features

- Grayscale and color image edge detection
- Real-time parameter adjustment using trackbars
- Side-by-side comparison view
- Native macOS file picker support
- Implementation of the complete Canny edge detection pipeline

## Dependencies

- macOS operating system
- OpenCV 4.x
- C++17 compatible compiler
- CMake 3.x

## Building

1. Create a build directory:
```bash
mkdir build && cd build
```
2. Run CMake to configure the project:
```bash
cmake ..
```
3. Build the project:
```bash
make
```
4. Run the application:
```bash
./edge_detector
```

## Usage
1. Click the "Load Image" button to select an image file.
2. Adjust the parameters using the trackbars to see real-time changes in edge detection.
3. The application will display the original image and the edge-detected image side by side.
4. Press "q" to exit the application.



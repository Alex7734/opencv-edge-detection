#ifndef EDGE_DETECTOR_UI_HPP
#define EDGE_DETECTOR_UI_HPP

#include "edge_detector.hpp"
#include <opencv2/opencv.hpp>

class EdgeDetectorUI {
public:
    explicit EdgeDetectorUI(const std::string& imagePath = "");
    void run();

private:
    bool loadImage(const std::string& imagePath);
    std::string selectImageFile();

    static void trackbarCallback(int, void* userdata);
    void updateDisplay();
    void createWindows();
    void createTrackbars();
    void processImages();
    void displayResults();

    cv::Mat originalImage;
    cv::Mat grayImage;
    cv::Mat display;
    cv::Mat banner;

    struct Parameters {
        int lowThresholdRatio = 5;
        int highThresholdRatio = 15;
        int sigmaValue = 14;
    } params;

    static constexpr int LABEL_HEIGHT = 30;
};

#endif // EDGE_DETECTOR_UI_HPP
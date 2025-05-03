#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <opencv2/opencv.hpp>

struct GradientParams {
    cv::Mat source;
    double sigma;
    float lowThreshold;
    float highThreshold;
    bool isColor;
};

struct GradientResult {
    cv::Mat magnitude;
    cv::Mat direction;
};

class EdgeDetector {
public:
    static cv::Mat process(const GradientParams& params);

private:
    static int calculateGaussianKernelSize(double sigma);
    static cv::Mat applyGaussianBlur(const cv::Mat& source, double sigma);
    static GradientResult computeGradients(const cv::Mat& image, bool isColor);
    static GradientResult computeGrayGradients(const cv::Mat& image);
    static GradientResult computeColorGradients(const cv::Mat& image);
    static cv::Mat applySuppression(const GradientResult& gradients);
    static cv::Mat applyThresholding(const cv::Mat& suppressed, float lowThreshold, float highThreshold);
};

cv::Mat cannyEdgeDetectionGray(const cv::Mat& src, double sigma = 1.4,
                              float lowThresholdRatio = 0.05,
                              float highThresholdRatio = 0.15);

cv::Mat cannyEdgeDetectionColor(const cv::Mat& src, double sigma = 1.4,
                               float lowThresholdRatio = 0.05,
                               float highThresholdRatio = 0.15);

#endif // EDGE_DETECTOR_HPP
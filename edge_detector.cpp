#include "edge_detector.hpp"
#include <cmath>

/**
 * Calculate Gaussian kernel size based on sigma
 * Ensures:
 * - Minimum size of 3x3
 * - Size is odd (required by OpenCV)
 * - Size is proportional to sigma (6*sigma+1 rule)
 *
 * @param sigma Gaussian standard deviation
 * @return Appropriate kernel size
 */
int EdgeDetector::calculateGaussianKernelSize(double sigma) {
    return std::max(3, static_cast<int>(6 * sigma + 1) | 1);
}

/**
 * Apply Gaussian blur to the source image
 * @param source Input image
 * @param sigma Standard deviation for Gaussian kernel
 * @return Blurred image
 */
cv::Mat EdgeDetector::applyGaussianBlur(const cv::Mat& source, double sigma) {
    cv::Mat blurred;
    int kernelSize = calculateGaussianKernelSize(sigma);
    cv::GaussianBlur(source, blurred, cv::Size(kernelSize, kernelSize), sigma);

    cv::Mat floatImage;
    blurred.convertTo(floatImage, CV_32F);
    return floatImage;
}

/**
* Calculate gradient coefficients (g-matrix)
* For grayscale:
* - gxx = |∂I/∂x|²
* - gyy = |∂I/∂y|²
* - gxy = (∂I/∂x)(∂I/∂y)
* @param gradientX
* @param gradientY
* @param gxx
* @param gyy
* @param gxy
*/
void computeCoefficientsGray(const cv::Mat& gradientX, const cv::Mat& gradientY,
                            cv::Mat& gxx, cv::Mat& gyy, cv::Mat& gxy) {
    gxx = cv::Mat::zeros(gradientX.size(), CV_32F);
    gyy = cv::Mat::zeros(gradientX.size(), CV_32F);
    gxy = cv::Mat::zeros(gradientX.size(), CV_32F);

    for (int y = 0; y < gradientX.rows; y++) {
        for (int x = 0; x < gradientX.cols; x++) {
            float dRdx = gradientX.at<float>(y, x);
            float dRdy = gradientY.at<float>(y, x);
            gxx.at<float>(y, x) = dRdx * dRdx;
            gyy.at<float>(y, x) = dRdy * dRdy;
            gxy.at<float>(y, x) = dRdx * dRdy;
        }
    }
}

/**
 * Calculate gradient coefficients for color images
 * Combines RGB channels:
 * - gxx = Σ|∂C/∂x|² for C in {R,G,B}
 * - gyy = Σ|∂C/∂y|² for C in {R,G,B}
 * - gxy = Σ(∂C/∂x)(∂C/∂y) for C in {R,G,B}
 * @param gradientX
 * @param gradientY
 * @param gxx
 * @param gyy
 * @param gxy
 */
void computeCoefficientsColor(const std::vector<cv::Mat>& gradientX,
                            const std::vector<cv::Mat>& gradientY,
                            cv::Mat& gxx, cv::Mat& gyy, cv::Mat& gxy) {
    gxx = cv::Mat::zeros(gradientX[0].size(), CV_32F);
    gyy = cv::Mat::zeros(gradientX[0].size(), CV_32F);
    gxy = cv::Mat::zeros(gradientX[0].size(), CV_32F);

    for (int y = 0; y < gradientX[0].rows; y++) {
        for (int x = 0; x < gradientX[0].cols; x++) {
            for (int i = 0; i < 3; i++) {
                gxx.at<float>(y, x) += std::pow(gradientX[i].at<float>(y, x), 2);
                gyy.at<float>(y, x) += std::pow(gradientY[i].at<float>(y, x), 2);
                gxy.at<float>(y, x) += gradientX[i].at<float>(y, x) * gradientY[i].at<float>(y, x);
            }
        }
    }
}

/**
 * Calculate gradient magnitude using formula:
 * F₀(x,y) = √[1/2((gxx + gyy) + (gxx - gyy)cos2θ + 2gxy sin2θ)]
 *
 * @param gxx
 * @param gyy
 * @param gxy
 * @param magnitude
 */
void calculateGradientMagnitude(const cv::Mat& gxx, const cv::Mat& gyy,
                              const cv::Mat& gxy, cv::Mat& magnitude) {
    magnitude = cv::Mat::zeros(gxx.size(), CV_32F);

    for (int y = 0; y < gxx.rows; y++) {
        for (int x = 0; x < gxx.cols; x++) {
            float numerator = 2 * gxy.at<float>(y, x);
            float denominator = gxx.at<float>(y, x) - gyy.at<float>(y, x);
            float theta = denominator < 0 ? CV_PI / 4.0 : 0.5 * std::atan2(numerator, denominator);

            float cos2theta = std::cos(2 * theta);
            float sin2theta = std::sin(2 * theta);

            magnitude.at<float>(y, x) = std::sqrt(0.5 * (
                (gxx.at<float>(y, x) + gyy.at<float>(y, x)) +
                (gxx.at<float>(y, x) - gyy.at<float>(y, x)) * cos2theta +
                2 * gxy.at<float>(y, x) * sin2theta
            ));
        }
    }
}

/**
 * Calculate gradient direction using formula:
 * θ(x,y) = (1/2)tan⁻¹[2gxy/(gxx - gyy)]
 *
 * @param gxx
 * @param gyy
 * @param gxy
 * @param direction
 */
void calculateGradientDirection(const cv::Mat& gxx, const cv::Mat& gyy,
                              const cv::Mat& gxy, cv::Mat& direction) {
    direction = cv::Mat::zeros(gxx.size(), CV_32F);

    for (int y = 0; y < gxx.rows; y++) {
        for (int x = 0; x < gxx.cols; x++) {
            float numerator = 2 * gxy.at<float>(y, x);
            float denominator = gxx.at<float>(y, x) - gyy.at<float>(y, x);
            direction.at<float>(y, x) = denominator < 0 ?
                CV_PI / 4.0 : 0.5 * std::atan2(numerator, denominator);
        }
    }
}

/**
 * Compute gradients for grayscale images
 * @param image Input image
 * @return GradientResult containing magnitude and direction
 */
GradientResult EdgeDetector::computeGrayGradients(const cv::Mat& image) {
    cv::Mat gradientX, gradientY;
    cv::Sobel(image, gradientX, CV_32F, 1, 0, 3);
    cv::Sobel(image, gradientY, CV_32F, 0, 1, 3);

    cv::Mat gxx, gyy, gxy;
    computeCoefficientsGray(gradientX, gradientY, gxx, gyy, gxy);

    GradientResult result;
    calculateGradientMagnitude(gxx, gyy, gxy, result.magnitude);
    calculateGradientDirection(gxx, gyy, gxy, result.direction);
    return result;
}

/**
 * Compute gradients for color images
 * @param image Input image
 * @return GradientResult containing magnitude and direction
 */
GradientResult EdgeDetector::computeColorGradients(const cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    std::vector<cv::Mat> gradientX(3), gradientY(3);
    for (int i = 0; i < 3; i++) {
        cv::Sobel(channels[i], gradientX[i], CV_32F, 1, 0, 3);
        cv::Sobel(channels[i], gradientY[i], CV_32F, 0, 1, 3);
    }

    cv::Mat gxx, gyy, gxy;
    computeCoefficientsColor(gradientX, gradientY, gxx, gyy, gxy);

    GradientResult result;
    calculateGradientMagnitude(gxx, gyy, gxy, result.magnitude);
    calculateGradientDirection(gxx, gyy, gxy, result.direction);
    return result;
}

/**
 * Compute gradients based on image type (color or grayscale)
 * @param image Input image
 * @param isColor Flag indicating if the image is color
 * @return GradientResult containing magnitude and direction
 */
GradientResult EdgeDetector::computeGradients(const cv::Mat& image, bool isColor) {
    return isColor ? computeColorGradients(image) : computeGrayGradients(image);
}

/**
 * Apply non-maximum suppression to the gradient magnitude
 * Thin edges by suppressing non-maximum pixels along gradient direction
 * Uses 8 possible directions (0°, 45°, 90°, 135°)
 * @param gradients GradientResult containing magnitude and direction
 * @return Suppressed gradient magnitude
 */
cv::Mat EdgeDetector::applySuppression(const GradientResult& gradients) {
    cv::Mat suppressed = cv::Mat::zeros(gradients.magnitude.size(), CV_32F);

    for (int y = 1; y < gradients.magnitude.rows - 1; y++) {
        for (int x = 1; x < gradients.magnitude.cols - 1; x++) {
            float angle = gradients.direction.at<float>(y, x);
            float angleDeg = angle * 180.0 / CV_PI;
            if (angleDeg < 0) angleDeg += 180.0;

            float q = 255.0, r = 255.0;

            if ((0 <= angleDeg && angleDeg < 22.5) || (157.5 <= angleDeg && angleDeg <= 180)) {
                q = gradients.magnitude.at<float>(y, x+1);
                r = gradients.magnitude.at<float>(y, x-1);
            }
            else if (22.5 <= angleDeg && angleDeg < 67.5) {
                q = gradients.magnitude.at<float>(y+1, x-1);
                r = gradients.magnitude.at<float>(y-1, x+1);
            }
            else if (67.5 <= angleDeg && angleDeg < 112.5) {
                q = gradients.magnitude.at<float>(y+1, x);
                r = gradients.magnitude.at<float>(y-1, x);
            }
            else if (112.5 <= angleDeg && angleDeg < 157.5) {
                q = gradients.magnitude.at<float>(y-1, x-1);
                r = gradients.magnitude.at<float>(y+1, x+1);
            }

            suppressed.at<float>(y, x) = gradients.magnitude.at<float>(y, x) >= q &&
                                        gradients.magnitude.at<float>(y, x) >= r ?
                                        gradients.magnitude.at<float>(y, x) : 0;
        }
    }
    return suppressed;
}

/**
 * Double thresholding and edge tracking
 * 1. Classify pixels as strong/weak edges using thresholds
 * 2. Keep strong edges
 * 3. Keep weak edges connected to strong edges
 * 4. Discard other weak edges
 *
 * @param suppressed
 * @param lowThreshold
 * @param highThreshold
 */
cv::Mat EdgeDetector::applyThresholding(const cv::Mat& suppressed,
                                           float lowThreshold, float highThreshold) {
    double maxVal;
    cv::minMaxLoc(suppressed, nullptr, &maxVal);

    float highThr = highThreshold * maxVal;
    float lowThr = lowThreshold * maxVal;

    cv::Mat strong = cv::Mat::zeros(suppressed.size(), CV_8U);
    cv::Mat weak = cv::Mat::zeros(suppressed.size(), CV_8U);

    for (int y = 0; y < suppressed.rows; y++) {
        for (int x = 0; x < suppressed.cols; x++) {
            float val = suppressed.at<float>(y, x);
            if (val >= highThr) strong.at<uchar>(y, x) = 255;
            else if (val >= lowThr) weak.at<uchar>(y, x) = 255;
        }
    }

    cv::Mat edges = strong.clone();

    static constexpr int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    bool changed;
    do {
        changed = false;
        for (int y = 1; y < edges.rows - 1; y++) {
            for (int x = 1; x < edges.cols - 1; x++) {
                if (weak.at<uchar>(y, x) == 255 && edges.at<uchar>(y, x) == 0) {
                    for (int i = 0; i < 8; i++) {
                        if (edges.at<uchar>(y + dy[i], x + dx[i]) == 255) {
                            edges.at<uchar>(y, x) = 255;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    } while (changed);

    return edges;
}

/**
 * Main processing function for Canny edge detection
 * 1. Apply Gaussian blur
 * 2. Compute gradients (magnitude and direction)
 * 3. Apply non-maximum suppression
 * 4. Apply double thresholding and edge tracking
 *
 * @param params GradientParams containing input image and parameters
 * @return Processed image with edges detected
 */
cv::Mat EdgeDetector::process(const GradientParams& params) {
    auto blurred = applyGaussianBlur(params.source, params.sigma);
    auto gradients = computeGradients(blurred, params.isColor);
    auto suppressed = applySuppression(gradients);
    return applyThresholding(suppressed, params.lowThreshold, params.highThreshold);
}
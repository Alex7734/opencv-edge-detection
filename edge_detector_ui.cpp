#include "edge_detector_ui.hpp"
#include <filesystem>
#include <sstream>

EdgeDetectorUI::EdgeDetectorUI(const std::string& imagePath) {
    if (imagePath.empty()) {
        if (!loadImage(selectImageFile())) {
            throw std::runtime_error("No image selected or invalid image");
        }
    } else {
        if (!loadImage(imagePath)) {
            throw std::runtime_error("Could not open or find the image: " + imagePath);
        }
    }
}

bool EdgeDetectorUI::loadImage(const std::string& imagePath) {
    originalImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (originalImage.empty()) {
        return false;
    }
    cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);
    return true;
}

std::string EdgeDetectorUI::selectImageFile() {
    FILE* pipe = popen("osascript -e 'tell application \"System Events\"' "
                      "-e 'activate' "
                      "-e 'set theFile to choose file with prompt \"Select an image:\" "
                      "of type {\"public.image\"}' "
                      "-e 'POSIX path of theFile' "
                      "-e 'end tell'", "r");

    if (!pipe) {
        return "";
    }

    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    result.erase(0, result.find_first_not_of(" \n\r\t"));
    result.erase(result.find_last_not_of(" \n\r\t") + 1);

    return result;
}

void EdgeDetectorUI::createWindows() {
    cv::namedWindow("Canny Edge Detection Comparison", cv::WINDOW_NORMAL);
    cv::namedWindow("Parameters", cv::WINDOW_NORMAL);

    cv::resizeWindow("Canny Edge Detection Comparison", originalImage.cols * 3, originalImage.rows);
    cv::resizeWindow("Parameters", originalImage.cols * 3, LABEL_HEIGHT);
}

void EdgeDetectorUI::trackbarCallback(int, void* userdata) {
    auto* ui = static_cast<EdgeDetectorUI*>(userdata);
    ui->updateDisplay();
}

void EdgeDetectorUI::createTrackbars() {
    cv::createTrackbar("Low Threshold (%)", "Parameters", &params.lowThresholdRatio, 100, trackbarCallback, this);
    cv::createTrackbar("High Threshold (%)", "Parameters", &params.highThresholdRatio, 100, trackbarCallback, this);
    cv::createTrackbar("Sigma (x10)", "Parameters", &params.sigmaValue, 50, trackbarCallback, this);
}

void EdgeDetectorUI::processImages() {
    float lowThr = static_cast<float>(params.lowThresholdRatio) / 100.0f;
    float highThr = static_cast<float>(params.highThresholdRatio) / 100.0f;
    double sigma = static_cast<double>(params.sigmaValue) / 10.0;

    // Process images
    cv::Mat grayEdges = EdgeDetector::process({
        .source = grayImage,
        .sigma = sigma,
        .lowThreshold = lowThr,
        .highThreshold = highThr,
        .isColor = false
    });

    cv::Mat colorEdges = EdgeDetector::process({
        .source = originalImage,
        .sigma = sigma,
        .lowThreshold = lowThr,
        .highThreshold = highThr,
        .isColor = true
    });

    // Create display
    int rows = originalImage.rows;
    int cols = originalImage.cols;
    display = cv::Mat(rows + LABEL_HEIGHT, cols * 3, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat labelRegion = display(cv::Rect(0, 0, cols * 3, LABEL_HEIGHT));
    cv::Mat imageRegion = display(cv::Rect(0, LABEL_HEIGHT, cols * 3, rows));

    // Add labels
    cv::putText(labelRegion, "Original", cv::Point(cols/3, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(labelRegion, "Grayscale ED", cv::Point(cols + cols/3 - 20, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(labelRegion, "Color ED", cv::Point(2*cols + cols/3 - 10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Copy images
    cv::Mat regionOriginal = imageRegion(cv::Rect(0, 0, cols, rows));
    cv::Mat regionGray = imageRegion(cv::Rect(cols, 0, cols, rows));
    cv::Mat regionColor = imageRegion(cv::Rect(cols * 2, 0, cols, rows));

    originalImage.copyTo(regionOriginal);
    cv::Mat grayEdgesColor, colorEdgesColor;
    cv::cvtColor(grayEdges, grayEdgesColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(colorEdges, colorEdgesColor, cv::COLOR_GRAY2BGR);
    grayEdgesColor.copyTo(regionGray);
    colorEdgesColor.copyTo(regionColor);

    // Update parameters banner
    std::stringstream ss;
    ss << "Low Threshold: " << lowThr << " | High Threshold: " << highThr << " | Sigma: " << sigma;
    banner = cv::Mat(LABEL_HEIGHT, cols * 3, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(banner, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

void EdgeDetectorUI::displayResults() {
    cv::imshow("Canny Edge Detection Comparison", display);
    cv::imshow("Parameters", banner);
}

void EdgeDetectorUI::updateDisplay() {
    processImages();
    displayResults();
}

void EdgeDetectorUI::run() {
    createWindows();
    createTrackbars();
    updateDisplay();

    std::cout << "Press 'q' to exit" << std::endl;

    while (true) {
        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27)
            break;
    }
}
#include "edge_detector_ui.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#define LENA "Lena_24bits.bmp"
#define KIDS "kids.bmp"
#define FLOWERS "flowers_24bits.bmp"

int main(int argc, char** argv) {
    try {
        fs::path execPath = fs::current_path();
        fs::path sourcePath = execPath.parent_path();

        if (fs::path imagePath = sourcePath / "images" / KIDS; fs::exists(imagePath)) {
            EdgeDetectorUI ui(imagePath.string());
            ui.run();
        } else {
            std::cout << "Image not found at: " << imagePath << std::endl;
            EdgeDetectorUI ui;
            ui.run();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
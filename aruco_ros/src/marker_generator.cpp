#include <aruco/arucofidmarkers.h>
#include <string>
#include <sstream>

int main(int argc, char **argv) {
  std::string directory = "/home/jake/aruco/";
  for(int i = 0; i < 100; i++) {
    std::stringstream spath;
    spath << directory << i << ".png";
    std::string path = spath.str();
    cv::Mat image = aruco::FiducidalMarkers::createMarkerImage(i, 100, 3);
    cv::imwrite(path, image);
  }
}


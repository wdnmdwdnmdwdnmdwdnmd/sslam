#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    Mat img = Mat::zeros(400, 400, CV_8UC3);
    circle(img, Point(200, 200), 100, Scalar(0, 255, 0), -1);
    putText(img, "OpenCV OK!", Point(50, 210), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

    imshow("Test", img);
    waitKey(0);
    return 0;
}

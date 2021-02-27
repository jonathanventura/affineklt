
#include <affineklt.h>

#include <opencv2/highgui.hpp>

using namespace affineklt;

int main(int argc, char **argv )
{
    cv::Mat im1 = cv::imread("../data/rafiki.png",0);
    std::vector<cv::Mat> pyramid;
    buildPyramid( im1, 4, pyramid );
    cv::imwrite("pyramid0.png",pyramid[0]);
    cv::imwrite("pyramid1.png",pyramid[1]);
    cv::imwrite("pyramid2.png",pyramid[2]);
    cv::imwrite("pyramid3.png",pyramid[3]);
}


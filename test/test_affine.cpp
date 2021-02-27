
#include <affineklt.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace affineklt;

int main(int argc, char **argv )
{
    cv::Size patchSize(256,256);
    cv::Mat im = cv::imread("../data/rafiki.png",0);
    cv::Mat im0, im1;
    cv::getRectSubPix(im,patchSize,cv::Point2f(320,240),im0);
    cv::getRectSubPix(im,patchSize,cv::Point2f(340,260),im1);
    cv::imwrite("im0.png",im0);
    cv::imwrite("im1.png",im1);
    AffineKLTParameters params;
    params.nlevels = 4;
    params.windowSize = 31;
    params.resolutionThresh = 0.001;
    AffineKLT klt( params );
    cv::Point2f pt0(im0.cols/2,im0.rows/2);
    std::vector<cv::Point2f> keypoints0;
    keypoints0.push_back(pt0);
    std::vector<cv::Point2f> keypoints1;
    std::vector<cv::Matx22f> affine;
    klt.track( im0, keypoints0, im1, keypoints1, affine );
    std::cout << keypoints1[0] << "\n";
}


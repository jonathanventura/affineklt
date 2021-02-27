#pragma once

#include <opencv2/core.hpp>

namespace affineklt {

struct AffineKLTParameters
{
    int nlevels;
    int windowSize;
    int maxIter;
    float resolutionThresh;

    AffineKLTParameters() :
        nlevels(4),
        windowSize(11),
        maxIter(100),
        resolutionThresh(1.f)
    { }
};

void buildPyramid( const cv::Mat &image, const int nlevels, std::vector<cv::Mat> &pyramid );

class AffineKLT
{
protected:
    AffineKLTParameters params;
    void refine_tracks( const cv::Mat &image0, const std::vector<cv::Point2f> &points0,
                        const cv::Mat &image1, std::vector<cv::Point2f> &points1,
                        std::vector<cv::Matx22f> &affine );

public:
    AffineKLT( const AffineKLTParameters &_params );
    void track( const cv::Mat &image0, const std::vector<cv::Point2f> &keypoints0,
                const cv::Mat &image1, std::vector<cv::Point2f> &keypoints1,
                std::vector<cv::Matx22f> &affine );
};

}

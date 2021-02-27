
#include <affineklt.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>

#include <iostream>
using namespace affineklt;

int main(int argc, char **argv )
{
    cv::Size patchSize(256,256);
    cv::Mat im = cv::imread("../data/rafiki.png",0);
    cv::Mat im0, im1;
    cv::getRectSubPix(im,patchSize,cv::Point2f(320,240),im0);
    cv::Matx<float,2,3> M;
    M(0,0) = 1.01;
    M(0,1) = 0.01;
    M(1,0) = -0.01;
    M(1,1) = 1.01;
    M(0,2) = 325 - patchSize.width/2;
    M(1,2) = 245 - patchSize.height/2;
    cv::warpAffine(im,im1,M,patchSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP);
    cv::imwrite("im0.png",im0);
    cv::imwrite("im1.png",im1);
    AffineKLTParameters params;
    params.nlevels = 4;
    params.windowSize = 11;
    params.resolutionThresh = 0.001;
    AffineKLT klt( params );

    std::vector<cv::Point2f> points0;
    cv::goodFeaturesToTrack(im0,points0,100,0.01,2);
    std::vector<cv::Point2f> points1;
    std::vector<cv::Matx22f> affine;
    klt.track( im0, points0, im1, points1, affine );
    
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    cv::KeyPoint::convert(points0,keypoints0);
    cv::KeyPoint::convert(points1,keypoints1);
    
    cv::Mat im0out, im1out;
    cv::drawKeypoints(im0,keypoints0,im0out);
    cv::drawKeypoints(im1,keypoints1,im1out);
    cv::imwrite("im0out.png",im0out);
    cv::imwrite("im1out.png",im1out);
    
    std::vector<cv::DMatch> matches;
    for ( int i = 0; i < keypoints0.size(); i++ ) matches.push_back( cv::DMatch(i,i,0));
    cv::Mat matchesim;
    cv::drawMatches( im0, keypoints0, im1, keypoints1, matches, matchesim );
    cv::imwrite("matches.png",matchesim);
    
    for ( int i = 0; i < keypoints0.size(); i++ )
    {
        std::cout << "myA:\n" << affine[i].inv() << "\n";
    }
}


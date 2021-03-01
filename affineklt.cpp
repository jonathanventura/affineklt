
#include "affineklt.h"

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>
#include <iostream>

namespace affineklt {
    
    AffineKLT::AffineKLT( const AffineKLTParameters &_params ) : params(_params)
    {
        // pre-compute window weights
        windowSize = cv::Size(params.windowSize,params.windowSize);
        half_size = (params.windowSize-1)/2;
        weightsx = cv::Mat(windowSize,CV_32F);
        for ( int i = 0; i < params.windowSize; i++ )
            for ( int j = 0; j < params.windowSize; j++ )
                weightsx.at<float>(i,j) = j-half_size;
        weightsy = cv::Mat(windowSize,CV_32F);
        for ( int i = 0; i < params.windowSize; i++ )
            for ( int j = 0; j < params.windowSize; j++ )
                weightsy.at<float>(i,j) = i-half_size;
        weightsx2 = weightsx.mul(weightsx);
        weightsxy = weightsx.mul(weightsy);
        weightsy2 = weightsy.mul(weightsy);
    }
    
    void buildPyramid( const cv::Mat &image, const int nlevels, std::vector<cv::Mat> &pyramid )
    {
        pyramid.resize(nlevels);
        image.convertTo(pyramid[0],CV_32FC1, 1.0/255.0);
        for ( int i = 1; i < nlevels; i++ ) cv::pyrDown(pyramid[i-1],pyramid[i]);
    }

    void AffineKLT::track( const cv::Mat &image0, const std::vector<cv::Point2f> &keypoints0,
                const cv::Mat &image1, std::vector<cv::Point2f> &keypoints1,
                std::vector<cv::Matx22f> &affine )
    {
        // build pyramid
        std::vector<cv::Mat> pyramid0;
        std::vector<cv::Mat> pyramid1;
        buildPyramid(image0,params.nlevels,pyramid0);
        buildPyramid(image1,params.nlevels,pyramid1);
        
        // initialize transformations
        std::vector<cv::Matx21f> points1(keypoints0.size());
        affine.resize(keypoints0.size());
        for ( int i = 0; i < keypoints0.size(); i++ )
        {
            affine[i](0,0) = 1;
            affine[i](0,1) = 0;
            affine[i](1,0) = 0;
            affine[i](1,1) = 1;
        }

        // initialize keypoints1
        keypoints1.resize(keypoints0.size());
        float scale = pow(2.f,-(params.nlevels-1));
        for ( int i = 0; i < keypoints0.size(); i++ )
        {
            keypoints1[i] = keypoints0[i]*scale;
        }

        // iterate over pyramid
        std::vector<cv::Point2f> level_points0(keypoints0.size());
        for ( int l = params.nlevels-1; l >= 0; l-- )
        {
            float scale = pow(2.f,-l);
            for ( int i = 0; i < keypoints0.size(); i++ )
            {
                level_points0[i].x = keypoints0[i].x*scale;
                level_points0[i].y = keypoints0[i].y*scale;
            }
        
            refine_tracks( pyramid0[l], level_points0,
                           pyramid1[l], keypoints1, affine );
            
            if ( l == 0 ) break;

            // move up pyramid
            for ( int i = 0; i < keypoints0.size(); i++ )
            {
                keypoints1[i] *= 2;
            }
        }
    }

    void AffineKLT::refine_tracks( const cv::Mat &image0, const std::vector<cv::Point2f> &points0,
                        const cv::Mat &image1, std::vector<cv::Point2f> &points1,
                        std::vector<cv::Matx22f> &affine )
    {
        // compute image derivatives
        cv::Mat Ix,Iy;
        cv::Scharr(image0,Ix,-1,1,0);
        cv::Scharr(image0,Iy,-1,0,1);
        const cv::Mat Ix2 = Ix.mul(Ix);
        const cv::Mat IxIy = Ix.mul(Iy);
        const cv::Mat Iy2 = Iy.mul(Iy);
        
        cv::parallel_for_(cv::Range(0, points0.size()), [&](const cv::Range& range){
        for ( int i = range.start; i < range.end; i++ )
        {
            // get windows
            cv::Mat I, wIx, wIy, wIx2, wIxIy, wIy2;
            cv::getRectSubPix(image0,windowSize,points0[i],I,CV_32F);
            cv::getRectSubPix(Ix,windowSize,points0[i],wIx,CV_32F);
            cv::getRectSubPix(Iy,windowSize,points0[i],wIy,CV_32F);
            cv::getRectSubPix(Ix2,windowSize,points0[i],wIx2,CV_32F);
            cv::getRectSubPix(IxIy,windowSize,points0[i],wIxIy,CV_32F);
            cv::getRectSubPix(Iy2,windowSize,points0[i],wIy2,CV_32F);

            // spatial gradient matrix
            cv::Matx<float,6,6> G;
            G(0,0) = cv::sum(wIx2)[0];
            G(0,1) = cv::sum(wIxIy)[0];
            G(0,2) = cv::sum(weightsx.mul(wIx2))[0];
            G(0,3) = cv::sum(weightsy.mul(wIx2))[0];
            G(0,4) = cv::sum(weightsx.mul(wIxIy))[0];
            G(0,5) = cv::sum(weightsy.mul(wIxIy))[0];
    
            G(1,0) = G(1,0);
            G(1,1) = cv::sum(wIy2)[0];
            G(1,2) = cv::sum(weightsx.mul(wIxIy))[0];
            G(1,3) = cv::sum(weightsy.mul(wIxIy))[0];
            G(1,4) = cv::sum(weightsx.mul(wIy2))[0];
            G(1,5) = cv::sum(weightsy.mul(wIy2))[0];

            G(2,0) = G(0,2);
            G(2,1) = G(1,2);
            G(2,2) = cv::sum(weightsx2.mul(wIx2))[0];
            G(2,3) = cv::sum(weightsxy.mul(wIx2))[0];
            G(2,4) = cv::sum(weightsx2.mul(wIxIy))[0];
            G(2,5) = cv::sum(weightsxy.mul(wIxIy))[0];

            G(3,0) = G(0,3);
            G(3,1) = G(1,3);
            G(3,2) = G(2,3);
            G(3,3) = cv::sum(weightsy2.mul(wIx2))[0];
            G(3,4) = cv::sum(weightsxy.mul(wIxIy))[0];
            G(3,5) = cv::sum(weightsy2.mul(wIxIy))[0];

            G(4,0) = G(0,4);
            G(4,1) = G(1,4);
            G(4,2) = G(2,4);
            G(4,3) = G(3,4);
            G(4,4) = cv::sum(weightsx2.mul(wIy2))[0];
            G(4,5) = cv::sum(weightsxy.mul(wIy2))[0];

            G(5,0) = G(0,5);
            G(5,1) = G(1,5);
            G(5,2) = G(2,5);
            G(5,3) = G(3,5);
            G(5,4) = G(4,5);
            G(5,5) = cv::sum(weightsy2.mul(wIy2))[0];

            cv::SVD svdG(G);
        
            for ( int iter = 0; iter < params.maxIter; iter++ )
            {
                // warping of second image
                cv::Matx<float,2,3> M;
                M(0,0) = affine[i](0,0);
                M(0,1) = affine[i](0,1);
                M(0,2) = points1[i].x-half_size;
                M(1,0) = affine[i](1,0);
                M(1,1) = affine[i](1,1);
                M(1,2) = points1[i].y-half_size;
                cv::Mat J;
                cv::warpAffine(image1,J,M,windowSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
                
                // calculate difference
                cv::Mat diff;
                cv::subtract(I,J,diff,cv::noArray(),CV_32F);
                
                // calculate image mismatch vector
                cv::Matx<float,6,1> b;
                b(0,0) = cv::sum(wIx.mul(diff))[0];
                b(1,0) = cv::sum(wIy.mul(diff))[0];
                b(2,0) = cv::sum(weightsx.mul(wIx.mul(diff)))[0];
                b(3,0) = cv::sum(weightsy.mul(wIx.mul(diff)))[0];
                b(4,0) = cv::sum(weightsx.mul(wIy.mul(diff)))[0];
                b(5,0) = cv::sum(weightsy.mul(wIy.mul(diff)))[0];
                
                // calculate inv(G)*b
                cv::Matx<float,6,1> update;
                svdG.backSubst(b,update);
                
                // update point
                points1[i].x += affine[i](0,0) * update(0,0) + affine[i](0,1) * update(1,0);
                points1[i].y += affine[i](1,0) * update(0,0) + affine[i](1,1) * update(1,0);

                // update affine
                cv::Matx<float,2,2> Aupdate;
                Aupdate(0,0) = 1+update(2,0);
                Aupdate(0,1) = update(3,0);
                Aupdate(1,0) = update(4,0);
                Aupdate(1,1) = 1+update(5,0);
                affine[i] = affine[i] * Aupdate;
                
                // check convergence
                if ( update(0,0)*update(0,0)+update(1,0)*update(1,0) < params.resolutionThresh*params.resolutionThresh ) break;
            }
        }
        }); // end of parallel for
    }

}

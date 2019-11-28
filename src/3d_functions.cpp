
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <thread>


Mat read_extrinsics()
{

}

Mat read_intrinsics()
{

}


vector<Mat> cam_pose_to_origin(Size boardSize, float squareSize, Mat K, Mat D)
{
    vector<Point2f> imgPts;
    vector<Point3f> objPts;


    vector<Mat> vecs;
    Mat rvec;
    Mat tvec;

    bool found = false;

    Mat groundpic = imread("groundpattern.jpg");
    Mat gray;
    cvtColor(groundpic, gray, COLOR_BGR2GRAY);
   
    found = cv::findChessboardCorners(gray, boardSize, imgPts,
                            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found)
    {
        cornerSubPix(gray, imgPts, cv::Size(5, 5), cv::Size(-1, -1),
            TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1));
    }

    for( int j = 0; j < boardSize.height; j++ )
            for( int k = 0; k < boardSize.width; k++ )
                objPts.push_back(Point3f(j*squareSize, k*squareSize, 0));
    
    
    

    solvePnP(objPts, imgPts, K, D, rvec, tvec);
    
    
    Mat newvec;
    Mat newvec_t;

    _Float64 x_offset = 0;
    _Float64 y_offset = 0;
    _Float64 z_offset = 0;

    Rodrigues(rvec, newvec);
   
    newvec.copyTo(newvec_t);
    
    Mat column = (newvec_t.col(2));
    column.copyTo(newvec.col(1));

    column = (newvec_t.col(1));
    column.copyTo(newvec.col(0));

    column = (0-newvec_t.col(0));
    column.copyTo(newvec.col(2));

    
    newvec = newvec.inv();
    
    
    Rodrigues(newvec, rvec);

    vecs.push_back(rvec);
    vecs.push_back(tvec);

    return vecs;

}
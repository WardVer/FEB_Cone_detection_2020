#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

#include "yolo_v2_class.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <thread>
#include <algorithm>

class projector3D
{
public:
    projector3D(cv::Size board_size, float square_size)
    {
        /*  
        *read the different parameters of our cameras
        * extrinsics is a combination of the relative rotation and translation
        * of the 2 cameras.
        * intrinsics describe how a camera projects 3D points to the 2D image plane.
        * 
        * more info in their respective read functions
        */

        read_extrinsics();
        read_intrinsics();


        hconcat(rotation_relative, translation_relative, P2);
        P2 = K2 * P2;

        P1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
        P1 = K1 * P1;

        boardSize = board_size;
        squareSize = square_size;

        cam_pose_to_origin();
        
    }

    void stereo3D(cv::Mat &frame1, cv::Mat &frame2, std::vector<bbox_t> *result_vec1, std::vector<bbox_t> *result_vec2, std::vector<cv::Point3f> *points3D, std::vector<cv::Point2f> *offsets);

private:
    cv::Mat rotation_relative;
    cv::Mat translation_relative;

    cv::Mat K1;
    cv::Mat D1;
    cv::Mat K2;
    cv::Mat D2;

    cv::Mat P1;
    cv::Mat P2;

    cv::Mat rotation_ground;
    cv::Mat translation_ground;

    cv::Size boardSize;
    float squareSize;

    int read_extrinsics();

    int read_intrinsics();

    void cam_pose_to_origin();

    cv::Mat estimate_3d(int cone_x, int cone_y);

    cv::Mat estimate_2d(cv::Mat &rough_3d);

    std::vector<bbox_t> new_boxes(const std::vector<bbox_t> &result_vec, std::vector<cv::Mat> &rough_2d_vec);

    cv::Mat draw_features(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &corners1, std::vector<cv::Point2f> &corners2);

    void cone_offset(const std::vector<bbox_t> *result_vec1, const std::vector<bbox_t> *result_vec2, cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> *offsets);

    void cone_positions(const std::vector<bbox_t> *result_vec1, std::vector<cv::Point3f> *points3D, std::vector<cv::Point2f> *offsets);
};
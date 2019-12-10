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


int read_extrinsics(cv::Mat *R, cv::Mat *T);

int read_intrinsics(cv::Mat *K1, cv::Mat *D1, cv::Mat *K2, cv::Mat *D2);

std::vector<cv::Mat> cam_pose_to_origin(cv::Size boardSize, float squareSize, cv::Mat & K, cv::Mat & D);

cv::Mat estimate_3d(int cone_x, int cone_y, cv::Mat & K, cv::Mat & ground_r, cv::Mat & ground_t);

cv::Mat estimate_2d(cv::Mat & rough_3d, cv::Mat & R, cv::Mat & T, cv::Mat & K2, cv::Mat & ground_R, cv::Mat & ground_t);

std::vector<bbox_t> new_boxes(const std::vector<bbox_t> * result_vec, std::vector<cv::Mat> & rough_2d_vec);

cv::Mat draw_features(cv::Mat & img1,cv::Mat & img2, std::vector<cv::Point2f> & corners1, std::vector<cv::Point2f> & corners2);

std::vector<cv::Point> cone_offset(const std::vector<bbox_t> * result_vec1, const std::vector<bbox_t> * result_vec2, cv::Mat & img1, cv::Mat & img2);

std::vector<cv::Point3d> cone_positions(const std::vector<bbox_t> * result_vec1, std::vector<cv::Point> & offsets, cv::Mat & P1, cv::Mat & P2, cv::Mat & ground_R, cv::Mat & ground_t);

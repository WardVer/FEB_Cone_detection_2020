#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/sfm/projection.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <thread>


int read_extrinsics(cv::Mat *R, cv::Mat *T);

int read_intrinsics(cv::Mat *K1, cv::Mat *D1, cv::Mat *K2, cv::Mat *D2);

std::vector<cv::Mat> cam_pose_to_origin(cv::Size boardSize, float squareSize, cv::Mat K, cv::Mat D);

cv::Mat estimate_3d_world(int cone_x, int cone_y, cv::Mat K, cv::Mat ground_r, cv::Mat ground_t);
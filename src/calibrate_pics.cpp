
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "GxIAPI.h"
#include "DxImageProc.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <thread>

using namespace cv;
using namespace std;

Mat frame1;
Mat frame2;

int framerate = 90;
int multiplier = 90;
int pics = 40;

void show_msg_in_feed(Mat frame, string message, int frame_ID)
{
  flip(frame, frame, 1);
  putText(frame, to_string(int(frame_ID / framerate)), Point(50, 50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 100), 2);
  putText(frame, message, Point(50, 100), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 100), 2);
  resize(frame, frame, Size(640, 512));
  imshow("yeet", frame);
  waitKey(1);
}

void show_msg_in_double_feed(VideoCapture cap1, VideoCapture cap2, string message)
{
  for (int i = 0; i < 80; i++)
  {

    Mat frame1;
    Mat frame2;
    Mat concatframe;

    cap1 >> frame1;
    cap2 >> frame2;

    hconcat(frame1, frame2, concatframe);
    flip(concatframe, concatframe, 1);
    resize(concatframe, concatframe, Size(1280, 512));
    putText(concatframe, message, Point(50, 100), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
  }
}

void extrinsics_pics(String location, Mat frame1, Mat frame2, FileStorage fs, int frame_ID)
{
  Mat concatframe;
  std::string picfilename;

 
    if (frame_ID % multiplier == 0)
    {
      std::string nr = to_string(frame_ID * 2 / multiplier);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      string fullpath = location + picfilename;
      printf("%s\n", fullpath.c_str());
      imwrite(fullpath, frame1);
      fs << (fullpath).c_str();

      nr = to_string(frame_ID * 2 / multiplier + 1);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      fullpath = location + picfilename;
      imwrite(fullpath, frame2);
      fs << (fullpath).c_str();
    }

    hconcat(frame1, frame2, concatframe);
    cv::flip(concatframe, concatframe, 1);
    resize(concatframe, concatframe, Size(1280, 512));
    putText(concatframe, to_string(frame_ID / multiplier), Point(50, 50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
  
}

void cam_pics(String location, Mat frame, FileStorage fs, int frame_ID)
{
  string picfilename;

  if (frame_ID % multiplier == 0)
  {
    string nr = to_string(int(frame_ID / multiplier));
    picfilename = "pic" + string(5 - nr.length(), '0') + nr + ".jpg";
    string fullpath = location + picfilename;
    imwrite(fullpath, frame);
    fs << (fullpath).c_str();
  }

  cv::flip(frame, frame, 1);
  cv::putText(frame, to_string(frame_ID / multiplier), Point(50, 50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0, 255, 100), 2);
  resize(frame, frame, Size(640, 512));
  cv::imshow("image", frame);
  cv::waitKey(1);

}


int main(int argc, char **argv)
{
  GX_STATUS status = GX_STATUS_SUCCESS;
  GX_DEV_HANDLE hDevice = NULL;
  uint32_t nDeviceNum = 0;
  uint32_t frame_ID = 0;
  FileStorage fs;
  int pausetime = 5;

  if (0 == strcmp(argv[1], "extr"))
  {
    fs = FileStorage("pairlist.xml", FileStorage::WRITE);
    fs << "strings"
       << "[";
  }
  else if (0 == strcmp(argv[1], "cam1"))
  {
    fs = FileStorage("imglist1.xml", FileStorage::WRITE);
    fs << "strings"
       << "[";
  }
  else if (0 == strcmp(argv[1], "cam2"))
  {
    fs = FileStorage("imglist2.xml", FileStorage::WRITE);
    fs << "strings"
       << "[";
  }

  std::string picpath = "./calibrationpics/";

  // Initializes the library.
  status = GXInitLib();

  if (status != GX_STATUS_SUCCESS)
  {
    return 0;
  }

  // Updates the enumeration list for the devices.
  status = GXUpdateDeviceList(&nDeviceNum, 1000);
  if ((status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0))
  {
    return 0;
  }

  // Opens the device.
  status = GXOpenDeviceByIndex(1, &hDevice);
  if (status == GX_STATUS_SUCCESS)
  {
    status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, 10000);
    status = GXSetFloat(hDevice, GX_FLOAT_GAIN, 16);
    status = GXSetEnum(hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_RED);
    status = GXSetFloat(hDevice, GX_FLOAT_BALANCE_RATIO, 1.4);
    status = GXSetEnum(hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_BLUE);
    status = GXSetFloat(hDevice, GX_FLOAT_BALANCE_RATIO, 1.5);
    // Define the incoming parameters of GXDQBuf.
    PGX_FRAME_BUFFER pFrameBuffer;
    // Stream On.
    status = GXStreamOn(hDevice);
    if (status == GX_STATUS_SUCCESS)
    {
      while(1)
      {
        // Calls GXDQBuf to get a frame of image.
        status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
        if (status == GX_STATUS_SUCCESS)
        {
          if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
          {

            cv::Mat image;
            image.create(pFrameBuffer->nHeight, pFrameBuffer->nWidth, CV_8UC1);
            VxInt32 DxStatus = DxBrightness((void *)pFrameBuffer->pImgBuf, (void *)pFrameBuffer->pImgBuf, pFrameBuffer->nWidth * pFrameBuffer->nHeight, 50);
            memcpy(image.data, pFrameBuffer->pImgBuf, pFrameBuffer->nWidth * pFrameBuffer->nHeight);
            cv::cvtColor(image, image, cv::COLOR_BayerRG2RGB);
            cv::flip(image, image, 0);
            cv::flip(image, image, 1);
            frame1 = image(cv::Rect(0, 0, 1280, 1024));
            frame2 = image(cv::Rect(1280, 0, 1280, 1024));

            

            if (frame_ID < framerate*pausetime)
            {
              if (0 == strcmp(argv[1], "extr"))
              {
                show_msg_in_feed(image, "extrinsics", frame_ID);
              }
              else if (0 == strcmp(argv[1], "cam1"))
              {
                show_msg_in_feed(frame1, "intrinsics cam1", frame_ID);
              }
              else if (0 == strcmp(argv[1], "cam2"))
              {
                show_msg_in_feed(frame2, "intrinsics cam2", frame_ID);
              }
              else if (0 == strcmp(argv[1], "ground"))
              {
                show_msg_in_feed(frame1, "ground", frame_ID);
              }
            }
            else if (frame_ID >= framerate*pausetime)
            {
              
              int new_frame_ID = frame_ID - framerate*pausetime;

              if (0 == strcmp(argv[1], "extr"))
              {
                extrinsics_pics(picpath + "extr/", frame1, frame2, fs, new_frame_ID);
              }
              else if (0 == strcmp(argv[1], "cam1"))
              {
                cam_pics(picpath + "cam1/", frame1, fs, new_frame_ID);
              }
              else if (0 == strcmp(argv[1], "cam2"))
              {
                cam_pics(picpath + "cam2/", frame2, fs, new_frame_ID);
              }
              else if (0 == strcmp(argv[1], "ground"))
              {
                imwrite("groundpattern.jpg", frame1);
                break;
              }
            }
            cout << frame_ID << endl;
            frame_ID++;
          }
          // Calls GXQBuf to put the image buffer back into the library
          //and continue acquiring.
          status = GXQBuf(hDevice, pFrameBuffer);
          if(frame_ID == framerate * pausetime + multiplier*pics)
          {
            break;
          }
          
        }
      }
    }
  }
}
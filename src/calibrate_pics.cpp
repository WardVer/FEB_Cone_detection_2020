
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

using namespace cv;
using namespace std;

using namespace cv;

void show_msg_in_feed(VideoCapture cap, string message)
{
  for(int i = 0; i < 120; i++)
  {

    Mat frame;
    cap >> frame; 
    flip(frame, frame, 1);
    putText(frame, to_string(i/30), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    putText(frame, message, Point(50,100), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", frame);
    waitKey(1);
  }
}

void show_msg_in_double_feed(VideoCapture cap1, VideoCapture cap2, string message)
{
  for(int i = 0; i < 80; i++)
  {

    Mat frame1;
    Mat frame2;
    Mat concatframe;

    cap1 >> frame1; 
    cap2 >> frame2;
 
    hconcat(frame1, frame2, concatframe);
    flip(concatframe, concatframe, 1);
    resize(concatframe, concatframe, Size(1280,360));
    putText(concatframe, message, Point(50,100), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
    

  }
}

void extrinsics_pics(String location, VideoCapture cap1, VideoCapture cap2)
{
  std::thread t_cap1;
  std::thread t_cap2;

  Mat frame1;
  Mat frame2;

  Mat concatframe;

  std::string picfilename;

  int multiplier = 20;
  int pics = 20;

  FileStorage fs("pairlist.xml", FileStorage::WRITE);
  fs << "strings" << "["; 
  show_msg_in_double_feed(cap1, cap2, "calibrating extrinsics");
  
  for(int i = 0; i < pics*multiplier; i++)
  {

    

    t_cap2 = std::thread([&]()
    {
      cap2 >> frame2; 
    });

    t_cap1 = std::thread([&]()
    {
      cap1 >> frame1; 
    });

    

    t_cap1.join();
    t_cap2.join();

    if(i%multiplier == 0){
      std::string nr = to_string(i*2/multiplier);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      string fullpath = location + picfilename;
      printf("%s\n", fullpath.c_str());
      imwrite(fullpath, frame1);
      fs << (fullpath).c_str();


      nr = to_string(i*2/multiplier+1);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      fullpath = location + picfilename;
      imwrite(fullpath, frame2);
      fs << (fullpath).c_str();
    }


    hconcat(frame1, frame2, concatframe);
    flip(concatframe, concatframe, 1);
    resize(concatframe, concatframe, Size(1280,360));
    putText(concatframe, to_string(i/multiplier), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
    

  }

  fs << "]";
  fs.release();

}

void cam1_pics(String location, VideoCapture cap)
{

 
  
  Mat frame;
  std::string picfilename;

  int multiplier = 20;
  int pics = 30;

  FileStorage fs1("imglist1.xml", FileStorage::WRITE);
  fs1 << "strings" << "["; 

  show_msg_in_feed(cap, "calibrating camera 1");

  for(int i = 0; i < pics*multiplier; i++)
  {

    cap >> frame; 
   
    if(i%multiplier == 0){
      std::string nr = to_string(i/multiplier);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      string fullpath = location + picfilename;
      imwrite(fullpath, frame);
      fs1 << (fullpath).c_str();
    }
    flip(frame,frame,1);
    putText(frame, to_string(i/multiplier), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", frame);
    waitKey(1);
    

  }

  fs1 << "]";
  fs1.release();
}

void cam2_pics(String location, VideoCapture cap)
{
  Mat frame;
  std::string picfilename;

  int multiplier = 20;
  int pics = 30;

  FileStorage fs2("imglist2.xml", FileStorage::WRITE);
  fs2 << "strings" << "[";  

  show_msg_in_feed(cap, "calibrating camera 2");

  for(int i = 0; i < pics*multiplier; i++)
  {

    
    cap >> frame; 
   
    if(i%multiplier == 0){
      std::string nr = to_string(i/multiplier);
      picfilename = "pic" + std::string(5 - nr.length(), '0') + nr + ".jpg";
      
      string fullpath = location + picfilename;
      imwrite(fullpath, frame);
      fs2 << (fullpath).c_str();
    }
    flip(frame,frame,1);
    putText(frame, to_string(i/multiplier), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", frame);
    waitKey(1);
    

  }

  fs2 << "]";
  fs2.release();
  
}

void ground_pic(VideoCapture cap)
{
  show_msg_in_feed(cap, "ground pattern");

  Mat frame;
  cap >> frame; 
  imwrite("groundpattern.jpg", frame);
  
}



int main(int argc, char** argv)
{

  

  VideoCapture cap1;
  VideoCapture cap2;

  cap1.open(0);
  cap2.open(2);

  cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
  cap1.set(CAP_PROP_FRAME_WIDTH, 640);
  cap2.set(CAP_PROP_FRAME_HEIGHT, 480);
  cap2.set(CAP_PROP_FRAME_WIDTH, 640);

  std::string picpath = "./calibrationpics/";
  std::string picext = ".jpg";

  

  

  if (0 == strcmp(argv[1], "extr")){
        extrinsics_pics(picpath + "extr/", cap1, cap2);
    } else if (0 == strcmp(argv[1], "cam1")){
        cam1_pics(picpath + "cam1/", cap1);
    } else if (0 == strcmp(argv[1], "cam2")){
        cam2_pics(picpath + "cam2/", cap2);
    } else if (0 == strcmp(argv[1], "ground")){
        ground_pic(cap1);
    }


  return 0;
}
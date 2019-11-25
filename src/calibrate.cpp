/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.
   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008
   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130
   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     GitHub:        https://github.com/opencv/opencv/
   ************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

static int print_help()
{
    cout <<
            " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use stereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n" << endl;
    return 0;
}





static void
StereoCalib(const vector<string>& pairImagelist, const vector<string>& imagelist1, const vector<string>& imagelist2, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true)
{

    
    if( pairImagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 1;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > pairImagePoints[2];
    vector<vector<Point3f> > pairObjectPoints;
    vector< Point2f > corners1, corners2;
    vector< vector< Point2f > > left_img_points, right_img_points;

    vector<vector<Point2f> > intrImagePoints1;
    vector<vector<Point2f> > intrImagePoints2;
    vector<vector<Point3f> > intrObjectPoints1;
    vector<vector<Point3f> > intrObjectPoints2;
    

    Size imageSize;

    int i, j, k = 0;
    int nPairImages = (int)pairImagelist.size()/2;
    int nImages1 = (int)imagelist1.size();
    int nImages2 = (int)imagelist2.size();

    intrImagePoints1.resize(nImages1);
    intrImagePoints2.resize(nImages2);
    vector<string> goodPairImageList;
    vector<string> goodImageList1;
    vector<string> goodImageList2;

    
    j = 0;
    for(i = 0; i < nImages1; i++)
    {
            const string& filename = imagelist1[i];
            
            Mat img = imread(filename, 0);
            imageSize = img.size();
            bool found = false;
            vector<Point2f>& corners = intrImagePoints1[j];
            
            found = cv::findChessboardCorners(img, boardSize, corners,
                                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
            if (found)
            {
                cornerSubPix(img, corners, cv::Size(5, 5), cv::Size(-1, -1),
                   TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1));
                drawChessboardCorners(img, boardSize, corners, found);
                j++;
            }
    }
    
    nImages1 = j;
    
    j = 0;
    for(i = 0; i < nImages2; i++)
    {
            
            const string& filename = imagelist2[i];
            Mat img = imread(filename, 0);
            bool found = false;
            vector<Point2f>& corners = intrImagePoints2[j];
            found = cv::findChessboardCorners(img, boardSize, corners,
                                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
            if (found)
            {
                cornerSubPix(img, corners, cv::Size(5, 5), cv::Size(-1, -1),
                   TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.1));
                drawChessboardCorners(img, boardSize, corners, found);
                j++;
            }
    }
    nImages2 = j;

    intrObjectPoints1.resize(nImages1);
    intrObjectPoints2.resize(nImages2);

    intrImagePoints1.resize(nImages1);
    intrImagePoints2.resize(nImages2);

    cout << intrImagePoints1.size() << endl;
    cout << intrObjectPoints1.size() << endl;
    cout << intrImagePoints2.size() << endl;
    cout << intrObjectPoints2.size() << endl;
    
    for( i = 0; i < nImages1; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                intrObjectPoints1[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }

    for( i = 0; i < nImages2; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                intrObjectPoints2[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }
    
    Mat K[2];
    Mat D[2];

    vector< Mat > rvecs1, tvecs1, rvecs2, tvecs2;
    calibrateCamera(intrObjectPoints1, intrImagePoints1, imageSize, K[0], D[0], rvecs1, tvecs1);
    calibrateCamera(intrObjectPoints2, intrImagePoints2, imageSize, K[1], D[1], rvecs2, tvecs2);

    const string& filename1 = imagelist1[3];
    Mat img = imread(filename1);
    Mat dst;
    undistort(img, dst, K[0], D[0]);
    
    cout << K[0];
    cout << K[1];

    cout << D[0];
    cout << D[1];

    /*
    while(1)
    {
        imshow("distorted", img);
        int key = waitKey(0);
        if(key == 27) break;
        imshow("distorted", dst);
        key = waitKey(0);
        if(key == 27) break;
    }

    const string& filename2 = imagelist2[3];
    img = imread(filename2);
    undistort(img, dst, K[1], D[1]);

    while(1)
    {
        imshow("distorted", img);
        int key = waitKey(0);
        if(key == 27) break;
        imshow("distorted", dst);
        key = waitKey(0);
        if(key == 27) break;
    }*/
    
    Mat img1;
    Mat img2;

    Mat gray1;
    Mat gray2;
    for( i = 0; i < nPairImages; i++)
    {
        img1 = imread(pairImagelist[2*i]);
        img2 = imread(pairImagelist[2*i+1]);

        cvtColor(img1, gray1, COLOR_BGR2GRAY);
        cvtColor(img2, gray2, COLOR_BGR2GRAY);

        bool found1 = false, found2 = false;

        found1 = cv::findChessboardCorners(gray1, boardSize, corners1);
        found2 = cv::findChessboardCorners(gray2, boardSize, corners2);

        if (found1)
        {
            cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
            cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
            cv::drawChessboardCorners(img1, boardSize, corners1, found1);
        }
        if (found2)
        {
            cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
            cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
            cv::drawChessboardCorners(img2, boardSize, corners2, found2);
        }

        vector< Point3f > obj;
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                obj.push_back(Point3f((float)i * squareSize, (float)j * squareSize, 0));

        if (found1 && found2) {

            /*
            imshow("yeet", img1);
            imshow("yoink", img2);
            waitKey(0);*/

            cout << endl << "corners1= " << corners1 << endl << endl;
            cout << endl << "corners2= " << corners2 << endl << endl;
            cout << i << ". Found corners!" << endl;
            pairImagePoints[0].push_back(corners1);
            pairImagePoints[1].push_back(corners2);
            pairObjectPoints.push_back(obj);

            goodPairImageList.push_back(pairImagelist[i*2]);
            goodPairImageList.push_back(pairImagelist[i*2+1]);
        }
    }
    nPairImages = pairObjectPoints.size();
    Mat R, F, E;
    Vec3d T;

    int flag = 0;
    flag |= CALIB_FIX_INTRINSIC;

    float rms = stereoCalibrate(pairObjectPoints, pairImagePoints[0], pairImagePoints[1], K[0], D[0], K[1], D[1], imageSize, R, T, E, F, CALIB_FIX_INTRINSIC, cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 300, 0.00000001));
    cout << "rms: " << rms << endl;

    const string& filename3 = imagelist1[3];
    img = imread(filename3);
    undistort(img, dst, K[0], D[0]);

    while(1)
    {
        imshow("distorted", img);
        int key = waitKey(0);
        if(key == 27) break;
        imshow("distorted", dst);
        key = waitKey(0);
        if(key == 27) break;
    }

    const string& filename4 = imagelist2[3];
    img = imread(filename4);
    undistort(img, dst, K[1], D[1]);

    while(1)
    {
        imshow("distorted", img);
        int key = waitKey(0);
        if(key == 27) break;
        imshow("distorted", dst);
        key = waitKey(0);
        if(key == 27) break;
    }
    /*

    for( i = j = 0; i < nPairImages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = pairImagelist[i*2+k];
            Mat img = imread(filename, 0);

            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }

            bool found = false;
            vector<Point2f>& corners = pairImagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                found = findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }

            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            goodPairImageList.push_back(pairImagelist[i*2]);
            goodPairImageList.push_back(pairImagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nPairImages = j;
    if( nPairImages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    pairImagePoints[0].resize(nPairImages);
    pairImagePoints[1].resize(nPairImages);
    pairObjectPoints.resize(nPairImages);

    for( i = 0; i < nPairImages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                pairObjectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat R, T, E, F;
    double rms = stereoCalibrate(pairObjectPoints, pairImagePoints[0], pairImagePoints[1], K[0], D[0], K[1], D[1], imageSize, R, T, E, F, CALIB_FIX_INTRINSIC);
    cout << "done with RMS error=" << rms << endl;
    */


    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(K[0], D[0],
                  K[1], D[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    FileStorage fs;
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;
    
    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nPairImages; i++ )
                std::copy(pairImagePoints[k][i].begin(), pairImagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = K[0].inv()*H1*K[0];
        R2 = K[1].inv()*H2*K[1];
        P1 = K[0];
        P2 = K[1];
    }
    
    //Precompute maps for cv::remap()
    initUndistortRectifyMap(K[0], D[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(K[1], D[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nPairImages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodPairImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
    
    
}


static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    
    Size boardSize;
    string imagelistpairs_t;
    string imagelistcam1_t;
    string imagelistcam2_t;
    bool showRectified;
    cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|stereo_calib.xml|}");
    if (parser.has("help"))
        return print_help();
    showRectified = true;
    imagelistpairs_t = "pairlist.xml";
    imagelistcam1_t = "imglist1.xml";
    imagelistcam2_t = "imglist2.xml";
    boardSize.width = 9;
    boardSize.height = 6;
    float squareSize = 3.78f;
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelistpairs;
    vector<string> imagelistcam1;
    vector<string> imagelistcam2;
    bool ok = readStringList(imagelistpairs_t, imagelistpairs);
    if(!ok || imagelistpairs.empty())
    {
        cout << "can not open " << imagelistpairs_t << " or the string list is empty" << endl;
        return print_help();
    }
    
    ok = readStringList(imagelistcam1_t, imagelistcam1);
    if(!ok || imagelistcam1.empty())
    {
        cout << "can not open " << imagelistcam1_t << " or the string list is empty" << endl;
        return print_help();
    }

    ok = readStringList(imagelistcam2_t, imagelistcam2);
    if(!ok || imagelistcam2.empty())
    {
        cout << "can not open " << imagelistcam2_t << " or the string list is empty" << endl;
        return print_help();
    }
    
    StereoCalib(imagelistpairs, imagelistcam1, imagelistcam2, boardSize, squareSize, false, true, showRectified);
    return 0;
}
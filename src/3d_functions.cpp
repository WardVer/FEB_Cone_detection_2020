#include "3d_functions.hpp"
#include <opencv2/sfm/projection.hpp>


using namespace cv;
using namespace std;


int read_extrinsics(cv::Mat *R, cv::Mat *T)
{

    FileStorage fs;
    fs.open("extrinsics.xml", FileStorage::READ);

    if (!fs.isOpened())
    {
            cerr << "Failed to open extrinsics" << endl;
            return 1;
    }

    FileNode node = fs["R"];
    node >> *R;

    node = fs["T"];
    node >> *T;

    fs.release();

    return 0;
}

int read_intrinsics(cv::Mat *K1, cv::Mat *D1, cv::Mat *K2, cv::Mat *D2)
{
    FileStorage fs;
    fs.open("intrinsics.xml", FileStorage::READ);

    if (!fs.isOpened())
    {
            cerr << "Failed to open extrinsics" << endl;
            return 1;
    }

    FileNode node = fs["K1"];
    node >> *K1;

    node = fs["D1"];
    node >> *D1;

    node = fs["K2"];
    node >> *K2;

    node = fs["D2"];
    node >> *D2;

    fs.release();

    return 0;

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

    Mat test = newvec * tvec;
    cout << test << endl;

    vecs.push_back(newvec);
    vecs.push_back(tvec);

    return vecs;

}


cv::Mat estimate_3d_world(int cone_x, int cone_y, cv::Mat K, cv::Mat ground_R, cv::Mat ground_t)
{
    
    
    cv::Mat uvPoint = (cv::Mat_<double>(3,1) << cone_x, cone_y, 1); 

    cv::Mat leftSideMat  = ground_R.inv() * K.inv() * uvPoint;
    cv::Mat rightSideMat = ground_R.inv() * ground_t;

    double s = rightSideMat.at<double>(1,0)/leftSideMat.at<double>(1,0); 

    Mat m_out = ground_R.inv() * (s * K.inv() * uvPoint - ground_t);
    std::cout << "P = " << m_out << std::endl;
    return m_out;
}


cv::Mat estimate_2d(cv::Mat rough_3d, cv::Mat R, cv::Mat T, cv::Mat K2, cv::Mat ground_R, cv::Mat ground_t)
{
    cv::Mat conepos_world_fromcamera = ground_R.inv()*ground_t + rough_3d;
    cout << conepos_world_fromcamera << endl;

    P = projectionFromKRt(K2)

    cv::Mat conepos_camera1 = ground_R*conepos_world_fromcamera;
    cv::Mat right_side = K2 * 

    
   
}


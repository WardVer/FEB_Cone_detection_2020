#include "3d_functions.hpp"



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

int read_intrinsics(Mat *K1, Mat *D1, Mat *K2, Mat *D2)
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


vector<Mat> cam_pose_to_origin(Size boardSize, float squareSize, Mat & K, Mat & D)
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
        cornerSubPix(gray, imgPts, Size(5, 5), Size(-1, -1),
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

    vecs.push_back(newvec);
    vecs.push_back(tvec);

    return vecs;

}


cv::Mat estimate_3d(int cone_x, int cone_y, Mat & K, Mat & ground_R, Mat & ground_t)
{
    
    
    Mat uvPoint = (Mat_<double>(3,1) << cone_x, cone_y, 1); 

    Mat leftSideMat  = ground_R.inv() * K.inv() * uvPoint;
    Mat rightSideMat = ground_R.inv() * ground_t;

    double s = rightSideMat.at<double>(1)/leftSideMat.at<double>(1); 

    Mat m_out = ground_R.inv() * (s * K.inv() * uvPoint - ground_t);
    //cout << "rough_3d = " << m_out << endl;
    return m_out;
}


cv::Mat estimate_2d(Mat & rough_3d, Mat & R, Mat & T, Mat & K2, Mat & ground_R, Mat & ground_t)
{
    //cout << "gt: " << ground_t << endl;
    //cout << "R: " << R << endl;
    //cout << "T: " << T << endl;


    Mat conepos_camera_fromcamera = ground_R * rough_3d + ground_t;

    Mat rough_3d_h;
    Mat one = Mat::ones(Size(1,1), CV_64FC1);
    vconcat(conepos_camera_fromcamera, one, rough_3d_h);
    //cout << "3d: " << rough_3d_h <<  endl;

    Mat P;
    hconcat(R,T,P);
    P = K2 * P;
    //cout << "P: " << P <<  endl;

    cv::Mat rough_2d = P * rough_3d_h;

   
    rough_2d /= rough_2d.at<double>(2);

    //cout << "2d: " << rough_2d <<  endl;
    
    return rough_2d;
}

std::vector<bbox_t> new_boxes(const std::vector<bbox_t> * result_vec, std::vector<Mat> & rough_2d_vec)
{
    std::vector<bbox_t> new_2d;
    
    //cout << "size " << result_vec.size() << endl;

    

    for (int a = 0; a < result_vec->size(); a++)
    {
        bbox_t result_copy = result_vec->at(a);
        result_copy.x = (int) rough_2d_vec[a].at<double>(0) - (int)(result_copy.w/2);
        result_copy.y = (int) rough_2d_vec[a].at<double>(1) - (int)(result_copy.h);
        new_2d.push_back(result_copy);
    }

    return new_2d;

}

Mat draw_features(Mat & img1, Mat & img2, vector<Point2f> & corners1, vector<Point2f> & corners2)
{
    Mat img1_copy;
    Mat img2_copy;

    img1.copyTo(img1_copy);
    img2.copyTo(img2_copy);


    for(int i = 0; i < corners1.size(); i++)
    {
        if(corners1[i].x >= 0 && corners1[i].y >= 0)
        circle(img1_copy, corners1[i], 2, Scalar(0,255,0));
    }

    for(int i = 0; i < corners2.size(); i++)
    {
        if(corners2[i].x >= 0 && corners2[i].y >= 0)
        circle(img2_copy, corners2[i], 2, Scalar(0,255,0));
    }
    


    Mat concatcone;

    hconcat(img1_copy, img2_copy, concatcone);

    return concatcone;

    


}

vector<Point2f> cone_offset(const vector<bbox_t> * result_vec1, const vector<bbox_t> * result_vec2, Mat & img1, Mat & img2)
{
    Mat img1ROI;
    Mat img2ROI;

    Mat gray1;
    Mat gray2;

    vector<Point2f> offsets;
    for (int a = 0; a < result_vec1->size(); a++)
    {
        /*
        int x1 = max((int)(result_vec1[a].x-result_vec1[a].w/4), 0);
        int y1 = max((int)(result_vec1[a].y-result_vec1[a].h/4), 0);
        int w1 = min((int)(result_vec1[a].w*1.5), img1.cols - (int)(result_vec1[a].x));
        int h1 = min((int)(result_vec1[a].h*1.5), img1.rows - (int)(result_vec1[a].y));

        int x2 = max((int)(result_vec2[a].x-result_vec2[a].w/4), 0);
        int y2 = max((int)(result_vec2[a].y-result_vec2[a].h/4), 0);
        int w2 = min((int)(result_vec2[a].w*1.5), img2.cols - (int)(result_vec2[a].x));
        int h2 = min((int)(result_vec2[a].h*1.5), img2.rows - (int)(result_vec2[a].y));
        */

       
        int x1 = max((int)result_vec1->at(a).x, 0);
        int y1 = max((int)result_vec1->at(a).y, 0);
        int w1 = min((int)(result_vec1->at(a).w), img1.cols - (int)(result_vec1->at(a).x));
        int h1 = min((int)(result_vec1->at(a).h), img1.rows - (int)(result_vec1->at(a).y));

        int x2 = max((int)(result_vec2->at(a).x), 0);
        int y2 = max((int)(result_vec2->at(a).y), 0);
        int w2 = min((int)(result_vec2->at(a).w), img2.cols - (int)(result_vec2->at(a).x));
        int h2 = min((int)(result_vec2->at(a).h), img2.rows - (int)(result_vec2->at(a).y));
        
        if(w2 <= 5)
        {
            offsets.push_back(Point(x2 - x1, y2 - y1));
            continue;
        }

        int w = min(w1, w2);
        int h = min(h1, h2);
        
        img1ROI = img1(Rect(x1, y1, w, h));
        img2ROI = img2(Rect(x2, y2, w, h));

        cvtColor(img1ROI, gray1, COLOR_BGR2GRAY);
        cvtColor(img2ROI, gray2, COLOR_BGR2GRAY);

        equalizeHist(gray1, gray1);
        equalizeHist(gray2, gray2);

        vector<Point2f> corners1;
        vector<Point2f> corners2;

        vector<uchar> status;
        vector<float> errors;

        std::chrono::steady_clock::time_point begin;
        std::chrono::steady_clock::time_point end;

        //begin = std::chrono::steady_clock::now();
        cv::goodFeaturesToTrack(gray1, corners1, 8, 0.1, 4);

        //status.resize(corners1.size());
        
       
        cv::calcOpticalFlowPyrLK(gray1, gray2, corners1, corners2, status, errors, Size(16,16));

        //end = std::chrono::steady_clock::now();
        //std::cout << "feature time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        //cout << corners1 << endl;

        //cout << corners2 << endl;

        float xsum = 0;
        float ysum = 0;
        int matches = 0;

        for (int i = 0; i < corners1.size(); i++) 
        {
            if(status[i]==1)
            {
                matches++;
                xsum += corners2[i].x - corners1[i].x;
                ysum += corners2[i].y - corners1[i].y;
            }
        }
        xsum /= matches;
        ysum /= matches;

        
        offsets.push_back(Point2f(xsum + (float)x2 - (float)x1, ysum + (float)y2 - (float)y1));

        Point testPoint((int)xsum, (int)ysum);
        Mat concatcone = draw_features(img1ROI, img2ROI, corners1, corners2);
        resize(concatcone, concatcone, Size(500,250));
       
        imshow("yeet", concatcone);

        

    }

    return offsets;

}

std::vector<cv::Point3f> cone_positions(const std::vector<bbox_t> * result_vec1, std::vector<cv::Point2f> & offsets, cv::Mat & P1, cv::Mat & P2, cv::Mat & ground_R, cv::Mat & ground_t)
{
    vector<Point2d> Points2D_1;
    vector<Point2d> Points2D_2;


    Mat points3D;

    for(int m = 0; m < result_vec1->size(); m++)
    {
        Points2D_1.push_back(Point2d(result_vec1->at(m).x + result_vec1->at(m).w/2, result_vec1->at(m).y + result_vec1->at(m).h/2));
        Points2D_2.push_back(Point2d(result_vec1->at(m).x + result_vec1->at(m).w/2 + offsets[m].x, result_vec1->at(m).y + result_vec1->at(m).h/2 + offsets[m].y));  
    }

    
    triangulatePoints(P1, P2, Points2D_1, Points2D_2, points3D);
    

    Mat points3D_2;
    Mat points3D_r = points3D.t();
    points3D_r = points3D_r.reshape(4);
    cv::convertPointsFromHomogeneous(points3D_r, points3D_2);
   
    
    Mat positions[3];
    vector<Point3f> final_3D_positions;

    split(points3D_2, positions); 

    for(int c = 0; c < points3D_2.rows; c++)
    {
        Mat new_3D_points = (Mat_<double>(3,1) << positions[0].at<double>(c), positions[1].at<double>(c), positions[2].at<double>(c));
        new_3D_points = ground_R.inv() * (new_3D_points);
        new_3D_points -= ground_R.inv() * (ground_t);
        final_3D_positions.push_back(Point3d(new_3D_points.at<double>(0), new_3D_points.at<double>(1), new_3D_points.at<double>(2)));
    }

    return final_3D_positions;
}


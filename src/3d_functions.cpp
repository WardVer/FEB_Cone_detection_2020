#include "3d_functions.hpp"

using namespace cv;
using namespace std;

void projector3D::stereo3D(Mat &frame1, Mat &frame2, std::vector<bbox_t> *result_vec1, std::vector<bbox_t> *result_vec2, std::vector<cv::Point3f> *points3D, std::vector<cv::Point2f> *offsets)
{

    cv::Mat rough_3d;
    cv::Mat rough_2d;

    std::vector<cv::Mat> rough_2d_vec;

    for (auto &i : *result_vec1)
    {
        rough_3d = estimate_3d(int(i.x + i.w / 2), int(i.y + i.h));
        rough_2d = estimate_2d(rough_3d);
        rough_2d_vec.push_back(rough_2d);
    }

    std::vector<bbox_t> new_result_vec = new_boxes(*result_vec1, rough_2d_vec);

    *result_vec2 = new_result_vec;

    cone_offset(result_vec1, &new_result_vec, frame1, frame2, offsets);

    cone_positions(result_vec1, points3D, offsets);

    return;
}

int projector3D::read_extrinsics()
{

    /*
        * read the extrinsic parameters of the cameras that we achieved from the calibration and saved in a file
        * This is a combination of the rotation and translation of the second camera compared to the first camera
        * 
        * This is needed to be able to locate a point in 3D with stereo vision
        * 
        * This also makes it possible to project a point from the 3D coordinate system of one camera to the 3D coordinate system of the other camera.
        */

    FileStorage fs;
    fs.open("extrinsics.xml", FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open extrinsics" << endl;
        return 1;
    }

    FileNode node = fs["R"];
    node >> rotation_relative;

    node = fs["T"];
    node >> translation_relative;


    fs.release();

    return 0;
}

int projector3D::read_intrinsics()
{

    /*
        * read the intrinsic parameters of the cameras that we achieved from the calibration and saved in a file
        * This is a combination of the camera matrix K and distortion matrix D of both cameras.
        * The combination of these matrices describes how the camera projects a 3D point to its 2D image plane
        * 
        * To calculate the location on the image of an object located in the 3D coordinate system of the camera, simply multiply
        * the 3D point vector with the camera matrix K of the camera that it was recorded on.
        */

    FileStorage fs;
    fs.open("intrinsics.xml", FileStorage::READ);

    if (!fs.isOpened())
    {
        cerr << "Failed to open extrinsics" << endl;
        return 1;
    }

    FileNode node = fs["K1"];
    node >> K1;

    node = fs["D1"];
    node >> D1;

    node = fs["K2"];
    node >> K2;

    node = fs["D2"];
    node >> D2;

    fs.release();

    return 0;
}

void projector3D::cam_pose_to_origin()

{

    /*
        * In this function we calculate the rotation and translation of the ground relative to the camera. 
        * The results are based in the 3D coordinate system of the camera.
        * To project a point from the 3D ground coordinate system to the 3D camera coordinate system,
        * we just add the translation vector from this function to the 3D vector of the point you want to project
        * and then multiply this vector with the rotation vector achieved from this function.
        */

    //initialize imgPts and objPts.
    //imgPts will be the pixel coordinates of the ground pattern points on our image.
    //objPts will be the real life coordinates of the ground pattern with the origin the top right corner.
    vector<Point2f> imgPts;
    vector<Point3f> objPts;

    Mat rvec;
    Mat tvec;

    bool found = false;

    Mat groundpicture = imread("groundpattern.jpg");
    Mat gray;
    cvtColor(groundpicture, gray, COLOR_BGR2GRAY);

    //find the chessboard pattern points on our image
    found = cv::findChessboardCorners(gray, boardSize, imgPts,
                                      CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
    if (found)
    {
        cornerSubPix(gray, imgPts, Size(5, 5), Size(-1, -1),
                     TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));
    }

    //fill in objPts with the real life chessboard pattern point coordinates
    for (int j = 0; j < boardSize.height; j++)
        for (int k = 0; k < boardSize.width; k++)
            objPts.push_back(Point3f(j * squareSize, k * squareSize, 0));

    //find the rotation and translation of the pattern in the 3D camera coordinate system
    solvePnP(objPts, imgPts, K1, D1, rvec, tvec);

    Mat newvec;
    Mat newvec_t;

    //convert rotation vector format to rotation matrix format, we can't use the vector anywhere but
    //the matrix is very useful for easy rotation calculations.
    Rodrigues(rvec, newvec);

    //copy the rotation vector into a temporary one to avoid copying already changed columns
    newvec.copyTo(newvec_t);

    /*the axi of the pattern are a bit messed up but here i switch the axes around to make
        *Z forward
        *X right
        *Y up
        */
    Mat column = (newvec_t.col(2));
    column.copyTo(newvec.col(1));

    column = (newvec_t.col(1));
    column.copyTo(newvec.col(0));

    column = (0 - newvec_t.col(0));
    column.copyTo(newvec.col(2));

    rotation_ground = newvec;
    translation_ground = tvec;

    
}

cv::Mat projector3D::estimate_3d(int cone_x, int cone_y)
{

    /*calculate the approximate 3D location by finding the intersection of the point of the image and the ground plane
        *we can find with this formula:
        *
        * s * {u, v, 1} = K * (R * {x, y, z} + T)
        * 
        * {x, y, z} is a point that is located in another coordinate system. This system has a rotation of R and rotation of T
        * compared to the camera coordinate system, and K is the camera matrix for the camera.
        * u, v are the 2D image pixel coordinates, s is just a constant.
        * 
        * transform this formula into this to continue:
        * 
        * s * R.inv * K.inv * {u, v, 1} = {x, y, z} + R.inv * T
        * 
        * this results in just 2 vectors of size 3 on each side of the '='
        * if we say y is 0 (ground), we can find s by just using the second element in the 2 vectors
        * with s we can find the rest of the 3D coordinates as well.
        * 
        */

    //init {u, v, 1}
    Mat uvPoint = (Mat_<double>(3, 1) << cone_x, cone_y, 1);

    //calculate the left and right side of the formula
    Mat leftSideMat = rotation_ground.inv() * K1.inv() * uvPoint;
    Mat rightSideMat = rotation_ground.inv() * translation_ground;

    //use the second element of the resulting vectors of size 3 to find s
    double s = rightSideMat.at<double>(1) / leftSideMat.at<double>(1);

    //calculate the approximate 3D location
    Mat m_out = rotation_ground.inv() * (s * K1.inv() * uvPoint - translation_ground);

    return m_out;
}

cv::Mat projector3D::estimate_2d(Mat &rough_3d)
{

    //Once we have a 3D estimate from the first image we can use this to predict the location of the cone on the second image
    //s * {u, v, 1} = P * {x, y, z, 1}

    //the rough 3D positions are in ground coordinates, so first convert them to 3D camera coordinates
    Mat conepos_camera_fromcamera = rotation_ground * rough_3d + translation_ground;

    //add a 1 to be able to multiply it with the projection matrix
    Mat rough_3d_h;
    Mat one = Mat::ones(Size(1, 1), CV_64FC1);
    vconcat(conepos_camera_fromcamera, one, rough_3d_h);

    //multiply with projection matrix to get the 2D locations on the second camera
    Mat P;
    hconcat(rotation_relative, translation_relative, P);
    P = K2 * P;

    cv::Mat rough_2d = P * rough_3d_h;

    //divide by s
    rough_2d /= rough_2d.at<double>(2);

    return rough_2d;
}

std::vector<bbox_t> projector3D::new_boxes(const std::vector<bbox_t> &result_vec, std::vector<Mat> &rough_2d_vec)
{
    std::vector<bbox_t> new_2d;

    //we calculated the 3D positions of the cones based on the bottom middle of the boxes.
    //use this function to revert back to top left of the box (for drawing purposes)
    for (int a = 0; a < result_vec.size(); a++)
    {
        bbox_t result_copy = result_vec.at(a);
        result_copy.x = (int)rough_2d_vec[a].at<double>(0) - (int)(result_copy.w / 2);
        result_copy.y = (int)rough_2d_vec[a].at<double>(1) - (int)(result_copy.h);
        new_2d.push_back(result_copy);
    }

    return new_2d;
}

Mat projector3D::draw_features(Mat &img1, Mat &img2, vector<Point2f> &corners1, vector<Point2f> &corners2)
{

    //draw the features on the cones (optional)

    Mat img1_copy;
    Mat img2_copy;

    img1.copyTo(img1_copy);
    img2.copyTo(img2_copy);

    for (int i = 0; i < corners1.size(); i++)
    {
        if (corners1[i].x >= 0 && corners1[i].y >= 0)
            circle(img1_copy, corners1[i], 2, Scalar(0, 255, 0));
    }

    for (int i = 0; i < corners2.size(); i++)
    {
        if (corners2[i].x >= 0 && corners2[i].y >= 0)
            circle(img2_copy, corners2[i], 2, Scalar(0, 255, 0));
    }

    Mat concatcone;

    hconcat(img1_copy, img2_copy, concatcone);

    return concatcone;
}

void projector3D::cone_offset(const vector<bbox_t> *result_vec1, const vector<bbox_t> *result_vec2, Mat &img1, Mat &img2, std::vector<cv::Point2f>* offsets)
{

    //find the exact relative positions of the 2 cones. This function finds it by comparing the 2 cones
    //its finds features on one cone and looks for the same features on the second cone. By finding the
    //average of the offsets between the features we can find the exact relative position between the 2 cones

    Mat img1ROI;
    Mat img2ROI;

    Mat gray1;
    Mat gray2;

    for (int a = 0; a < result_vec1->size(); a++)
    {

        int x1 = max((int)result_vec1->at(a).x, 0);
        int y1 = max((int)result_vec1->at(a).y, 0);
        int w1 = min((int)(result_vec1->at(a).w), img1.cols - (int)(result_vec1->at(a).x));
        int h1 = min((int)(result_vec1->at(a).h), img1.rows - (int)(result_vec1->at(a).y));

        int x2 = max((int)(result_vec2->at(a).x), 0);
        int y2 = max((int)(result_vec2->at(a).y), 0);
        int w2 = min((int)(result_vec2->at(a).w), img2.cols - (int)(result_vec2->at(a).x));
        int h2 = min((int)(result_vec2->at(a).h), img2.rows - (int)(result_vec2->at(a).y));

        if (w2 <= 5)
        {
            offsets->push_back(Point(x2 - x1, y2 - y1));
            continue;
        }

        int w = min(w1, w2);
        int h = min(h1, h2);

        img1ROI = img1(Rect(x1, y1, w, h));
        img2ROI = img2(Rect(x2, y2, w, h));

        cv::cuda::GpuMat ROI1;
        cv::cuda::GpuMat ROI2;
        cv::cuda::GpuMat ROI1g;
        cv::cuda::GpuMat ROI2g;

        ROI1.upload(img1ROI);
        ROI2.upload(img2ROI);
        
        cv::cuda::cvtColor(ROI1, ROI1g, COLOR_BGR2GRAY);
        cv::cuda::cvtColor(ROI2, ROI2g, COLOR_BGR2GRAY);

        //cv::cuda::equalizeHist(ROI1g, ROI1g);
        //cv::cuda::equalizeHist(ROI2g, ROI2g);

        cv::cuda::GpuMat corners1;
        cv::cuda::GpuMat corners2;

        cv::cuda::GpuMat status;
        cv::cuda::GpuMat errors;

        std::chrono::steady_clock::time_point begin1;
        std::chrono::steady_clock::time_point end1;
        std::chrono::steady_clock::time_point begin2;
        std::chrono::steady_clock::time_point end2;

        begin1 = std::chrono::steady_clock::now();

        
        cv::Ptr<cv::cuda::CornersDetector> cornerDetector = cv::cuda::createGoodFeaturesToTrackDetector(ROI1g.type(), 8, 0.1, 4);
        cornerDetector->detect(ROI1g, corners1);

        end1 = std::chrono::steady_clock::now();
        begin2 = std::chrono::steady_clock::now();
        //status.resize(corners1.size());
        Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(cv::Size(21,21));
        d_pyrLK_sparse->calc(ROI1g, ROI2g, corners1, corners2, status, errors);

        end2 = std::chrono::steady_clock::now();
        std::cout << "feature time1 = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() << "[µs]" << std::endl;
        std::cout << "feature time2 = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[µs]" << std::endl;

        float xsum = 0;
        float ysum = 0;
        int matches = 0;

        /*std::vector<cv::cuda::GpuMat> corners1p;
        std::vector<cv::cuda::GpuMat> corners2p;

        cv::cuda::split(corners1, corners1p);*/
        
        Mat cpucorners1(corners1);
        Mat cpucorners2(corners2);
        Mat cpustatus(status);

        std::vector<Point2f> corners1p;
        std::vector<Point2f> corners2p;

        
        for (int i = 0; i < corners1.size().width; i++)
        {
            
            
            if (int(cpustatus.at<uchar>(i)) == 1)
            {
                matches++;
                xsum += cpucorners2.at<Vec2f>(0,i)[0] - cpucorners1.at<Vec2f>(0,i)[0];
                ysum += cpucorners2.at<Vec2f>(0,i)[1] - cpucorners1.at<Vec2f>(0,i)[1];

                corners1p.push_back(Point2f(cpucorners1.at<Vec2f>(0,i)[0], cpucorners1.at<Vec2f>(0,i)[1]));
                corners2p.push_back(Point2f(cpucorners2.at<Vec2f>(0,i)[0], cpucorners2.at<Vec2f>(0,i)[1]));

                
            }
        }
        cout << "xsum:  " << cpustatus.at<float>(2) << endl;
        xsum /= matches;
        ysum /= matches;

        offsets->push_back(Point2f(xsum + (float)x2 - (float)x1, ysum + (float)y2 - (float)y1));

        Point testPoint((int)xsum, (int)ysum);
        Mat concatcone = draw_features(img1ROI, img2ROI, corners1p, corners2p);
        resize(concatcone, concatcone, Size(500, 250));

        imshow("yeet", concatcone);
    }

    return;
}

void projector3D::cone_positions(const std::vector<bbox_t> *result_vec1,  std::vector<cv::Point3f> *points3D, std::vector<cv::Point2f> *offsets)
{
    //get the final 3D position of the cone

    vector<Point2d> Points2D_1;
    vector<Point2d> Points2D_2;

    Mat points3D_t;

    for (int m = 0; m < result_vec1->size(); m++)
    {
        //these are the locations of 1. the cones on the first image and 2. the cones on the second image calculated by adding the offsets from cone_offset()
        //the 2D points are stored as the top left of the boxes but we use the bottom middle of the boxes.
        Points2D_1.push_back(Point2d(result_vec1->at(m).x + result_vec1->at(m).w / 2, result_vec1->at(m).y + result_vec1->at(m).h));
        Points2D_2.push_back(Point2d(result_vec1->at(m).x + result_vec1->at(m).w / 2 + offsets->at(m).x, result_vec1->at(m).y + result_vec1->at(m).h + offsets->at(m).y));
    }

    triangulatePoints(P1, P2, Points2D_1, Points2D_2, points3D_t);

    //some transforms to get a usable 3D vector format
    Mat points3D_2;
    Mat points3D_r = points3D_t.t();
    points3D_r = points3D_r.reshape(4);
    cv::convertPointsFromHomogeneous(points3D_r, points3D_2);

    Mat positions[3];

    split(points3D_2, positions);

    //some transforms to get a usable 3D vector format
    for (int c = 0; c < points3D_2.rows; c++)
    {
        Mat new_3D_points = (Mat_<double>(3, 1) << positions[0].at<double>(c), positions[1].at<double>(c), positions[2].at<double>(c));
        new_3D_points = rotation_ground.inv() * (new_3D_points);
        new_3D_points -= rotation_ground.inv() * (translation_ground);
        points3D->push_back(Point3d(new_3D_points.at<double>(0), new_3D_points.at<double>(1), new_3D_points.at<double>(2)));
    }

    cout << *points3D << endl;
    
    return;
}

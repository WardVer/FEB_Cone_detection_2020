

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>


#define GPU
#define OPENCV

#include "3d_functions.hpp"
#include "yolo_v2_class.hpp"    // imported functions from DLL
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    for (auto &i : result_vec) {
        cv::Scalar color;
        if(i.obj_id == 0) color = cv::Scalar(255,100,0);
        if(i.obj_id == 1) color = cv::Scalar(0,255,255);
        if(i.obj_id == 2) color = cv::Scalar(0,100,255);

        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}


template<typename T>
class send_one_replaceable_object_t {
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t() : a_ptr(NULL)
    {}
};

int main(int argc, char *argv[])
{
    std::string  names_file = "data/3cones.names";
    std::string  cfg_file = "cfg/3cones.cfg";
    std::string  weights_file = "weights/3cones_last.weights";
    std::string filename;

    float const thresh = 0.2;

    Detector detector(cfg_file, weights_file);

    bool const send_network = false;        // true - for remote detection
    bool const use_kalman_filter = false;   // true - for stationary camera

    bool detection_sync = false;             // true - for video-file

    while (true)
    {

        try {
            
                cv::Mat cur_frame;
                std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
                std::atomic<int> current_fps_cap(0), current_fps_det(0);
                std::atomic<bool> exit_flag(false);
                std::chrono::steady_clock::time_point steady_start, steady_end;
                int video_fps = 30;

                cv::VideoCapture cap1;
                cap1.open(0);
                cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap1.set(cv::CAP_PROP_FPS, 30);
                cap1 >> cur_frame;

                cv::VideoCapture cap2;
                cap2.open(2);
                cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap2.set(cv::CAP_PROP_FPS, 30);
                cap2 >> cur_frame;


                
                cv::Mat R;
                cv::Mat T;

                cv::Mat K1;
                cv::Mat K2;
                cv::Mat D1;
                cv::Mat D2;

                read_extrinsics(&R, &T);
                read_intrinsics(&K1, &D1, &K2, &D2);

                std::vector<cv::Mat> camera_data;
                camera_data = cam_pose_to_origin(cv::Size(9,6), 3.78f, K1, D1);
                

                cv::Size const frame_size = cur_frame.size();

                std::cout << "\n Video size: " << frame_size << std::endl;

                struct detection_data_t {
                    cv::Mat cap_frame1;
                    cv::Mat cap_frame2;
                    std::shared_ptr<image_t> det_image;
                    std::vector<bbox_t> result_vec1;
                    std::vector<bbox_t> result_vec2;
                    cv::Mat draw_frame;
                    cv::Mat rough_3d;
                    std::vector<cv::Point> offsets;
                    std::vector<cv::Point3d> points3D;
                    bool new_detection;
                    uint64_t frame_id;
                    bool exit_flag;
                    detection_data_t() : exit_flag(false), new_detection(false) {}
                };

                const bool sync = detection_sync; // sync data exchange
                send_one_replaceable_object_t<detection_data_t> cap2prepare, cap2draw,
                    prepare2detect, draw2show, draw2write, draw2net, detect2threed, threed2draw;

                std::thread t_cap, t_cap1, t_cap2, t_prepare, t_detect, t_3d, t_post, t_draw, t_write, t_network;

                // capture new video-frame
                if (t_cap.joinable()) t_cap.join();
                t_cap = std::thread([&]()
                {
                    uint64_t frame_id = 0;
                    detection_data_t detection_data;
                    do {
                        detection_data = detection_data_t();
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        t_cap1 = std::thread([&]()
                        {
                            cap1 >> detection_data.cap_frame1;
                        });

                        t_cap2 = std::thread([&]()
                        {
                            cap2 >> detection_data.cap_frame2;
                        });

                        t_cap1.join();
                        t_cap2.join();

                        fps_cap_counter++;
                        detection_data.frame_id = frame_id++;
                        if (detection_data.cap_frame1.empty() || detection_data.cap_frame1.empty() || exit_flag) {
                            std::cout << " exit_flag: detection_data.cap_frame1.size = " << detection_data.cap_frame1.size() << std::endl;
                            std::cout << " exit_flag: detection_data.cap_frame2.size = " << detection_data.cap_frame2.size() << std::endl;
                            detection_data.exit_flag = true;
                            detection_data.cap_frame1 = cv::Mat(frame_size, CV_8UC3);
                            detection_data.cap_frame2 = cv::Mat(frame_size, CV_8UC3);
                        }

                        cap2prepare.send(detection_data);
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

                    } while (!detection_data.exit_flag);
                    std::cout << " t_cap exit \n";
                });



                // pre-processing video frame (resize, convertion)
                t_prepare = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = cap2prepare.receive();

                        det_image = detector.mat_to_image_resize(detection_data.cap_frame1);
                        detection_data.det_image = det_image;
                        prepare2detect.send(detection_data);    // detection

                    } while (!detection_data.exit_flag);
                    std::cout << " t_prepare exit \n";
                });


                // detection by Yolo
                if (t_detect.joinable()) t_detect.join();
                t_detect = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = prepare2detect.receive();
                        det_image = detection_data.det_image;
                        std::vector<bbox_t> result_vec;

                        if(det_image)
                            result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height, thresh, true);  // true
                        fps_det_counter++;
                        //std::this_thread::sleep_for(std::chrono::milliseconds(150));

                        detection_data.new_detection = true;
                        detection_data.result_vec1 = result_vec;
                        detect2threed.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_detect exit \n";
                });

                // draw rectangles (and track objects)
                t_draw = std::thread([&]()
                {
                    
                    detection_data_t detection_data;
                    do {

                            // get new Detection result if present
                        if (threed2draw.is_object_present()) {   // use old captured frame
                            detection_data = threed2draw.receive();
                        }
                        // get new Captured frame
                        else {
                            continue;
                            std::vector<bbox_t> old_result_vec1 = detection_data.result_vec1; 
                            std::vector<bbox_t> old_result_vec2 = detection_data.result_vec2;// use old detections
                            detection_data = cap2draw.receive();
                            detection_data.result_vec1 = old_result_vec1;
                            detection_data.result_vec2 = old_result_vec2;
                        }
                        cv::Mat draw_frame1 = detection_data.cap_frame1.clone();
                        cv::Mat draw_frame2 = detection_data.cap_frame2.clone();
                        cv::Mat draw_frame_c;
                        std::vector<bbox_t> result_vec1 = detection_data.result_vec1;
                        std::vector<bbox_t> result_vec2 = detection_data.result_vec2;

                        draw_boxes(draw_frame1, result_vec1, current_fps_det, current_fps_cap);
                        draw_boxes(draw_frame2, result_vec2, current_fps_det, current_fps_cap);

                        

                        cv::hconcat(draw_frame1, draw_frame2, draw_frame_c);

                        for(int l = 0; l < detection_data.offsets.size(); l++)
                        {
                            int cone1_x = detection_data.result_vec1[l].x + detection_data.result_vec1[l].w/2;
                            int cone1_y = detection_data.result_vec1[l].y + detection_data.result_vec1[l].h/2;

                            int cone2_x = cone1_x + draw_frame1.cols + detection_data.offsets[l].x;
                            int cone2_y = cone1_y + detection_data.offsets[l].y;

                            cv::line(draw_frame_c, cv::Point(cone1_x, cone1_y), cv::Point(cone2_x, cone2_y), cv::Scalar(0,0,255), 3);
                        }

                        detection_data.draw_frame = draw_frame_c;
                        draw2show.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_draw exit \n";
                });

                //estimate 3D
                
                t_3d = std::thread([&]()
                {
                    
                    detection_data_t detection_data;
                    do {
                        
                        detection_data = detect2threed.receive();
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        std::vector<bbox_t> const result_vec = detection_data.result_vec1;

                        if(result_vec.size() == 0) continue;

                        cv::Mat rough_3d;
                        cv::Mat rough_2d;
                        
                        std::vector<cv::Mat> rough_2d_vec;
                        
                        for (auto &i : result_vec) {
                            rough_3d = estimate_3d(int(i.x+i.w/2), int(i.y + i.h), K1, camera_data[0], camera_data[1]);
                            rough_2d = estimate_2d(rough_3d, R, T, K2, camera_data[0], camera_data[1]);
                            rough_2d_vec.push_back(rough_2d);
                        }
                        std::vector<bbox_t> new_result_vec = new_boxes(result_vec, rough_2d_vec);

                        
                        
                        detection_data.result_vec2 =  new_result_vec;

                        std::vector<cv::Point> offsets;
                        offsets = cone_offset(result_vec, new_result_vec, detection_data.cap_frame1, detection_data.cap_frame2);
                        
                        


                        cv::Mat P2;
                        hconcat(R,T,P2);
                        P2 = K2 * P2;

                        cv::Mat P1 = (cv::Mat_<double>(3,4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
                        P1 = K2 * P1;

                        detection_data.offsets = offsets;
                        std::vector<cv::Point3d> points3D = cone_positions(result_vec, offsets, P1, P2, camera_data[0], camera_data[1]);

                        detection_data.points3D = points3D;
                        //std::cout << detection_data.cap_frame1 << std::endl;
                        /*cv::imshow("yoink", detection_data.cap_frame1);
                        cv::waitKey(0);

                        cv::imshow("yoink", detection_data.cap_frame2);
                        cv::waitKey(0);*/

                        threed2draw.send(detection_data);
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

                        

                    } while (!detection_data.exit_flag);
                    std::cout << " t_draw exit \n";
                });


                // show detection
                detection_data_t detection_data;
                do {
                    
                    steady_end = std::chrono::steady_clock::now();
                    float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                    if (time_sec >= 1) {
                        current_fps_det = fps_det_counter.load() / time_sec;
                        current_fps_cap = fps_cap_counter.load() / time_sec;
                        steady_start = steady_end;
                        fps_det_counter = 0;
                        fps_cap_counter = 0;
                    }

                    detection_data = draw2show.receive();
                    cv::Mat draw_frame;
                    cv::resize(detection_data.draw_frame, draw_frame, cv::Size(1280, 480));
                    
                    //if (extrapolate_flag) {
                    //    cv::putText(draw_frame, "extrapolate", cv::Point2f(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
                    //}

                    cv::imshow("window name", draw_frame);
                    int key = cv::waitKey(1);    // 3 or 16ms
                    if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
                    //if (key == 'e') extrapolate_flag = !extrapolate_flag;
                    if (key == 27) { exit_flag = true;}

                    //std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap << std::endl;
                } while (!detection_data.exit_flag);
                std::cout << " show detection exit \n";

                cv::destroyWindow("window name");
                // wait for all threads
                if (t_cap.joinable()) t_cap.join();
                if (t_3d.joinable()) t_3d.join();
                if (t_prepare.joinable()) t_prepare.join();
                if (t_detect.joinable()) t_detect.join();
                if (t_post.joinable()) t_post.join();
                if (t_draw.joinable()) t_draw.join();
                if (t_write.joinable()) t_write.join();
                if (t_network.joinable()) t_network.join();

                break;

            }
   
        catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { std::cerr << "unknown exception \n"; getchar(); }
    }

    return 0;
}
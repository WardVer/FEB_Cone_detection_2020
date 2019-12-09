

#include <opencv2/opencv.hpp> // C++
#include <opencv2/core/version.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "GxIAPI.h"
#include "DxImageProc.h"

int main(int argc, char *argv[])
{
    GX_STATUS
    status = GX_STATUS_SUCCESS;
    GX_DEV_HANDLE hDevice = NULL;
    uint32_t nDeviceNum = 0;
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
    status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, 20000);
    status = GXSetFloat(hDevice, GX_FLOAT_GAIN, 16);
    status = GXSetEnum(hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_RED);
    status = GXSetFloat(hDevice, GX_FLOAT_BALANCE_RATIO, 1.4);
    status = GXSetEnum(hDevice, GX_ENUM_BALANCE_RATIO_SELECTOR, GX_BALANCE_RATIO_SELECTOR_BLUE);
    status = GXSetFloat(hDevice, GX_FLOAT_BALANCE_RATIO, 1.5);
    if (status == GX_STATUS_SUCCESS)
    {
        // Define the incoming parameters of GXDQBuf.
        PGX_FRAME_BUFFER pFrameBuffer;
        // Stream On.
        status = GXStreamOn(hDevice);
        if (status == GX_STATUS_SUCCESS)
        {
            // Calls GXDQBuf to get a frame of image.
            while (1)
            {
                status = GXDQBuf(hDevice, &pFrameBuffer, 20);
                if (status == GX_STATUS_SUCCESS)
                {
                    if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
                    {
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        cv::Mat yeet;
                        yeet.create(pFrameBuffer->nHeight, pFrameBuffer->nWidth, CV_8UC1);
                        VxInt32 DxStatus = DxBrightness(pFrameBuffer->pImgBuf,pFrameBuffer->pImgBuf,pFrameBuffer->nWidth * pFrameBuffer->nHeight,0);
                        memcpy(yeet.data, pFrameBuffer->pImgBuf, pFrameBuffer->nWidth * pFrameBuffer->nHeight);                                          
                        cv::cvtColor(yeet, yeet, cv::COLOR_BayerRG2RGB);
                        cv::resize(yeet, yeet, cv::Size(1280, 512));

                        cv::flip(yeet,yeet, 0);
                        cv::imshow("yeet", yeet);
                        cv::waitKey(1);

                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

                    }
                    // Calls GXQBuf to put the image buffer back into the library
                    //and continue acquiring.
                    status = GXQBuf(hDevice, pFrameBuffer);
                }
            }
        }
        // Sends a stop acquisition command.
        status = GXStreamOff(hDevice);
    }
    status = GXCloseDevice(hDevice);
    status = GXCloseLib();

    return 0;
}
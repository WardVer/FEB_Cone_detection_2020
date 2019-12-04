

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
    uint32_t
        nDeviceNum = 0;
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
    status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, 100000);
    if (status == GX_STATUS_SUCCESS)
    {
        // Define the incoming parameters of GXDQBuf.
        PGX_FRAME_BUFFER pFrameBuffer;
        // Stream On.
        status = GXStreamOn(hDevice);
        if (status == GX_STATUS_SUCCESS)
        {
            // Calls GXDQBuf to get a frame of image.
            status = GXDQBuf(hDevice, &pFrameBuffer, 1000);
            if (status == GX_STATUS_SUCCESS)
            {
                if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS)
                {
                    
                    cv::Mat yeet;
                    yeet.create(pFrameBuffer->nHeight, pFrameBuffer->nWidth, CV_8UC3);
                    void *pRGB24Buffer = malloc(pFrameBuffer->nWidth * pFrameBuffer->nHeight * 3);
                    std::cout << pFrameBuffer->nWidth * pFrameBuffer->nHeight * 3 << std::endl;
                    DxRaw8toRGB24(pFrameBuffer->pImgBuf, pRGB24Buffer, pFrameBuffer->nWidth, pFrameBuffer->nHeight, RAW2RGB_NEIGHBOUR, BAYERRG, true);
                    std::cout << "yeet "<< std::endl;
                    memcpy(yeet.data, pRGB24Buffer, pFrameBuffer->nWidth * pFrameBuffer->nHeight * 3);
                    cv::resize(yeet, yeet, cv::Size(1280, 512));
                    cv::cvtColor(yeet, yeet, cv::COLOR_BGR2RGB);
                    cv::imshow("yeet", yeet);
                    cv::waitKey(0);
                }
                // Calls GXQBuf to put the image buffer back into the library
                //and continue acquiring.
                status = GXQBuf(hDevice, pFrameBuffer);
            }
        }
        // Sends a stop acquisition command.
        status = GXStreamOff(hDevice);
    }
    status = GXCloseDevice(hDevice);
    status = GXCloseLib();

    return 0;
}
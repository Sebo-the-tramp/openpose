#include <openpose/producer/baslerWrapper.hpp>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> // cv::undistort, cv::initUndistortRectifyMap
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap for OpenCV 4
#endif
#ifdef USE_BASLER_CAMERA
    
    #include <cstdio>
    #include <chrono>

    // Include files to use the pylon API.
    #include <pylon/PylonIncludes.h>
    #include <GenApi/INodeMap.h>
    #ifdef PYLON_WIN_BUILD
    #    include <pylon/PylonGUI.h>
    #endif

#endif
#include <openpose/3d/cameraParameterReader.hpp>

using namespace cv;
using namespace std;
using namespace Pylon;

namespace op
{       
    
    #ifdef USE_BASLER_CAMERA

        /*
         * This function converts between Spinnaker::ImagePtr container to cv::Mat container used in OpenCV.
        */
        cv::Mat BaslerWrapperToCvMat(const CGrabResultPtr ptrGrabResult)
        {
            try
            {                                                                                              
                CImageFormatConverter formatConverter;//seb
                formatConverter.OutputPixelFormat = PixelType_BGR8packed;//seb
                CPylonImage pylonImage;//seb                

                // Convert the grabbed buffer to pylon imag
                formatConverter.Convert(pylonImage, ptrGrabResult);

                // Create an OpenCV image out of pylon image
                Mat openCvImage(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());

                // Create a new matrix for dst and copy the data from openCvImage
                cv::Mat dst(openCvImage.size(), openCvImage.type());
                openCvImage.copyTo(dst);

                return dst;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return cv::Mat();
            }
        }        

    #else
        const std::string USE_BASLER_CAMERA_ERROR{"OpenPose CMake must be compiled with the `USE_BASLER_CAMERA`"
            " flag in order to use the BASLER camera. Alternatively, disable `--BASLER_camera`."};
    #endif

    struct BaslerWrapper::ImplBaslerWrapper
    {
        #ifdef USE_BASLER_CAMERA
            bool mInitialized;
            CameraParameterReader mCameraParameterReader;
            Point<int> mResolution;
            
            //BASLER variables
            uint32_t no_sources = 0;            		
			std::vector<CGrabResultPtr> pBASLER_video_mBuffer;
           
            std::vector<cv::Mat> mCvMats;
            std::vector<std::string> mSerialNumbers;
            
            // Camera index
            const int mCameraIndex;
            // Undistortion
            const bool mUndistortImage;
            std::vector<cv::Mat> mRemoveDistortionMaps1;
            std::vector<cv::Mat> mRemoveDistortionMaps2;
            // Thread
            bool mThreadOpened;
            
            // CONVERTER

            // TODO create the buffer
            //std::vector<Spinnaker::ImagePtr> mBuffer;
            
            std::mutex mBufferMutex;
            std::atomic<bool> mCloseThread;
            std::thread mThread;

            ImplBaslerWrapper(const bool undistortImage, const int cameraIndex) :
                mInitialized{false},
                mCameraIndex{cameraIndex},
                mUndistortImage{undistortImage}
            {
            }

            void readAndUndistortImage(const int i, const CGrabResultPtr ptrGrabResult,
                                       const cv::Mat& cameraIntrinsics = cv::Mat(),
                                       const cv::Mat& cameraDistorsions = cv::Mat())
            {
                try
                {
                    // Spinnaker to cv::Mat
					cv::Mat cvMatDistorted = BaslerWrapperToCvMat(ptrGrabResult);
                    //const auto cvMatDistorted = BaslerWrapperToCvMat(ptrGrabResult);
                    
                    // Undistort
                    if (mUndistortImage)
                    {
                        // Sanity check
                        if (cameraIntrinsics.empty() || cameraDistorsions.empty())
                            error("Camera intrinsics/distortions were empty.", __LINE__, __FUNCTION__, __FILE__);
                        // // Option a - 80 ms / 3 images
                        // // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
                        // cv::undistort(cvMatDistorted, mCvMats[i], cameraIntrinsics, cameraDistorsions);
                        // // In OpenCV 2.4, cv::undistort is exactly equal than cv::initUndistortRectifyMap
                        // (with CV_16SC2) + cv::remap (with LINEAR). I.e., opLog(cv::norm(cvMatMethod1-cvMatMethod2)) = 0.
                        // Option b - 15 ms / 3 images (LINEAR) or 25 ms (CUBIC)
                        // Distortion removal - not required and more expensive (applied to the whole image instead of
                        // only to our interest points)
                        if (mRemoveDistortionMaps1[i].empty() || mRemoveDistortionMaps2[i].empty())
                        {
                            const auto imageSize = cvMatDistorted.size();
                            cv::initUndistortRectifyMap(cameraIntrinsics,
                                                        cameraDistorsions,
                                                        cv::Mat(),
                                                        // cameraIntrinsics instead of cv::getOptimalNewCameraMatrix to
                                                        // avoid black borders
                                                        cameraIntrinsics,
                                                        // #include <opencv2/calib3d/calib3d.hpp> for next line
                                                        // cv::getOptimalNewCameraMatrix(cameraIntrinsics,
                                                        //                               cameraDistorsions,
                                                        //                               imageSize, 1,
                                                        //                               imageSize, 0),
                                                        imageSize,
                                                        CV_16SC2, // Faster, less memory
                                                        // CV_32FC1, // More accurate
                                                        mRemoveDistortionMaps1[i],
                                                        mRemoveDistortionMaps2[i]);
                        }
                        cv::remap(cvMatDistorted, mCvMats[i],
                                  mRemoveDistortionMaps1[i], mRemoveDistortionMaps2[i],
                                  // cv::INTER_NEAREST);
                                  cv::INTER_LINEAR);
                                  // cv::INTER_CUBIC);
                                  // cv::INTER_LANCZOS4); // Smoother, but we do not need this quality & its >>expensive
                    }
                    // Baseline (do not undistort)
                    else
                      mCvMats[i] = cvMatDistorted.clone();
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void bufferingThread()
            {
                #ifdef USE_BASLER_CAMERA
                    try
                    {                                                
                        mCloseThread = false;

                        // helper variables
                        CGrabResultPtr ptrGrabResult;
                        // create support vector
                        std::vector<CGrabResultPtr> ptrGrabResult_vector;

                        // Get the transport layer factory.
                        CTlFactory& tlFactory = CTlFactory::GetInstance();

                        // Get all attached devices and exit application if no device is found.
                        DeviceInfoList_t devices;
                        if (tlFactory.EnumerateDevices(devices) == 0)
                        {
                            throw RUNTIME_EXCEPTION("No camera present.");
                        }

                        // list all devices
						for (auto& device : devices)
						{
							std::cout << "Device found: " << device.GetModelName() << std::endl;                            
						}
                                                
                        // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
                        CInstantCameraArray cameras(min(devices.size(), 4));

                        // Create and attach all Pylon Devices.
                        for (size_t i = 0; i < cameras.GetSize(); ++i)
                        {
                            cameras[i].Attach(tlFactory.CreateDevice(devices[i]));
							cameras[i].Open();
                            
							GENAPI_NAMESPACE::INodeMap& nodemap = cameras[i].GetNodeMap();                            
                            
                            // ** Custom Test Images **
                            // Disable standard test images
                            CEnumParameter(nodemap, "TestImageSelector").SetValue("Off");
                            // Enable custom test images
                            CEnumParameter(nodemap, "ImageFileMode").SetValue("On");
                            // Load custom test image from disk
							string path = "C:\\Users\\Sebastian Cavada\\\Documents\\SCSV\\Thesis\\data\\_images\\Synthetic\\DigitalTwin\\Test_2\\Cam" + to_string(i+1) + "\\DigitalTwinCameras.0000.png";
                            CStringParameter(nodemap, "ImageFilename").SetValue(path.c_str());
                            // Set custom framerate                            
							CFloatParameter(nodemap, "AcquisitionFrameRate").SetValue(4);                            
                            // Set colors
							CEnumParameter(nodemap, "PixelFormat").SetValue("BGR8Packed");
                            
                            // Print the camera information.
                            cout << "Using device " << cameras[i].GetDeviceInfo().GetModelName() << endl;
                            cout << "Friendly Name: " << cameras[i].GetDeviceInfo().GetFriendlyName() << endl;
                            cout << "Full Name    : " << cameras[i].GetDeviceInfo().GetFullName() << endl;
                            cout << "SerialNumber : " << cameras[i].GetDeviceInfo().GetSerialNumber() << endl;
                            cout << endl;
                        }                        

                        // Starts grabbing for all cameras starting with index 0. The grabbing
                        // is started for one camera after the other. That's why the images of all
                        // cameras are not taken at the same time.
                        // However, a hardware trigger setup can be used to cause all cameras to grab images synchronously.
                        // According to their default configuration, the cameras are
                        // set up for free-running continuous acquisition.
                        cameras.StartGrabbing();
                                                
                        while (!mCloseThread)
                        {                           
                            ptrGrabResult_vector.resize(no_sources);

                            // Grab c_countOfImagesToGrab from the cameras.
                            for (uint32_t i = 0; i < cameras.GetSize() && cameras.IsGrabbing(); ++i)
                            {
                                cameras.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

                                // Image grabbed successfully?
                                if (ptrGrabResult->GrabSucceeded())
                                {
                                    
                                    // When the cameras in the array are created the camera context value
                                    // is set to the index of the camera in the array.
                                    // The camera context is a user settable value.
                                    // This value is attached to each grab result and can be used
                                    // to determine the camera that produced the grab result.
                                    intptr_t cameraContextValue = ptrGrabResult->GetCameraContext();

                                    // do the check if I can just add the image to the buffer

                                    // add image to the vector
                                    ptrGrabResult_vector[cameraContextValue] = ptrGrabResult;
                                    //ptrGrabResult_vector[i] = ptrGrabResult;
                                    
                                }
                                else
                                {
                                    cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << endl;
                                }
                            }                           
                            
                            // Move to buffer
                            bool imagesExtracted = true;
                            
                            for (auto& ptrGrabResult : ptrGrabResult_vector)
                            {                                
                                if (!ptrGrabResult->GrabSucceeded())
                                {
									opLog("Image incomplete...", Priority::High, __LINE__, __FUNCTION__, __FILE__);

                                    imagesExtracted = false;
                                    break;
                                }
                            }

                            if (imagesExtracted)
                            {
                                std::unique_lock<std::mutex> lock{mBufferMutex};
                                std::swap(pBASLER_video_mBuffer, ptrGrabResult_vector);
                                lock.unlock();
                                
								//sleep for the number of delay between fps 33000 micro = 33ms -> 30 fps = 1000ms/30 = 33ms
                                std::this_thread::sleep_for(std::chrono::microseconds{100});
                            }
                            
                        }                        
                    }
                    catch (const std::exception& e)
                    {
                        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    }
                #endif
            }

            // This function acquires and displays images from each device.
            std::vector<Matrix> acquireImages(
                const std::vector<Matrix>& opCameraIntrinsics,
                const std::vector<Matrix>& opCameraDistorsions,
                const int cameraIndex = -1)
            {
                try
                {
                    OP_OP2CVVECTORMAT(cameraIntrinsics, opCameraIntrinsics)
                    OP_OP2CVVECTORMAT(cameraDistorsions, opCameraDistorsions)
                    std::vector<cv::Mat> cvMats;

                    // Retrieve, convert, and return an image for each camera
                    // In order to work with simultaneous camera streams, nested loops are
                    // needed. It is important that the inner loop be the one iterating
                    // through the cameras; otherwise, all images will be grabbed from a
                    // single camera before grabbing any images from another.

                    // // Get cameras - ~0.005 ms (3 cameras)
                    // std::vector<Spinnaker::CameraPtr> cameraPtrs(cameraList.GetSize());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     cameraPtrs.at(i) = cameraList.GetByIndex(i);

                    // Read raw images - ~0.15 ms (3 cameras)
                    // std::vector<Spinnaker::ImagePtr> imagePtrs(cameraPtrs.size());
                    // for (auto i = 0u; i < cameraPtrs.size(); i++)
                    //     imagePtrs.at(i) = cameraPtrs.at(i)->GetNextImage();
                    
                    // std::vector<Spinnaker::ImagePtr> imagePtrs;
                        
                    std::vector<CGrabResultPtr> ptrGrabResult_vector;

                    // Retrieve frame
                    auto cvMatRetrieved = false;
                    while (!cvMatRetrieved)
                    {
                        // Retrieve frame
                        std::unique_lock<std::mutex> lock{mBufferMutex};
                        if (!pBASLER_video_mBuffer.empty())
                        {
                            std::swap(ptrGrabResult_vector, pBASLER_video_mBuffer);
                            cvMatRetrieved = true;
                        }
                        // No frames available -> sleep & wait
                        else
                        {
                            lock.unlock();
                            std::this_thread::sleep_for(std::chrono::microseconds{5});
                        }
                    }
                    // Getting frames
                    // Retrieve next received image and ensure image completion
                    // Spinnaker::ImagePtr imagePtr = cameraPtrs.at(i)->GetNextImage();

                    // All images completed
                    bool imagesExtracted = true;
                    for (auto& imagePtr : ptrGrabResult_vector)
                    {
                        if (!imagePtr->GrabSucceeded())
                        {
                            opLog("Image incomplete with image status ", Priority::High, __LINE__, __FUNCTION__, __FILE__);
                            imagesExtracted = false;
                            break;
                        }
                    }
                    mCvMats.clear();
                    // Convert to cv::Mat
                    if (imagesExtracted)
                    {
                        // // Original image --> BGR uchar image - ~4 ms (3 cameras)
                        // for (auto& imagePtr : imagePtrs)
                        //     imagePtr = spinnakerImagePtrToColor(imagePtr);

                        // NOT SURE HERE IF I AM DOING GOOD
                        
                        // Init anti-distortion matrices first time
                        
                        if (mRemoveDistortionMaps1.empty())
                            mRemoveDistortionMaps1.resize(no_sources);
                        if (mRemoveDistortionMaps2.empty())
                            mRemoveDistortionMaps2.resize(no_sources);                        
                            
                        // Multi-thread undistort (slowest function in the class)
                        //     ~7.7msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + LINEAR)
                        //     ~23.2msec (3 cameras + multi-thread + (initUndistortRectifyMap + remap) + CUBIC)
                        //     ~35msec (3 cameras + multi-thread + undistort)
                        //     ~59msec (2 cameras + single-thread + undistort)
                        //     ~75msec (3 cameras + single-thread + undistort)
                        mCvMats.resize(no_sources);
                        // All cameras
                        if (cameraIndex < 0)
                        {
                            // Undistort image
                            if (mUndistortImage)
                            {
                                
                                std::vector<std::thread> threads(no_sources-1);
                                for (auto i = 0u; i < threads.size(); i++)
                                {
                                    // Multi-thread option
                                    
                                    threads.at(i) = std::thread{&ImplBaslerWrapper::readAndUndistortImage, this, i,
                                                                ptrGrabResult_vector.at(i), cameraIntrinsics.at(i),
                                                                cameraDistorsions.at(i)};                                                                
                                    //Single-thread option
                                    //readAndUndistortImage(i, video_frame_vector.at(i), cameraIntrinsics.at(i), cameraDistorsions.at(i));
                                }
                                readAndUndistortImage((int)ptrGrabResult_vector.size()-1, ptrGrabResult_vector.back(), cameraIntrinsics.back(),
                                                      cameraDistorsions.back());
                                // Close threads
                                for (auto& thread : threads)
                                    if (thread.joinable())
                                        thread.join();
                                
                            }
                            // Do not undistort image
                            else
                            {
                                for (auto i = 0u; i < no_sources; i++)
                                    readAndUndistortImage(i, ptrGrabResult_vector.at(i));
                            }
                        }
                        // Only 1 camera
                        else
                        {
                            // Sanity check
                            if ((unsigned int)cameraIndex >= ptrGrabResult_vector.size())
                                error("There are only " + std::to_string(ptrGrabResult_vector.size())
                                      + " cameras, but you asked for the "
                                      + std::to_string(cameraIndex+1) +"-th camera (i.e., `--NDI_camera_index "
                                      + std::to_string(cameraIndex) +"`), which doesn't exist. Note that the index is"
                                      + " 0-based.", __LINE__, __FUNCTION__, __FILE__);
                            /*
                            // Undistort image
                            if (mUndistortImage)
                                readAndUndistortImage(cameraIndex, video_frame_vector.at(cameraIndex), cameraIntrinsics.at(cameraIndex),
                                                      cameraDistorsions.at(cameraIndex));
                                                      */
                            // Do not undistort image
                            else
                                readAndUndistortImage(cameraIndex, ptrGrabResult_vector.at(cameraIndex));
                            mCvMats = std::vector<cv::Mat>{mCvMats[cameraIndex]};
                        }
                    }
                    OP_CV2OPVECTORMAT(opMats, mCvMats)
                    return opMats;
                }
                //catch (Exception &e)
                catch (std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            }
        #endif
    };

    BaslerWrapper::BaslerWrapper(const std::string& cameraParameterPath, const Point<int>& resolution,
                                       const bool undistortImage, const int cameraIndex)
        #ifdef USE_BASLER_CAMERA
            : upImpl{new ImplBaslerWrapper{undistortImage, cameraIndex}}
        #endif
    {
        #ifdef USE_BASLER_CAMERA
            try
            {

                // Clean previous unclosed builds (e.g., if core dumped in the previous code using the cameras)
                release();

                upImpl->mInitialized = true;

                // Print application build information
                opLog(std::string{ "Application build date: " } + __DATE__ + " " + __TIME__, Priority::High);

                // Retrieve list of cameras from the upImpl->mSystemPtr
                //upImpl->mCameraParameterReader = new cameraParameterReader();

                // STARTING OF CONFIGURATION OF THE LIBRARY

                // The exit code of the sample application.
                int exitCode = 0;
                int c_maxCamerasToUse = 4;

                // Before using any pylon methods, the pylon runtime must be initialized.
                PylonInitialize();

                // Get the transport layer factory.
                CTlFactory& tlFactory = CTlFactory::GetInstance();                

                // Get all attached devices and exit application if no device is found.
                DeviceInfoList_t devices;
                if (tlFactory.EnumerateDevices(devices) == 0)
                {
                    error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
                }

                // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
                CInstantCameraArray cameras(min(devices.size(), c_maxCamerasToUse));

                upImpl->no_sources = cameras.GetSize();

                upImpl->mSerialNumbers.resize(upImpl->no_sources);

                // Create and attach all Pylon Devices.
                for (size_t i = 0; i < cameras.GetSize(); ++i)
                {
                    cameras[i].Attach(tlFactory.CreateDevice(devices[i]));

                    // Print the model name of the camera.
                    cout << "Using device " << cameras[i].GetDeviceInfo().GetModelName() << endl;

					upImpl->mSerialNumbers[i] = cameras[i].GetDeviceInfo().GetSerialNumber();
                    //upImpl->mSerialNumbers[i] = i;
                }      

                opLog("Camera system initialized...", Priority::High);
        
                const auto& serialNumbers = upImpl->mSerialNumbers;
                for (auto i = 0u; i < serialNumbers.size(); i++)
                    opLog("Camera " + std::to_string(i) + " serial number set to "
                        + serialNumbers[i] + "...", Priority::High);
                if (upImpl->mCameraIndex >= 0)
                    opLog("Only using camera index " + std::to_string(upImpl->mCameraIndex) + ", i.e., serial number "
                        + serialNumbers[upImpl->mCameraIndex] + "...", Priority::High);


                // Read camera parameters from SN
                // Very Important for later on when more cameras are needed
                
                if (upImpl->mUndistortImage)
                {
                    // If all images required
                    if (upImpl->mCameraIndex < 0)
                        upImpl->mCameraParameterReader.readParameters(cameraParameterPath, serialNumbers);
                    // If only one required
                    else
                    {
                        upImpl->mCameraParameterReader.readParameters(
                            cameraParameterPath,
                            std::vector<std::string>(serialNumbers.size(), serialNumbers.at(upImpl->mCameraIndex)));
                    }
                }                                                                

                // Start buffering thread
                upImpl->mThreadOpened = true;                                
                upImpl->mThread = std::thread{&BaslerWrapper::ImplBaslerWrapper::bufferingThread, this->upImpl};

                // Get resolution
                const auto cvMats = getRawFrames();
                // Sanity check
                if (cvMats.empty())
                    error("Cameras could not be opened.", __LINE__, __FUNCTION__, __FILE__);
                // Get resolution
                upImpl->mResolution = Point<int>{cvMats[0].cols(), cvMats[0].rows()};

                const std::string numberCameras = std::to_string(upImpl->no_sources);
                opLog("\nRunning for " + numberCameras + " out of " /* add code to get the IP of the camera*/
                    + " camera(s)...\n\n*** IMAGE ACQUISITION ***\n", Priority::High);
            }            
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            UNUSED(cameraParameterPath);
            UNUSED(resolution);
            UNUSED(undistortImage);
            UNUSED(cameraIndex);
            error(USE_BASLER_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    BaslerWrapper::~BaslerWrapper()
    {
        try
        {
            release();
        }
        catch (const std::exception& e)
        {
            errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<Matrix> BaslerWrapper::getRawFrames()
    {
        try
        {
            #ifdef USE_BASLER_CAMERA
                try
                {                    
                    // Sanity check
                    if (upImpl->mUndistortImage &&
                        (unsigned long long) upImpl->no_sources
                            != upImpl->mCameraParameterReader.getNumberCameras())
                        error("The number of cameras must be the same as the INTRINSICS vector size.",
                          __LINE__, __FUNCTION__, __FILE__);
                    // Return frames                    
                    return upImpl->acquireImages(upImpl->mCameraParameterReader.getCameraIntrinsics(),
                                                 upImpl->mCameraParameterReader.getCameraDistortions(),
                                                 upImpl->mCameraIndex);
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            #else
                error(op::USE_BASLER_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> BaslerWrapper::getCameraMatrices() const
    {
        try
        {
            #ifdef USE_BASLER_CAMERA
                return upImpl->mCameraParameterReader.getCameraMatrices();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> BaslerWrapper::getCameraExtrinsics() const
    {
        try
        {
            #ifdef USE_BASLER_CAMERA
                return upImpl->mCameraParameterReader.getCameraExtrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> BaslerWrapper::getCameraIntrinsics() const
    {
        try
        {
            #ifdef USE_BASLER_CAMERA

                return upImpl->mCameraParameterReader.getCameraIntrinsics();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    Point<int> BaslerWrapper::getResolution() const
    {
        try
        {
            #ifdef USE_BASLER_CAMERA
                return upImpl->mResolution;
            #else
                return Point<int>{};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<int>{};
        }
    }

    bool BaslerWrapper::isOpened() const
    {
        try
        {
            #ifdef USE_BASLER_CAMERA
                return upImpl->mInitialized;
            #else
                return false;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void BaslerWrapper::release()
    {
        #ifdef USE_BASLER_CAMERA
            try
            {                
                // Releases all pylon resources.
                PylonTerminate();
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
    }
}
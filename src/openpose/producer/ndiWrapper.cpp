#include <openpose/producer/ndiWrapper.hpp>
#include <atomic>
#include <mutex>
#include <opencv2/imgproc/imgproc.hpp> // cv::undistort, cv::initUndistortRectifyMap
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp> // OPEN_CV_IS_4_OR_HIGHER
#ifdef OPEN_CV_IS_4_OR_HIGHER
    #include <opencv2/calib3d.hpp> // cv::initUndistortRectifyMap for OpenCV 4
#endif
#ifdef USE_NDI_CAMERA
    
    #include <cstdio>
    #include <chrono>
    #include <Processing.NDI.Lib.h>

    #ifdef _WIN32
    #ifdef _WIN64
    #pragma comment(lib, "Processing.NDI.Lib.x64.lib")
    #else // _WIN64
    #pragma comment(lib, "Processing.NDI.Lib.x86.lib")
    #endif // _WIN64
    #endif // _WIN32

#endif
#include <openpose/3d/cameraParameterReader.hpp>

using namespace cv;
using namespace std;

namespace op
{
        
    #ifdef USE_NDI_CAMERA        

        /*
         * This function converts between Spinnaker::ImagePtr container to cv::Mat container used in OpenCV.
        */
        cv::Mat NdiWrapperToCvMat(const NDIlib_video_frame_v2_t video_frame)
        {
            try
            {              
                
                Mat frame(video_frame.yres, video_frame.xres, CV_8UC4);
                Mat dst(video_frame.yres, video_frame.xres, CV_8UC3);
                
                frame.data = video_frame.p_data;
                cvtColor(frame, dst, COLOR_BGRA2BGR);
                
                /*
                Mat frame(video_frame.yres, video_frame.xres, CV_8UC2);                          
                frame.data = video_frame.p_data;				
                cvtColor(frame, frame, COLOR_YUV2BGR_UYVY);
                */

                // check if we succeeded
                if (dst.empty()) {
                    throw std::invalid_argument("Not succeded");
					return dst;
                }

                return dst;                
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return cv::Mat();
            }
        }        

    #else
        const std::string USE_NDI_CAMERA_ERROR{"OpenPose CMake must be compiled with the `USE_NDI_CAMERA`"
            " flag in order to use the NDI camera. Alternatively, disable `--NDI_camera`."};
    #endif

    struct NdiWrapper::ImplNdiWrapper
    {
        #ifdef USE_NDI_CAMERA
            bool mInitialized;
            CameraParameterReader mCameraParameterReader;
            Point<int> mResolution;
            
            //NDI variables
            const NDIlib_source_t* p_sources;
            uint32_t no_sources = 0;
            NDIlib_recv_instance_t pNDI_recv;
			std::vector<NDIlib_recv_instance_t> pNDI_recv_sources;
			std::vector<NDIlib_video_frame_v2_t> pNDI_video_mBuffer;


            //ndi::CameraList mCameraList;
            //Spinnaker::SystemPtr mSystemPtr;
            
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
            
            // TODO create the buffer
            //std::vector<Spinnaker::ImagePtr> mBuffer;
            
            std::mutex mBufferMutex;
            std::atomic<bool> mCloseThread;
            std::thread mThread;

            ImplNdiWrapper(const bool undistortImage, const int cameraIndex) :
                mInitialized{false},
                mCameraIndex{cameraIndex},
                mUndistortImage{undistortImage}
            {
            }

            void readAndUndistortImage(const int i, const NDIlib_video_frame_v2_t video_frame,
                                       const cv::Mat& cameraIntrinsics = cv::Mat(),
                                       const cv::Mat& cameraDistorsions = cv::Mat())
            {
                try
                {
                    // Spinnaker to cv::Mat
                    const auto cvMatDistorted = NdiWrapperToCvMat(video_frame);
                    
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
                #ifdef USE_NDI_CAMERA
                    try
                    {                                                
                        mCloseThread = false;                                                                      

                        // helper variables
                        NDIlib_video_frame_v2_t video_frame;                        
                        // create support vector
                        std::vector<NDIlib_video_frame_v2_t> video_frame_vector;                               
                                                
                        while (!mCloseThread)
                        {                           
                            video_frame_vector.resize(no_sources);

                            // Get image from each camera
                            for (auto i = 0u; i < no_sources; i++) {
                                
                                if (NDIlib_recv_capture_v2(pNDI_recv_sources[i], &video_frame, nullptr, nullptr, 5000) == NDIlib_frame_type_video) {
                                                                        
                                    // add image to the vector
									video_frame_vector[i] = video_frame;
                                }                                
								
								// Destroy the video frame
								NDIlib_recv_free_video_v2(pNDI_recv_sources[i], &video_frame);
                                
                            }
                            
                            // Move to buffer
                            bool imagesExtracted = true;
                            
                            for (auto& video_frame_ : video_frame_vector)
                            {                                
                                if (!video_frame_.p_data)
                                {
									opLog("Image incomplete...", Priority::High, __LINE__, __FUNCTION__, __FILE__);

                                    imagesExtracted = false;
                                    break;
                                }
                            }

                            if (imagesExtracted)
                            {
                                std::unique_lock<std::mutex> lock{mBufferMutex};
                                std::swap(pNDI_video_mBuffer, video_frame_vector);
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
                        
                    std::vector<NDIlib_video_frame_v2_t> video_frame_vector;

                    // Retrieve frame
                    auto cvMatRetrieved = false;
                    while (!cvMatRetrieved)
                    {
                        // Retrieve frame
                        std::unique_lock<std::mutex> lock{mBufferMutex};
                        if (!pNDI_video_mBuffer.empty())
                        {
                            std::swap(video_frame_vector, pNDI_video_mBuffer);
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
                    for (auto& imagePtr : video_frame_vector)
                    {
                        if (!imagePtr.p_data)
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
                                    
                                    threads.at(i) = std::thread{&ImplNdiWrapper::readAndUndistortImage, this, i,
                                                                video_frame_vector.at(i), cameraIntrinsics.at(i),
                                                                cameraDistorsions.at(i)};                                                                
                                    //Single-thread option
                                    //readAndUndistortImage(i, video_frame_vector.at(i), cameraIntrinsics.at(i), cameraDistorsions.at(i));
                                }
                                readAndUndistortImage((int)video_frame_vector.size()-1, video_frame_vector.back(), cameraIntrinsics.back(),
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
                                    readAndUndistortImage(i, video_frame_vector.at(i));
                            }
                        }
                        // Only 1 camera
                        else
                        {
                            // Sanity check
                            if ((unsigned int)cameraIndex >= video_frame_vector.size())
                                error("There are only " + std::to_string(video_frame_vector.size())
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
                                readAndUndistortImage(cameraIndex, video_frame_vector.at(cameraIndex));
                            mCvMats = std::vector<cv::Mat>{mCvMats[cameraIndex]};
                        }
                    }
                    OP_CV2OPVECTORMAT(opMats, mCvMats)
                    return opMats;
                }
                catch (Exception &e)
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

    NdiWrapper::NdiWrapper(const std::string& cameraParameterPath, const Point<int>& resolution,
                                       const bool undistortImage, const int cameraIndex)
        #ifdef USE_NDI_CAMERA
            : upImpl{new ImplNdiWrapper{undistortImage, cameraIndex}}
        #endif
    {
        #ifdef USE_NDI_CAMERA
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

                // Clean previous unclosed builds (e.g., if core dumped in the previous code using the cameras)
                // Not required, but "correct" (see the SDK documentation).
                if (!NDIlib_initialize())
                    error("Error on init");

                // Create a finder
                NDIlib_find_instance_t pNDI_find = NDIlib_find_create_v2();
                if (!pNDI_find)
					error("Error on finder");

                // Wait until there is one source
                upImpl->no_sources = 0;
                upImpl->p_sources = NULL;
                while (!upImpl->no_sources) {
                    // Wait until the sources on the network have changed
                    printf("Looking for sources ...\n");
                    NDIlib_find_wait_for_sources(pNDI_find, 1000/* One second */);
                    upImpl->p_sources = NDIlib_find_get_current_sources(pNDI_find, &upImpl->no_sources);
                }                

                // Display all the sources.
                printf("Network sources (%u found).\n", upImpl->no_sources);

                // Finish if there are no cameras
                if (upImpl->no_sources == 0)
                {
                    // Not required, but nice
                    NDIlib_destroy();
                    error("No cameras detected.", __LINE__, __FUNCTION__, __FILE__);
                }
                opLog("Camera system initialized...", Priority::High);
            

                //
                // Retrieve transport layer nodemaps and print device information for
                // each camera
                //
                // *** NOTES ***
                // This example retrieves information from the transport layer nodemap
                // twice: once to print device information and once to grab the device
                // serial number. Rather than caching the nodemap, each nodemap is
                // retrieved both times as needed.
                //
                opLog("\n*** DEVICE INFORMATION ***\n", Priority::High);
				upImpl->mSerialNumbers.resize(upImpl->no_sources);
                for (uint32_t i = 0; i < upImpl->no_sources; i++)
                {
                    printf("%u. %s  %s\n", i + 1, upImpl->p_sources[i].p_ndi_name, upImpl->p_sources[i].p_url_address);                    
					upImpl->mSerialNumbers[i] = upImpl->p_sources[i].p_ndi_name;
                }
                    
                // Connect to each camera
				upImpl->pNDI_recv_sources.resize(upImpl->no_sources);                
                for (auto i = 0u; i < upImpl->no_sources; i++)
                {
                    // Create an NDI receiver
                    NDIlib_recv_create_v3_t recv_create_desc;

                    //qui ho perso due giorni di lavoro
                    recv_create_desc.color_format = NDIlib_recv_color_format_BGRX_BGRA;
                    upImpl->pNDI_recv_sources[i] = NDIlib_recv_create_v3(&recv_create_desc);
                    if (!upImpl->pNDI_recv_sources[i]) {
                        // Failed to create the NDI receiver
                        error("Failed to create the NDI receiver.", __LINE__, __FUNCTION__, __FILE__);
                    }

                    // Connect the NDI receiver to the custom camera's NDI source	
                    NDIlib_recv_connect(upImpl->pNDI_recv_sources[i], upImpl->p_sources + i);
                }                

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
                upImpl->mThread = std::thread{&NdiWrapper::ImplNdiWrapper::bufferingThread, this->upImpl};

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
            catch (const Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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
            error(USE_NDI_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
        #endif
    }

    NdiWrapper::~NdiWrapper()
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

    std::vector<Matrix> NdiWrapper::getRawFrames()
    {
        try
        {
            #ifdef USE_NDI_CAMERA
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
                catch (const Exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                    return {};
                }
            #else
                error(USE_NDI_CAMERA_ERROR, __LINE__, __FUNCTION__, __FILE__);
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> NdiWrapper::getCameraMatrices() const
    {
        try
        {
            #ifdef USE_NDI_CAMERA
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

    std::vector<Matrix> NdiWrapper::getCameraExtrinsics() const
    {
        try
        {
            #ifdef USE_NDI_CAMERA
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

    std::vector<Matrix> NdiWrapper::getCameraIntrinsics() const
    {
        try
        {
            #ifdef USE_NDI_CAMERA

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

    Point<int> NdiWrapper::getResolution() const
    {
        try
        {
            #ifdef USE_NDI_CAMERA
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

    bool NdiWrapper::isOpened() const
    {
        try
        {
            #ifdef USE_NDI_CAMERA
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

    void NdiWrapper::release()
    {
        #ifdef USE_NDI_CAMERA
            try
            {
                // Destroy the receiver
                NDIlib_recv_destroy(upImpl->pNDI_recv);

                // Not required, but nice
                NDIlib_destroy();
            }
            catch (const Exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #endif
    }
}
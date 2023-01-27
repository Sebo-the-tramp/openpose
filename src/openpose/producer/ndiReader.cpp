#include <openpose/producer/ndiReader.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/string.hpp>
#include <openpose_private/utilities/openCvMultiversionHeaders.hpp>

namespace op
{
    NdiReader::NdiReader(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                           const bool undistortImage, const int cameraIndex) :
        Producer{ProducerType::NdiCamera, cameraParameterPath, undistortImage, -1},
        mNdiWrapper{cameraParameterPath, cameraResolution, undistortImage, cameraIndex},
        mFrameNameCounter{0ull}
    {
        try
        {
            // Get resolution
            const auto resolution = mNdiWrapper.getResolution();                        
            
            // Set resolution
            set(CV_CAP_PROP_FRAME_WIDTH, resolution.x);
            set(CV_CAP_PROP_FRAME_HEIGHT, resolution.y);            

            // TODO automatize the resolution

            // Custom resolution for now: 1280x720
            //set(CV_CAP_PROP_FRAME_WIDTH, 1280);
            //set(CV_CAP_PROP_FRAME_HEIGHT, 720);

        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NdiReader::~NdiReader()
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

    std::vector<Matrix> NdiReader::getCameraMatrices()
    {
        try
        {
            return mNdiWrapper.getCameraMatrices();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> NdiReader::getCameraExtrinsics()
    {
        try
        {
            return mNdiWrapper.getCameraExtrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::vector<Matrix> NdiReader::getCameraIntrinsics()
    {
        try
        {
            return mNdiWrapper.getCameraIntrinsics();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    std::string NdiReader::getNextFrameName()
    {
        try
        {
            const auto stringLength = 12u;
            return toFixedLengthString(mFrameNameCounter, stringLength);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    bool NdiReader::isOpened() const
    {
        try
        {
            return mNdiWrapper.isOpened();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return false;
        }
    }

    void NdiReader::release()
    {
        try
        {
            mNdiWrapper.release();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Matrix NdiReader::getRawFrame()
    {
        try
        {
            return mNdiWrapper.getRawFrames().at(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Matrix();
        }
    }
    
    std::vector<Matrix> NdiReader::getRawFrames()
    {
        try
        {
            mFrameNameCounter++; // Simple counter: 0,1,2,3,...
            return mNdiWrapper.getRawFrames();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }    

    double NdiReader::get(const int capProperty)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
            {
                if (Producer::get(ProducerProperty::Rotation) == 0.
                    || Producer::get(ProducerProperty::Rotation) == 180.)
                    return mResolution.x;
                else
                    return mResolution.y;
            }
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
            {
                if (Producer::get(ProducerProperty::Rotation) == 0.
                    || Producer::get(ProducerProperty::Rotation) == 180.)
                    return mResolution.y;
                else
                    return mResolution.x;
            }
            else if (capProperty == CV_CAP_PROP_POS_FRAMES)
                return (double)mFrameNameCounter;
            else if (capProperty == CV_CAP_PROP_FRAME_COUNT)
                return -1.;
            else if (capProperty == CV_CAP_PROP_FPS)
                return -1.;
            else
            {
                opLog("Unknown property.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
                return -1.;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }
    
    void NdiReader::set(const int capProperty, const double value)
    {
        try
        {
            if (capProperty == CV_CAP_PROP_FRAME_WIDTH)
                mResolution.x = {(int)value};
            else if (capProperty == CV_CAP_PROP_FRAME_HEIGHT)
                mResolution.y = {(int)value};
            else if (capProperty == CV_CAP_PROP_POS_FRAMES)
                opLog("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else if (capProperty == CV_CAP_PROP_FRAME_COUNT || capProperty == CV_CAP_PROP_FPS)
                opLog("This property is read-only.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
            else
                opLog("Unknown property.", Priority::Max, __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }    
}

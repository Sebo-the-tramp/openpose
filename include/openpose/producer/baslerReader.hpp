#ifndef OPENPOSE_PRODUCER_BASLER_READER_HPP
#define OPENPOSE_PRODUCER_BASLER_READER_HPP

#include <openpose/core/common.hpp>
#include <openpose/producer/producer.hpp>
#include <openpose/producer/baslerWrapper.hpp>

namespace op
{
    /**
     * BaslerReader is an abstract class to extract frames from a NDI camera. Its interface imitates the
     * cv::VideoCapture class, so it can be used quite similarly to the cv::VideoCapture class. Thus,
     * it is quite similar to VideoReader and WebcamReader. --> example: 192.168.0.164:5078
     */
    class OP_API BaslerReader : public Producer
    {
    public:
        /**
         * Constructor of BaslerReader. It opens all the available NDI cameras
         */
        explicit BaslerReader(const std::string& cameraParametersPath, const Point<int>& cameraResolution,
                            const bool undistortImage = true, const int cameraIndex = -1);

        virtual ~BaslerReader();

        std::vector<Matrix> getCameraMatrices();

        std::vector<Matrix> getCameraExtrinsics();

        std::vector<Matrix> getCameraIntrinsics();

        std::string getNextFrameName();

        bool isOpened() const;

        void release();

        double get(const int capProperty);

        void set(const int capProperty, const double value);

    private:
        BaslerWrapper mBaslerWrapper;
        Point<int> mResolution;
        unsigned long long mFrameNameCounter;

        Matrix getRawFrame();

        std::vector<Matrix> getRawFrames();

        DELETE_COPY(BaslerReader);
    };
}

#endif // OPENPOSE_PRODUCER_BASLER_READER_HPP
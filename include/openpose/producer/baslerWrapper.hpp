#ifndef OPENPOSE_PRODUCER_BASLER_WRAPPER_HPP
#define OPENPOSE_PRODUCER_BASLER_WRAPPER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * BaslerWrapper is a subclass of BaslerWrapper. It decouples the final interface (meant to imitates
     * cv::VideoCapture) from the NDI SDK wrapper.
     */
    class OP_API BaslerWrapper
    {
    public:
        /**
         * Constructor of BaslerWrapper. It opens all the available Pylon cameras
         * cameraIndex = -1 means that all cameras are taken
         */
        explicit BaslerWrapper(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                                  const bool undistortImage, const int cameraIndex = -1);

        virtual ~BaslerWrapper();

        std::vector<Matrix> getRawFrames();

        /**
         * Note: The camera parameters are only read if undistortImage is true. This should be changed to add a
         * new bool flag in the constructor, e.g., readCameraParameters
         */
        std::vector<Matrix> getCameraMatrices() const;

        std::vector<Matrix> getCameraExtrinsics() const;

        std::vector<Matrix> getCameraIntrinsics() const;

        Point<int> getResolution() const;

        bool isOpened() const;

        void release();

    private:
        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplBaslerWrapper;
        std::shared_ptr<ImplBaslerWrapper> upImpl;

        DELETE_COPY(BaslerWrapper);
    };
}

#endif // OPENPOSE_PRODUCER_BASLER_WRAPPER_HPP

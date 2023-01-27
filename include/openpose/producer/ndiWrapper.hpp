#ifndef OPENPOSE_PRODUCER_NDI_WRAPPER_HPP
#define OPENPOSE_PRODUCER_NDI_WRAPPER_HPP

#include <openpose/core/common.hpp>

namespace op
{
    /**
     * NdiWrapper is a subclass of NdiWrapper. It decouples the final interface (meant to imitates
     * cv::VideoCapture) from the NDI SDK wrapper.
     */
    class OP_API NdiWrapper
    {
    public:
        /**
         * Constructor of NdiWrapper. It opens all the available NDI cameras
         * cameraIndex = -1 means that all cameras are taken
         */
        explicit NdiWrapper(const std::string& cameraParameterPath, const Point<int>& cameraResolution,
                                  const bool undistortImage, const int cameraIndex = -1);

        virtual ~NdiWrapper();

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
        struct ImplNdiWrapper;
        std::shared_ptr<ImplNdiWrapper> upImpl;

        DELETE_COPY(NdiWrapper);
    };
}

#endif // OPENPOSE_PRODUCER_NDI_WRAPPER_HPP

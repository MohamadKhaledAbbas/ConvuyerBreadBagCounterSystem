"""
Frame source factory for creating appropriate frame sources.
"""

from src.frame_source.FrameSource import FrameSource
from src.frame_source.OpenCvFrameSource import OpenCVFrameSource
from src.utils.platform import IS_RDK


class FrameSourceFactory:
    """
    Factory for creating frame sources based on type.
    """
    
    @staticmethod
    def create(source_type: str, **kwargs) -> FrameSource:
        """
        Create a frame source.
        
        Args:
            source_type: 'ros2' or 'opencv'
            
        Kwargs for ROS2:
            topic: ROS2 topic name
            target_fps: Target frames per second
            
        Kwargs for OpenCV:
            source: Video source (file path, camera index, or RTSP URL)
            target_fps: Target FPS for frame pacing
            testing_mode: If True, enables synchronous reading
            
        Returns:
            FrameSource instance
        """
        if source_type.lower() == 'ros2':
            if not IS_RDK:
                raise ValueError(
                    "ROS2 frame source only available on RDK platform. "
                    "Use 'opencv' source type on Windows/other platforms."
                )
            # Import ROS2 frame source only when needed
            from src.frame_source.Ros2FrameServer import FrameServer
            topic = kwargs.get('topic', '/nv12_images')
            return FrameServer(topic=topic)
        
        elif source_type.lower() == 'opencv':
            source = kwargs.get('source', 0)
            target_fps = kwargs.get('target_fps', None)
            testing_mode = kwargs.get('testing_mode', False)
            return OpenCVFrameSource(
                source,
                target_fps=target_fps,
                testing_mode=testing_mode
            )
        
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

def rgb2ntsc(src):
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    return np.dot(src, T.T)

def ntsc2rbg(src):
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    return np.dot(src, T.T)

def build_pyramid(src, pyramid_type='gaussian', levels=3):
    pyramid = [src]
    for _ in range(levels):
        src = cv2.pyrDown(src)
        pyramid.append(src)
    
    if pyramid_type == 'laplacian':
        for i in range(levels, 0, -1):
            expanded = cv2.pyrUp(pyramid[i])
            pyramid[i-1] = cv2.subtract(pyramid[i-1], expanded)
    return pyramid

def load_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    video_tensor = np.array([cap.read()[1] for _ in range(frame_count)], dtype=np.float32)
    return video_tensor, fps

def temporal_ideal_filter(tensor, low, high, fps):
    fft = fftpack.fft(tensor, axis=0)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low, bound_high = (np.abs(frequencies - low)).argmin(), (np.abs(frequencies - high)).argmin()
    
    fft[:bound_low], fft[bound_high:-bound_high], fft[-bound_low:] = 0, 0, 0
    return np.abs(fftpack.ifft(fft, axis=0))

def process_video(video_tensor, levels, amplify, pyramid_type):
    tensor_transformed = [build_pyramid(frame, pyramid_type=pyramid_type, levels=levels)[-1] for frame in video_tensor]
    tensor_transformed = np.array(tensor_transformed, dtype=np.float32)
    tensor_transformed *= amplify
    return tensor_transformed

def save_video(video_tensor, filename="out.avi"):
    height, width = video_tensor.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
    
    for frame in video_tensor:
        writer.write(cv2.convertScaleAbs(frame))
    writer.release()

def upsample_to_original(tensor, original_shape):
    while tensor.shape[1] < original_shape[1] or tensor.shape[2] < original_shape[2]:
        tensor = cv2.pyrUp(tensor)
    return tensor

def magnify(video_name, low, high, levels=3, amplification=20, mode='color'):
    video_tensor, fps = load_video(video_name)
    if mode == 'color':
        pyramid_type = 'gaussian'
    elif mode == 'motion':
        pyramid_type = 'laplacian'
    else:
        raise ValueError("Invalid mode. Choose 'color' or 'motion'.")
    
    tensor_transformed = process_video(video_tensor, levels, amplification, pyramid_type)
    filtered = temporal_ideal_filter(tensor_transformed, low, high, fps)
    
    # Up-sample the filtered tensor to match the original's shape
    filtered_upsampled = np.array([upsample_to_original(frame, video_tensor[0].shape) for frame in filtered])
    
    result = video_tensor + filtered_upsampled
    
    save_video(result)



if __name__ == "__main__":
    magnify("drone.mp4", 0.4, 3, mode='motion')

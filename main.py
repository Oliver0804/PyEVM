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

#def upsample_to_original(tensor, original_shape):
#    while tensor.shape[1] < original_shape[1] or tensor.shape[2] < original_shape[2]:
#        tensor = cv2.pyrUp(tensor)
#    return tensor
def upsample_to_original(tensor, original_shape):
    while tensor.shape[0] < original_shape[0] or tensor.shape[1] < original_shape[1]:
        tensor = cv2.pyrUp(tensor)

    # Calculate the required padding or cropping
    diff_height = tensor.shape[0] - original_shape[0]
    diff_width = tensor.shape[1] - original_shape[1]

    if diff_height > 0:
        tensor = tensor[:-diff_height, :, :]
    elif diff_height < 0:
        tensor = np.pad(tensor, ((0, -diff_height), (0, 0), (0, 0)), mode='constant')

    if diff_width > 0:
        tensor = tensor[:, :-diff_width, :]
    elif diff_width < 0:
        tensor = np.pad(tensor, ((0, 0), (0, -diff_width), (0, 0)), mode='constant')

    return tensor




def load_video_chunk(video_filename, start_frame, chunk_size):
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    video_tensor = np.array([cap.read()[1] for _ in range(chunk_size) if cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count], dtype=np.float32)
    return video_tensor, fps


def magnify(video_name, low, high, levels=3, amplification=20, mode='color', chunk_size=100):
    # If chunk_size is not provided, process the whole video in one go
    if chunk_size is None:
        video_tensor, fps = load_video(video_name)
        total_frames = video_tensor.shape[0]
        chunks = 1
    else:
        cap = cv2.VideoCapture(video_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate the number of chunks based on user input
        chunks = total_frames // chunk_size + (total_frames % chunk_size != 0)

    output_frames = []

    for i in range(chunks):
        start_frame = i * chunk_size
        end_frame = min(start_frame + chunk_size, total_frames)  # Handle the last chunk which might be smaller
        video_tensor, fps = load_video_chunk(video_name, start_frame, end_frame - start_frame)

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
        output_frames.extend(result)

    save_video(np.array(output_frames))



if __name__ == "__main__":
    magnify("drone.mp4", 2, 3, mode='motion', chunk_size=100)

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# Assuming a keypoints model that outputs 18 keypoints for each person
KEYPOINTS = {
    0: 'Nose',
    1: 'LeftEye',
    2: 'RightEye',
    3: 'LeftEar',
    4: 'RightEar',
    5: 'LeftShoulder',
    6: 'RightShoulder',
    7: 'LeftElbow',
    8: 'RightElbow',
    9: 'LeftWrist',
    10: 'RightWrist',
    11: 'LeftHip',
    12: 'RightHip',
    13: 'LeftKnee',
    14: 'RightKnee',
    15: 'LeftAnkle',
    16: 'RightAnkle',
    17: 'Neck'
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    try:
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    except (OSError, IOError) as e:
        print(f"Error opening engine file {engine_file_path}: {str(e)}")
    except Exception as e:
        print(f"Error loading the engine from file {engine_file_path}: {str(e)}")



def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

def infer(engine, h_input, d_input, h_output, d_output, image):
    stream = cuda.Stream()
    # Normalize the image and copy it to the pagelocked memory
    np.copyto(h_input, 1.0 - image.ravel())
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference
    with engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    # Reshape the output to a 2D array with shape (num_detections, num_keypoints*2)
    output = h_output.reshape(-1, len(KEYPOINTS)*2)
    return output

def main():
    engine = load_engine('weights/yolov8l-pose.engine')

    if engine is None:
        print("Engine could not be loaded")
    else:
        print("Engine loaded successfully")
    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (engine.get_binding_shape(0)[1:3]))
        output = infer(engine, h_input, d_input, h_output, d_output, frame_resized)

        for person in output:
            keypoints = person.reshape(-1, 2)
            for i, keypoint in enumerate(keypoints):
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

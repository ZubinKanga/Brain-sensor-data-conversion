from pylsl import StreamInlet, resolve_stream
import numpy as np


class ndf_lsl:

    def __init__(self, stream_type=None, stream_name=None, source_id=None, frame_rate=16, buffersize = 1.0):
        self.stream_type = stream_type
        self.stream_name = stream_name
        self.source_id = source_id
        self.frame_rate = frame_rate
        self.buffer_size_time = buffersize


    def ndf_setup(self):
        # Resolve stream
        self.ndf_resolve()
        self.stream = self.streams[0]

        # Get stream info
        self.ndf_stream_info()

        # Create LSL inlet for first resolved stream
        self.inlet = StreamInlet(self.stream)
        if self.inlet:
            print("Inlet created")
        else:
            print("Failed to create inlet for stream")

        # Create NaN data buffer
        self.ndf_ringbuffer()

        # Create empty 2D np array to represent the frame remnant from
        # a previous call to ndf_read()
        self.frame_remnant = np.empty((0, self.channel_count), float)

    def ndf_ringbuffer(self):
        self.buffer = np.empty((self.buffer_size_samples, self.channel_count), float)
        self.buffer[:] = np.nan

    def ndf_updatebuffer(self, frame):
        self.buffer = np.concatenate((self.buffer[self.frame_size:, :], frame), axis=0)


    def ndf_resolve(self):
        # Resolve stream the first non-None property available
        self.streams = None
        if self.stream_type:
            self.streams = resolve_stream('type', self.stream_type)
            print("Stream resolved by type " + self.stream_type)
            return
        if self.stream_name:
            self.streams = resolve_stream('name', self.stream_name)
            print("Stream resolved by name " + self.stream_name)
            return
        if self.source_id:
            self.streams = resolve_stream('source_id', self.source_id)
            print("Stream resolved by source ID " + self.source_id)
            return
        print("Failed to resolve stream with given properties")

    def ndf_stream_info(self):
        self.stream_name = self.stream.name()
        self.stream_type = self.stream.type()
        self.source_id = self.stream.source_id()
        self.channel_count = self.stream.channel_count()
        self.sampling_rate = self.stream.nominal_srate()
        self.host_name = self.stream.hostname()
        self.host_name = self.stream.hostname()
        self.metadata = self.stream.desc()
        self.buffer_size_samples = int(self.buffer_size_time*self.sampling_rate)

        # Calculate frame size (in samples)
        self.frame_period = 1.0 / self.frame_rate  # in seconds
        self.frame_size = int(float(self.sampling_rate) / float(self.frame_rate))

    # Reads a single frame of data
    def ndf_read(self):
        frame = self.frame_remnant
        while len(frame) < self.frame_size:
            # Read new chunk
            chunk = self.inlet.pull_chunk(max_samples=self.frame_size)
            chunk = np.array(chunk[0])
            if len(chunk) == 0:
                continue
            # Concatenate new chunk with frame
            frame = np.concatenate((frame, chunk), axis=0)

        if (len(frame) == self.frame_size):
            self.frame_remnant = np.empty((0, self.channel_count), float)
        else:
            self.frame_remnant = frame[self.frame_size:, :]
            frame = frame[:self.frame_size,:]

        # Update buffer with new frame (discarding older buffer)
        self.ndf_updatebuffer(frame)

        return frame, self.buffer

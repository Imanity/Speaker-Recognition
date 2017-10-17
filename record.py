import pyaudio
import wave
import configs

def recordAudio(time, output_filename):
    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()

    stream = p.open(format = FORMAT,
                    channels = configs.channels,
                    rate = configs.rate,
                    input = True,
                    frames_per_buffer = configs.chunk)

    str = input('Press Enter to begin recording.')

    print("* recording")

    frames = []

    for i in range(0, int(configs.rate / configs.chunk * time)):
        data = stream.read(configs.chunk)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(configs.channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(configs.rate)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    recordAudio(10, 'output.wav')

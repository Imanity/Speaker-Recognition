import os
from partition import getParts
from predict import predict

def analyzeAudio(filename, output):
    audio_parts = getParts(filename)
    #for i in audio_parts:
        #print(i.shape)
    #result = predict(audio_parts)
    #save_result = save(result, output)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Error arguments! Please used as "python analyzer.py AUDIO_FILE OUTPUT_FILE"')
        exit()
    if not os.path.exists(sys.argv[1]):
        print('File not exist')
        exit()
    print('Processing...')
    analyzeAudio(sys.argv[1], output)
    print('Done!')
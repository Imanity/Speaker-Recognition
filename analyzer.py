import os
import sys
from partition import getParts
from predict import predict


def analyzeAudio(filename, output):
    audio_parts = getParts(filename)
    result = predict(audio_parts)
    save_result = save(result, output)

def save(res, output):
    labels = res[0]
    texts = res[1]
    outFile = open(output, 'w')
    for i in range(0, len(labels)):
        outFile.write('Speaker ' + str(labels[i]) + ' : ' + texts[i] + '\n')
    outFile.close()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Error arguments! Please used as "python analyzer.py AUDIO_FILE OUTPUT_FILE"')
        exit()
    if not os.path.exists(sys.argv[1]):
        print('File not exist')
        exit()
    print('Processing...')
    analyzeAudio(sys.argv[1], sys.argv[2])
    print('Done!')
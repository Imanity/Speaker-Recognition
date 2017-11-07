# 音频处理
数字媒体2 多媒体 课程作业：音频处理
清华大学 软件41、42 唐人杰 卫国扬 罗皓天
***

### Requirements:
* Python 3.5.2
* numPy
* sciPy
* pyAudio
* python_speech_features
* sklearn
* matplotlib
* pycurl
* pillow
* tensorflow

### Intro:
* 音频分段：使用连续的一段低振幅波形作为分段依据, 预先使用均值滤波过滤音频噪声
* 话者识别：首先使用快速傅里叶变换(fft)绘制音频的时频图, 然后使用cifar-10卷积神经网络训练, 再将预测结果进行k-means聚类得到无监督话者识别结果
* 文本识别：调用百度api实现，运行过程需要联网

![ftimg](/img/ftimg.bmp)

![network](/img/network.png)

### 音频录制
* python ./record.py time wav_path

### 分段:
* 更改 main 函数中 partition 函数参数, 第一个参数为输入wav音频文件, 第二个参数为输出的分号段的wav音频文件名前缀
* python ./partition.py

### 生成音频的时频图:
* 更改 main 函数中 wav2bmp 函数参数, 第一个参数为wav音频文件, 第二个参数为生成图片路径
* python ./wav2bmp.py

### 训练:
* 使用网络：改造过的cifar-10网络
* 更改 getBMP(path, num) 函数, 使其可以从给定图片文件夹获取num张图片作为训练集
* 更改 getTestBMP(path) 函数, 使其可以读取测试集图片
* 更改 saver.save(sess, path) 函数, 存储训练好的模型
* python ./cifar.py

### 预测:
* python ./analyzer.py AUDIO_FILE OUTPUT_FILE
* 控制台会输出分段及话者识别label结果, 文本识别需要联网, 大约2min后会在OUTPUT_FILE中输出所有结果

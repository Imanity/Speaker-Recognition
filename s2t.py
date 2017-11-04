import wave  
import urllib, pycurl  
import urllib.request
import base64  
import json
import configs

#s2t_baiduapi(wav_path)
#input:path of wav
#output:str

#change MAC_cuid before using
MAC_cuid = configs.MAC

s2t_result = {}


def get_token():  
    apiKey = "S6ZSyKR5HH5Z3qvn7CeoipNn"  
    secretKey = "a44b961ffd7900ade7845d1787f94bd7"  
      
    auth_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + apiKey + "&client_secret=" + secretKey;  
      
    res = urllib.request.urlopen(auth_url)  
    json_data = res.read()
    return json.loads(json_data.decode("utf-8"))['access_token']  
      
def dump_res(buf):
    data = json.loads(buf.decode("utf-8"))
    global s2t_result
    s2t_result = data['result']
    #print(data['result'])
      
## post audio to server  
def use_cloud(token,wav_path):  
    fp = wave.open(wav_path, 'rb')  
    nf = fp.getnframes()  
    f_len = nf * 2  
    audio_data = fp.readframes(nf)  
    
    global MAC_cuid  
    cuid = MAC_cuid
    srv_url = 'http://vop.baidu.com/server_api' + '?cuid=' + cuid + '&token=' + token  
    http_header = [  
        'Content-Type: audio/wav; rate=16000',  
        'Content-Length: %d' % f_len  
    ]  
      
    c = pycurl.Curl()  
    c.setopt(pycurl.URL, str(srv_url))
    #c.setopt(c.RETURNTRANSFER, 1)  
    c.setopt(c.HTTPHEADER, http_header) 
    c.setopt(c.POST, 1)  
    c.setopt(c.CONNECTTIMEOUT, 60)  
    c.setopt(c.TIMEOUT, 60)  
    c.setopt(c.WRITEFUNCTION, dump_res)  
    c.setopt(c.POSTFIELDS, audio_data)  
    c.setopt(c.POSTFIELDSIZE, f_len)  
    c.perform()
      
def s2t_baiduapi(wav_path):
    token = get_token()
    use_cloud(token,wav_path)
    return s2t_result[0]


if __name__ == "__main__":  
    data = s2t_baiduapi('part_1.wav')
    print(data)

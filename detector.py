from threading import Thread, Lock
#from onvif import ONVIFCamera
import cv2
import sys

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, width)
        self.stream.set(4, height)
        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        #self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

def initCascade(type = 'face'):
    if type == 'face':
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    elif type == 'profile':
        return cv2.CascadeClassifier('haarcascade_profileface.xml')
    elif type == 'fullbody':
        return cv2.CascadeClassifier('haarcascade_fullbody.xml')
    elif type == 'upper':
        return cv2.CascadeClassifier('haarcascade_upperbody.xml')
    else:
        return None

def resize(img):
    height, width, layers = img.shape
    #print(img.shape)
    k = 1
    if (width >= 2000):
        k = 3
    elif (width >= 1000):
        k = 2
    new_w = int(width / k)
    new_h = int(height / k)
    img = cv2.resize(img, (new_w, new_h))
    #print(img.shape)

    return img

def findCenter(x, y, w, h):
    return (int(x + w/2), int(y + h/2))


def createPTZService(ip, port, login, passwd):
    '''
    mycam = ONVIFCamera(ip, port, login, passwd)
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    request = ptz.create_type('ContinuousMove')

    return ptz, request
    '''
    pass

def performMove(ptz, request, x, y, zoom, timeout):
    '''
    #print 'performing continuous move...'
    request.Velocity.PanTilt._x = x
    request.Velocity.PanTilt._y = y
    ptz.ContinuousMove(request)
    '''
    pass

def moveCam(frameCenter, objCenter):
    k = 1
    x = (objCenter[0] - frameCenter[0]) / (frameCenter[0] * k)
    y = (frameCenter[1] - objCenter[1]) / (frameCenter[1] * k)
    print(x, y)
    pass

def initDetector(ip, port, log, passwd):
    source = ip
    cascade = initCascade('profile')
    #ptz, request = createPTZService(source, port, log, passwd)
    source = int(ip) if len(ip) == 1 else 'rtsp://' + ip

    vs = WebcamVideoStream(src = source).start()
    while True:
        frame = vs.read()
        frame = resize(frame)
        frameCenter = findCenter(0, 0, frame.shape[1], frame.shape[0])
        #print(frameCenter)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        if (len(faces)):
            a = faces[0]
            print(a[0], a[1], a[2], a[3])
            faceCenter = findCenter(a[0], a[1], a[2], a[3])
            moveCam(frameCenter, faceCenter)

        for (x,y,w,h) in faces:
            faceCenter = findCenter(x,y,w,h)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 :
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    if (len(sys.argv) == 1):
        ip = '192.168.13.12'
        port = 80
        log = 'admin'
        passwd = 'Supervisor'
    else:
        ip = sys.argv[1]
        port = int(sys.argv[2])
        log = sys.argv[3]
        passwd = sys.argv[4]
    initDetector(ip, port, log, passwd)
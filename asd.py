import signal

done = False
def handle(*args):
    global done
    done = True
    print("ohno")
signal.signal(signal.SIGINT, handle)
signal.signal(signal.SIGTERM, handle)
import time
while not done:
    time.sleep(1)
    print("waiting")
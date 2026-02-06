import os,sys
import time

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds


client = fsds.FSDSClient()
client.confirmConnection()
client.enableApiControl(True)

carControl = fsds.CarControls()
carControl.steering =0
carControl.throttle = 1
carControl.brake=0
client.setCarControls(carControl)
time.sleep(5)
carControl.throttle = 0
carControl.brake=1
client.setCarControls(carControl)
time.sleep(10)
client.reset()
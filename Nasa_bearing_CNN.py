import time
import  tflite_runtime.interpreter as tflite
import numpy as np


interpreter = tflite.Interpreter('/home/pi/tflite/CNN_quan_float_in.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print ('---------------------------------------------')
#print (input_details)
print ('---------------------------------------------')
#print (output_details)
print ('---------------------------------------------')
# Load data
data = np.loadtxt(open('/home/pi/tflite/gear_s.csv', 'rb'), delimiter=',', skiprows=0)
for i in range(10):
	time.sleep(1)
	start = time.time()
	x = data[(500*i):(500*(i+1))].reshape(40, 100)
	x = x[np.newaxis, :, :, np.newaxis].astype('float32')
	#print ('x: ', x)

	interpreter.set_tensor(input_details[0]['index'],x)

	# Call lite model
	interpreter.invoke()
	out = interpreter.get_tensor(output_details[0]['index'])
	end = time.time()
	
	#out = out.tolist()
	#predict = out.index(max(out))
	#if out >= 0.5:
	#	print('Bearing condition: Normal')
	#if out < 0.5:
	#	print('Bearing condition: Fault')
	print('Inference time: ', (end - start))
	

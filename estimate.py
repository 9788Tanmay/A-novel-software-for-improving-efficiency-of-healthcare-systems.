import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from statistics import mean
import statistics
import tensorflow as tf
import matplotlib.pyplot as plt
#from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import feature_selection as fs

import keras
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from tkinter import *
master = Tk()
master.title("BESTIMATOR") 
master.geometry("1200x800")
candidates=[]
def findcandidates(data,tdd,target):
	forcandidates=[]
	forcandidates.append(tdd)
	forcandidates.append(mean(data))
	forcandidates.append(max(data))
	forcandidates.append(min(data))
	forcandidates.append(statistics.pstdev(data))
	candidates.append([forcandidates,target])
def regression(inputs,target,testinput,targetoutput):
	model = LinearRegression()
	model.fit(inputs, target)
	
	
	"""mallu=[8,2.3,5,6,1.11]
	mallu=np.array(mallu)
	mallu=mallu.transpose()
	mallu=mallu.transpose()"""
	yhat = model.predict(inputs)
	print("ESTMATING THE DEFECT CONTENT USING LINEAR REGRESSION ....")
	Label(master, text="ESTMATING THE DEFECT CONTENT USING LINEAR REGRESSION ....", fg="blue", font="none 22 ",anchor="center").grid(row=30,sticky=NSEW)
	#print(yhat)
	#res = "\n".join("{} {}".format(x, y) for x, y in zip(target, yhat))
	print("for inputset:")
	Label(master, text="for inputset:", fg="black", font="none 12 ",anchor="center").grid(row=31,sticky=NSEW)
	

	mae = mean_absolute_error(target, yhat)
	print('MAE: %.3f' % mae)
	linearmae=[]
	linearmae.append(mae)

	n_groups = len(target)
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, target, bar_width,
	alpha=opacity,
	color='b',
	label='targetoutput')

	rects2 = plt.bar(index + bar_width, yhat, bar_width,
	alpha=opacity,
	color='g',
	label='predicted')

	plt.xlabel('')
	plt.ylabel('value')
	plt.title('prediction ')
	
	plt.legend()

	plt.tight_layout()
	plt.show()
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=32,sticky=NSEW)

	yhat = model.predict(testinput)
	
	#print(yhat)
	print("for testset:")
	Label(master, text="for testset:", fg="black", font="none 12 ",anchor="center").grid(row=33,sticky=NSEW)
	res = "\n".join("{} {}".format(x, y) for x, y in zip(targetoutput, yhat))
	Label(master, text=res, fg="black", font="none 12 ",anchor="center").grid(row=34,sticky=NSEW)
	mae = mean_absolute_error(targetoutput, yhat)
	print('MAE: %.3f' % mae)
	linearmae.append(mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=35,sticky=NSEW)
	print(yhat)

def nonlinearregression(inputs,target,testinput,targetoutput):
	print("ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....")
	model = keras.Sequential()
	model.add(keras.layers.Dense(units = 1, activation = 'linear'))
	model.add(keras.layers.Dense(units = 64, activation = 'sigmoid'))
	model.add(keras.layers.Dense(units = 64, activation = 'sigmoid'))
	model.add(keras.layers.Dense(units = 1, activation = 'linear'))
	model.compile(loss='mse', optimizer="adam")

	# Display the model
	#model.summary()

	model.fit( inputs, target, epochs=10000,verbose=0)
	print(model.summary())
	#Label(master, text=str(model.summary()), fg="black", font="none 12 ",anchor="center").grid(row=100,sticky=NSEW)

	#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	inputs=inputs[:-2]
	target=target[:-2]
	y_predicted = model.predict(inputs)
	#print(y_predicted)
	print("ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....")
	Label(master, text="ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....", fg="blue", font="none 22 ",anchor="center").grid(row=40,sticky=NSEW)
	print("NUM    ESTIMATED" )
	#Label(master, text=, fg="black", font="none 12 ",anchor="center").grid(row=41,sticky=NSEW)

	res = "\n".join("{} {}".format(x, y) for x, y in zip(target, y_predicted))
	print(res)
	print(target)
	print(y_predicted)
	
	pred=[]
	targ=[]
	for j in y_predicted:
		pred.append(float(j[0]))

	for j in target:
		targ.append(float(j))
	print(pred)
	print(targ)
	n_groups = len(targ)
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, targ, bar_width,
	alpha=opacity,
	color='b',
	label='targetoutput')

	rects2 = plt.bar(index + bar_width, pred, bar_width,
	alpha=opacity,
	color='g',
	label='predicted')

	plt.xlabel('')
	plt.ylabel('value')
	plt.title('prediction ')
	
	plt.legend()

	plt.tight_layout()
	plt.show()



	#Label(master, text="NUM    ESTIMATED", fg="black", font="none 12 ",anchor="center").grid(row=60,sticky=NSEW) 


	mae = mean_absolute_error(target, y_predicted)
	nonlinearmae=[]
	print("for inputset:")
	Label(master, text="for inputset:", fg="black", font="none 12 ",anchor="center").grid(row=62,sticky=NSEW)
	print('MAE: %.3f' % mae)
	nonlinearmae.append(mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=63,sticky=NSEW)
	"""fig, ax = plt.subplots()
	ax.hist(target, 10, None, ec='red', fc='none', lw=1.5, histtype='step', label='n-gram')
	ax.hist(y_predicted, 10, None, ec='green', fc='none', lw=1.5, histtype='step', label='ensemble')
	ax.legend(loc='upper left')
	plt.show()"""
	#plt.plot(target,y_predicted)
	y_predicted = model.predict(testinput)
	print(y_predicted)
	print("NUM   [ ESTIMATED ]" )
	Label(master, text="NUM   [] ESTIMATED  ]", fg="black", font="none 12 ",anchor="center").grid(row=65,sticky=NSEW)
	res = "\n".join("{} {}".format(x, y) for x, y in zip(targetoutput, y_predicted))

	print(res)
	Label(master, text=res, fg="black", font="none 12 ",anchor="center").grid(row=66,sticky=NSEW)
	print("for testset:")
	Label(master, text="for testset:", fg="black", font="none 12 ",anchor="center").grid(row=80,sticky=NSEW)
	mae = mean_absolute_error(targetoutput, y_predicted)
	nonlinearmae.append(mae)
	print('MAE: %.3f' % mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=81,sticky=NSEW)
	button = Button(master, text='EXIT', fg='red', command=master.destroy)
	button.grid(row=83,sticky=NSEW)
	maegraph.append(linearmae)
	maegraph.append(nonlinearmae)
	print(linearmae)
	print(nonlinearmae)

	n_groups = 2
	

	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, linearmae, bar_width,
	alpha=opacity,
	color='b',
	label='LINEAR REGRESSION')

	rects2 = plt.bar(index + bar_width, nonlinearmae, bar_width,
	alpha=opacity,
	color='g',
	label='NON-LINEAR REGRESSION')

	plt.xlabel('setname')
	plt.ylabel('MEAN ABSOLUTE ERROR')
	plt.title('MAE COMPARISON')
	plt.xticks(index + bar_width, ('inputset', 'testset'))
	plt.legend()

	plt.tight_layout()
	plt.show()


def mutualinformation_neural():
	inputs=[]
	target=[]
	for i in candidates:
		inputs.append([i[0][0],i[0][-1]])
		#plt.scatter(i[0][0], i[0][-1])
		#inputs.append(i[0])
		target.append(i[1])
	#print((inputs))
	#print(len(target))
	#plt.show()
	testinput=inputs[-2:-1]
	#print(testinput)
	#inputs=inputs[:-2]
	targetoutput=target[-2:-1]
	#target=target[:-2]
	inputs=np.array(inputs)
	target=np.array(target)
	testinput=np.array(testinput)
	targetoutput=np.array(targetoutput)

	#print(inputs)
	inputs=inputs.transpose()
	#print(inputs)
	inputs=inputs.transpose()
	testinput=testinput.transpose()
	#print(testinput)
	testinput=testinput.transpose()
	#print(inputs)
	#print(inputs.shape)
	#print(target.shape)
	#print(inputs.data())
	#f_test, _ = f_regression(inputs, target)

	mi = mutual_info_regression(inputs, target)
	print("printing mutual information ")
	print(mi)
	regression(inputs,target,testinput,targetoutput)
	nonlinearregression(inputs,target,testinput,targetoutput)

	"""model = LinearRegression()
	model.fit(inputs, target)
	
	maegraph=[]
	mallu=[8,2.3,5,6,1.11]
	mallu=np.array(mallu)
	mallu=mallu.transpose()
	mallu=mallu.transpose()
	yhat = model.predict(inputs)
	print("ESTMATING THE DEFECT CONTENT USING LINEAR REGRESSION ....")
	Label(master, text="ESTMATING THE DEFECT CONTENT USING LINEAR REGRESSION ....", fg="blue", font="none 22 ",anchor="center").grid(row=30,sticky=NSEW)
	#print(yhat)
	#res = "\n".join("{} {}".format(x, y) for x, y in zip(target, yhat))
	print("for inputset:")
	Label(master, text="for inputset:", fg="black", font="none 12 ",anchor="center").grid(row=31,sticky=NSEW)
	

	mae = mean_absolute_error(target, yhat)
	print('MAE: %.3f' % mae)
	linearmae=[]
	linearmae.append(mae)

	n_groups = len(target)
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, target, bar_width,
	alpha=opacity,
	color='b',
	label='targetoutput')

	rects2 = plt.bar(index + bar_width, yhat, bar_width,
	alpha=opacity,
	color='g',
	label='predicted')

	plt.xlabel('')
	plt.ylabel('value')
	plt.title('prediction ')
	
	plt.legend()

	plt.tight_layout()
	plt.show()
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=32,sticky=NSEW)

	yhat = model.predict(testinput)
	
	#print(yhat)
	print("for testset:")
	Label(master, text="for testset:", fg="black", font="none 12 ",anchor="center").grid(row=33,sticky=NSEW)
	res = "\n".join("{} {}".format(x, y) for x, y in zip(targetoutput, yhat))
	Label(master, text=res, fg="black", font="none 12 ",anchor="center").grid(row=34,sticky=NSEW)
	mae = mean_absolute_error(targetoutput, yhat)
	print('MAE: %.3f' % mae)
	linearmae.append(mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=35,sticky=NSEW)
	print(yhat)"""
	
	"""fs = SelectKBest(score_func=mutual_info_regression, k='all')
	fs.fit(inputs, target)
	X_train_fs = fs.transform(inputs)
	print(X_train_fs)

	print(fs.scores_)"""
	"""model = Sequential()
	model.add(Dense(1, activation="sigmoid"))
	model.add(Dense(1, activation="ide"))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	model.fit(inputs,target)
	PredTestSet = model.predict(inputs)
	PredValSet = model.predict(inputs)
	print(PredTestSet)
	print(PredValSet)"""
	"""print("ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....")
	model = keras.Sequential()
	model.add(keras.layers.Dense(units = 1, activation = 'linear'))
	model.add(keras.layers.Dense(units = 64, activation = 'sigmoid'))
	model.add(keras.layers.Dense(units = 64, activation = 'sigmoid'))
	model.add(keras.layers.Dense(units = 1, activation = 'linear'))
	model.compile(loss='mse', optimizer="adam")

	# Display the model
	#model.summary()

	model.fit( inputs, target, epochs=10000,verbose=0)
	print(model.summary())
	#Label(master, text=str(model.summary()), fg="black", font="none 12 ",anchor="center").grid(row=100,sticky=NSEW)

	#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	inputs=inputs[:-2]
	target=target[:-2]
	y_predicted = model.predict(inputs)
	#print(y_predicted)
	print("ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....")
	Label(master, text="ESTMATING THE DEFECT CONTENT USING NON-LINEAR REGRESSION ....", fg="blue", font="none 22 ",anchor="center").grid(row=40,sticky=NSEW)
	print("NUM    ESTIMATED" )
	#Label(master, text=, fg="black", font="none 12 ",anchor="center").grid(row=41,sticky=NSEW)

	res = "\n".join("{} {}".format(x, y) for x, y in zip(target, y_predicted))
	print(res)
	print(target)
	print(y_predicted)
	
	pred=[]
	targ=[]
	for j in y_predicted:
		pred.append(float(j[0]))

	for j in target:
		targ.append(float(j))
	print(pred)
	print(targ)
	n_groups = len(targ)
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, targ, bar_width,
	alpha=opacity,
	color='b',
	label='targetoutput')

	rects2 = plt.bar(index + bar_width, pred, bar_width,
	alpha=opacity,
	color='g',
	label='predicted')

	plt.xlabel('')
	plt.ylabel('value')
	plt.title('prediction ')
	
	plt.legend()

	plt.tight_layout()
	plt.show()



	#Label(master, text="NUM    ESTIMATED", fg="black", font="none 12 ",anchor="center").grid(row=60,sticky=NSEW) 


	mae = mean_absolute_error(target, y_predicted)
	nonlinearmae=[]
	print("for inputset:")
	Label(master, text="for inputset:", fg="black", font="none 12 ",anchor="center").grid(row=62,sticky=NSEW)
	print('MAE: %.3f' % mae)
	nonlinearmae.append(mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=63,sticky=NSEW)
	"""#fig, ax = plt.subplots()
	##ax.hist(y_predicted, 10, None, ec='green', fc='none', lw=1.5, histtype='step', label='ensemble')
	#ax.legend(loc='upper left')
	#plt.show()
	"""
	#plt.plot(target,y_predicted)
	y_predicted = model.predict(testinput)
	print(y_predicted)
	print("NUM   [ ESTIMATED ]" )
	Label(master, text="NUM   [] ESTIMATED  ]", fg="black", font="none 12 ",anchor="center").grid(row=65,sticky=NSEW)
	res = "\n".join("{} {}".format(x, y) for x, y in zip(targetoutput, y_predicted))

	print(res)
	Label(master, text=res, fg="black", font="none 12 ",anchor="center").grid(row=66,sticky=NSEW)
	print("for testset:")
	Label(master, text="for testset:", fg="black", font="none 12 ",anchor="center").grid(row=80,sticky=NSEW)
	mae = mean_absolute_error(targetoutput, y_predicted)
	nonlinearmae.append(mae)
	print('MAE: %.3f' % mae)
	Label(master, text='MAE: %.3f' % mae, fg="red", font="none 12 ",anchor="center").grid(row=81,sticky=NSEW)
	button = Button(master, text='EXIT', fg='red', command=master.destroy)
	button.grid(row=83,sticky=NSEW)
	maegraph.append(linearmae)
	maegraph.append(nonlinearmae)
	print(linearmae)
	print(nonlinearmae)

	n_groups = 2
	

	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, linearmae, bar_width,
	alpha=opacity,
	color='b',
	label='LINEAR REGRESSION')

	rects2 = plt.bar(index + bar_width, nonlinearmae, bar_width,
	alpha=opacity,
	color='g',
	label='NON-LINEAR REGRESSION')

	plt.xlabel('setname')
	plt.ylabel('MEAN ABSOLUTE ERROR')
	plt.title('MAE COMPARISON')
	plt.xticks(index + bar_width, ('inputset', 'testset'))
	plt.legend()

	plt.tight_layout()
	plt.show()
	"""





def readdata():
	df=pd.read_csv('data.csv', sep=',',header=None)
		#print(df)
	#print(len(df))
	#print(df[0][1])
	



	for i in range(len(df)):
		firstarray = np.genfromtxt(df[0][i], delimiter=",")
		array=firstarray[1:]
		#print(len(array[1]))
		data=[]
	
		for j in array:
			#print(j[-1])
			data.append(j[-1])
		findcandidates(data,len(array[1])-2,df[1][i])
	mutualinformation_neural()
	#print(df[1][i])
	#forcandiadtes.append([data,df[1][i]])
	#print(forcandiadtes)
	#print(candidates)
#print(candidates)





candidates=[]
Label(master, text='    ESTIMATING....          ', fg="blue", font="none 24 bold",anchor="center").grid(row=0,column=0,sticky=NSEW)

Label(master, text='FILENAME', fg="black", font="none 12 ").grid(row=2,column=0,sticky=NSEW,pady=10)
e1 = Entry(master,justify='center') 
e1.grid(row=2, column=1,sticky=NSEW,  pady=10)
df=pd.read_csv('data.csv', sep=',',header=None)
for i in range(len(df)):
	firstarray = np.genfromtxt(df[0][i], delimiter=",")
	array=firstarray[1:]
	#print(len(array[1]))
	data=[]
	
	for j in array:
		#print(j[-1])
		data.append(j[-1])
	findcandidates(data,len(array[1])-2,df[1][i])
	#print(df[1][i])
	#forcandiadtes.append([data,df[1][i]])
	#print(forcandiadtes)
	#print(candidates)
print(candidates)

blackbutton = Button(text ='Calculate', fg ='black',command=mutualinformation_neural) 
blackbutton.grid(row=10,column=1,sticky=NSEW, padx=10, pady=10) 
#blackbutton.config(anchor=CENTER)

#button.config(anchor=CENTER)

#print(df)
#print(len(df))
#print(df[0][1])





#mutualinformation_neural()

mainloop() 

	

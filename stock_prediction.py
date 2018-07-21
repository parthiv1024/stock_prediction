import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates =[]
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		# Skip the first row since it's just column names
		next(csvFileReader)
		next(csvFileReader)
		for row in csvFileReader:
			# Only get days from 07 month (July)
			if row[0].split('/')[1] == '07':
				# First element in row is date. Split gets rid of the dashes and the 1 element of the date is the day
				dates.append(int(row[0].split('/')[2]))
				prices.append(float(row[1]))
			else:
				break
	return
def predict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n x 1

	svr_lin = SVR(kernel='linear', C=1e3)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_lin.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='Data') # plotting of initial data points
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model') # plotting the line made by the RBF kernel
	plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model') # plotting the line made by the linear kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]

get_data('aapl.csv')
predicted_price = predict_prices(dates, prices, 29)
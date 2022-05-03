from cgi import test
import functions as f

ticker = input('Enter a ticker: ')
start_time = input('Enter a starting period: ')
size = float(input('Enter a train size: '))
test_case_size = int(input('Enter a test case size: '))
print()

#f.test_values(ticker, start_time, size, test_case_size)

#Get the data
data, dataset= f.ticker_data(ticker, start_time)
train_size = int(dataset.shape[0] * size)

#Build the model
y_test, x_test, scaler, model = f.build_model(dataset, train_size, test_case_size)

#Prediciton
predictions = f.model_predictions(y_test, x_test, scaler, model)

f.Visualize(data, predictions, train_size)



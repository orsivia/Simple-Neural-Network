import NN, data_loader, perceptron

def tuning():
	training_data_c, test_data_c = data_loader.load_circle_data()
	training_data_m, test_data_m = data_loader.load_mnist_data()

	# five-fold cross validation partition
	len_c = len(training_data_c)
	len_m = len(training_data_m)
	training_c = []
	testing_c = []
	training_c.append(training_data_c[int(0.2*len_c):])
	testing_c.append(training_data_c[0:int(0.2*len_c)])
	training_c.append(training_data_c[0:int(0.2*len_c)]+training_data_c[int(0.4*len_c):])
	testing_c.append(training_data_c[int(0.2*len_c):int(0.4*len_c)])
	training_c.append(training_data_c[0:int(0.4*len_c)]+training_data_c[int(0.6*len_c):])
	testing_c.append(training_data_c[int(0.4*len_c):int(0.6*len_c)])
	training_c.append(training_data_c[0:int(0.6*len_c)]+training_data_c[int(0.8*len_c):])
	testing_c.append(training_data_c[int(0.6*len_c):int(0.8*len_c)])
	training_c.append(training_data_c[0:int(0.8*len_c)])
	testing_c.append(training_data_c[int(0.8*len_c):])

	training_m = []
	testing_m = []
	training_m.append(training_data_m[int(0.2*len_m):])
	testing_m.append(training_data_m[0:int(0.2*len_m)])

	training_m.append(training_data_m[0:int(0.2*len_m)]+training_data_m[int(0.4*len_m):])
	testing_m.append(training_data_m[int(0.2*len_m):int(0.4*len_m)])

	training_m.append(training_data_m[0:int(0.4*len_m)]+training_data_m[int(0.6*len_m):])
	testing_m.append(training_data_m[int(0.4*len_m):int(0.6*len_m)])

	training_m.append(training_data_m[0:int(0.6*len_m)]+training_data_m[int(0.8*len_m):])
	testing_m.append(training_data_m[int(0.6*len_m):int(0.8*len_m)])

	training_m.append(training_data_m[0:int(0.8*len_m)])
	testing_m.append(training_data_m[int(0.8*len_m):])
	# end partition

	domains = ['circles','mnist']
	activation_functions = ['relu','tanh']
	batch_sizes = [10,50,100]
	learning_rates = [0.1,0.01]
	hidden_layer_widths = [10,50]

	# tuning circles
	result_c = {}
	domain = domains[0]
	for a in range (len(batch_sizes)):
	    for b in range (len(learning_rates)):
		for c in range (len(activation_functions)):
		    for d in range (len(hidden_layer_widths)):
			net = NN.create_NN(20, domain, batch_sizes[a], learning_rates[b], activation_functions[c], hidden_layer_widths[d])
			acc = 0.0
			for numIter in range (5):
			    training_data = training_c[numIter]
			    testing_data = testing_c[numIter]
			    net.train(training_data)
			    acc = acc+net.evaluate(testing_data)
			acc = acc/5.0		    
			result_c[(batch_sizes[a],learning_rates[b],activation_functions[c],hidden_layer_widths[d])] = acc
	
	parameters_c = max(result_c.iterkeys(), key=(lambda key:result_c[key]))
	for key,val in result_c.items():
	    print '{k}: {v}%'.format(k=key, v=val)
	print '---- ---- ---- ----'
	print '(batch_size, learning_rate, activation_function, hidden_layer_width)'
	print 'Best parameters for circles dataset: {p}'.format(p=parameters_c)
	
	print '#----#----#----#----#'
	print '#----#----#----#----#'
	    
	# tuning mnist
	result_m = {}
	domain = domains[1]
	for a in range (len(batch_sizes)):
	    for b in range (len(learning_rates)):
		for c in range (len(activation_functions)):
		    for d in range (len(hidden_layer_widths)):
			net = NN.create_NN(20, domain, batch_sizes[a], learning_rates[b], activation_functions[c], hidden_layer_widths[d])
			acc = 0.0
			for numIter in range (5):
			    training_data = training_m[numIter]
			    testing_data = testing_m[numIter]
			    net.train(training_data)
			    acc = acc+net.evaluate(testing_data)
			acc = acc/5.0		    
			result_m[(batch_sizes[a],learning_rates[b],activation_functions[c],hidden_layer_widths[d])] = acc
	
	parameters_m = max(result_m.iterkeys(), key=(lambda key:result_m[key]))
	for key,val in result_m.items():
	    print '{k}: {v}%'.format(k=key, v=val)
	print '---- ---- ---- ----'
	print '(batch_size, learning_rate, activation_function, hidden_layer_width)'
	print 'Best parameters for mnist dataset: {p}'.format(p=parameters_m)
	
	return parameters_c, parameters_m





'''
net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
print net.train(training_data)
print net.evaluate(test_data)
print('--------')
net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
print net.train_with_learning_curve(training_data)
print net.evaluate(test_data)
print('--------')
perc = perceptron.Perceptron(data_dim)
print perc.train(training_data)
print perc.evaluate(test_data)
print('--------')
perc = perceptron.Perceptron(data_dim)
print perc.train_with_learning_curve(training_data)
print perc.evaluate(test_data)
'''


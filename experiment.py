from tuning import tuning
import NN, data_loader, perceptron
import matplotlib.pyplot as plt

parameters_c, parameters_m = tuning()

training_data_c, test_data_c = data_loader.load_circle_data()
training_data_m, test_data_m = data_loader.load_mnist_data()
data_dim_c = len(training_data_c[0][0])
data_dim_m = len(training_data_m[0][0])

net_c = NN.create_NN(100, 'circles', parameters_c[0], parameters_c[1], parameters_c[2], parameters_c[3])
#net_c = NN.create_NN('circles', 50, 0.1, 'relu', 50)
acc_net_c = net_c.train_with_learning_curve(training_data_c)
perc_c = perceptron.Perceptron(data_dim_c)
acc_perc_c = perc_c.train_with_learning_curve(training_data_c)

plt.figure()
plt.title('Learning curve: perceptron VS NN, circles dataset')
plt.xlabel('training steps')
plt.ylabel('accuracy(%)')
#plt.xlim((0,30))
plt.ylim((40,120))
line1_c, = plt.plot([t[0] for t in acc_net_c], [t[1] for t in acc_net_c], color='red', label='NN')
line2_c, = plt.plot([t[0] for t in acc_perc_c], [100*t[1] for t in acc_perc_c], color='blue', label='Perc')
plt.legend(handles=[line1_c, line2_c], loc = 2)

net_m = NN.create_NN(100, 'mnist', parameters_m[0], parameters_m[1], parameters_m[2], parameters_m[3])
acc_net_m = net_m.train_with_learning_curve(training_data_m)
perc_m = perceptron.Perceptron(data_dim_m)
acc_perc_m = perc_m.train_with_learning_curve(training_data_m)
plt.figure()
plt.title('Learning curve: perceptron VS NN, mnist dataset')
plt.xlabel('training steps')
plt.ylabel('accuracy(%)')
#plt.xlim((0,30))
plt.ylim((40,120))
line1_m, = plt.plot([t[0] for t in acc_net_m], [t[1] for t in acc_net_m], color='red', label='NN')
line2_m, = plt.plot([t[0] for t in acc_perc_m], [100*t[1] for t in acc_perc_m], color='blue', label='Perc')
plt.legend(handles=[line1_m, line2_m], loc = 2)

plt.show()

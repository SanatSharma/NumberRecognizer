import mnist_loader as load
import mnist_implementation as implement

training_data, validation_data, test_data = load.load_wrapper()
net = implement.Mnist_Network([784,30,10])
net.train(training_data, 30,10,3.0, test_data = test_data)
import numpy as np
from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import validation_curve
from tqdm import tqdm
from random import randint
from numba import cuda
import math

def load_batch(file):
    """ Copied from: https://www.cs.toronto.edu/~kriz/cifar.html """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data']    # Data (3072x10000)
        Y = np.array(dict[b'labels']) # Labels as numbers
        encoded_Y = np.zeros((len(Y), np.max(Y)+1), dtype=int) 
        encoded_Y[np.arange(len(Y)), Y] = 1 # This gives each row 0 for all elements execpt for the number in each row in Y. Ex. Y = 3, encoded_Y = [0,0,0,1,0,0,0,0,0,0]
    return np.transpose(X), np.transpose(Y), np.transpose(encoded_Y) 

def montage(images, rows=2, cols=5):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(rows,cols)
	for i in range(rows):
		for j in range(cols):
			im  = images[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def normalize_data(train_data, validation_data, test_data):
    std_data = np.std(train_data, axis=0)
    mean_data = np.mean(train_data, axis=0)
    train_data = train_data - mean_data
    train_data = train_data / std_data
    validation_data = validation_data - mean_data
    validation_data = validation_data / std_data
    test_data = test_data - mean_data
    test_data = test_data / std_data
    return train_data, validation_data, test_data

def init_params(size, mean=0, std_dev=0.01):
    W = np.random.normal(mean, std_dev, size)
    b = np.zeros((size[0],1))
    return W,b

def evaluate_classifier(X, W1, W2, b1, b2):
    s1 = np.matmul(W1, X) + b1               # Change W to use weight layer 1
    s1[s1 <= 0] = 0
    s2 = np.matmul(W2, s1) + b2               # Change W to use weight layer 2
    P = np.exp(s2)/np.sum(np.exp(s2), axis=0)
    return P, s1

def compute_cost(X, Y, W1, W2, b1, b2, lambda_val):
    P, h = evaluate_classifier(X, W1, W2, b1, b2)
    D = X.shape[1]
    regularization_term = lambda_val*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    cross_loss = np.trace(-np.matmul(np.log(P),np.transpose(Y)))
    return (cross_loss/D) + regularization_term

def compute_accuracy(X, y, W1, W2, b1, b2):
    y_pred = np.argmax(evaluate_classifier(X, W1, W2, b1, b2)[0], axis=0)
    return accuracy_score(y, y_pred)

def compute_gradients(X, Y, W1, W2, b1, b2, lambda_val):
    P, h = evaluate_classifier(X, W1, W2, b1, b2)
    batch_gradient = -(Y - P)
    D = X.shape[1]
    grad_W2 = np.add(np.dot(batch_gradient,np.transpose(h))/D, 2 * lambda_val * W2)
    grad_b2 = np.dot(batch_gradient,np.ones((h.shape[1],1)))/D
    batch_gradient = np.dot(np.transpose(W2), batch_gradient)
    batch_gradient = np.multiply(batch_gradient, h>0)
    grad_W1 = np.add(np.dot(batch_gradient, np.transpose(X))/D, 2*lambda_val*W1)
    grad_b1 = np.dot(batch_gradient, np.ones((X.shape[1],1)))/D
    


    return grad_W1, grad_W2, grad_b1, grad_b2

def compute_grads_num(X, Y, W1, W2, b1, b2, lambda_val, h=10**(-5)):
    c = compute_cost(X, Y, W1, W2, b1, b2, lambda_val)

    grad_b1 = np.zeros(b1.shape)
    for i in range(grad_b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i,0] += h
        c2 = compute_cost(X, Y, W1, W2, b1_try, b2, lambda_val)
        grad_b1[i] = (c2-c) / h

    grad_b2 = np.zeros(b2.shape)
    for i in range(grad_b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i,0] += h
        c2 = compute_cost(X, Y, W1, W2, b1, b2_try, lambda_val)
        grad_b2[i] = (c2-c) / h

    grad_W1 = np.zeros(W1.shape)
    for i in range(grad_W1.shape[0]):
        for j in range(grad_W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i,j] += h
            c2 = compute_cost(X, Y, W1_try, W2, b1, b2, lambda_val)
            grad_W1[i,j] = (c2-c) / h

    grad_W2 = np.zeros(W2.shape)
    for i in range(grad_W2.shape[0]):
        for j in range(grad_W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i,j] += h
            c2 = compute_cost(X, Y, W1, W2_try, b1, b2, lambda_val)
            grad_W2[i,j] = (c2-c) / h
    return grad_W1, grad_W2, grad_b1, grad_b2

def compare_computed_gradients(ga, gn, eps=10**(-5)):
    # Default eps value comes from the Standford’s course Convolutional Neural Networks for Visual Recognition recommendation 
    # https://cs231n.github.io/neural-networks-3/#gradcheck
    relative_error = np.abs(ga - gn).sum()
    denom = max(eps, np.abs(ga + gn).sum())
    return relative_error/denom < eps, relative_error, denom

def gradient_descent(X, Y, y, X_val, Y_val, y_val, n_batch, eta, n_epochs, W1, W2, b1, b2, lambda_val, eta_min=1e-5, eta_max=1e-1, n_s=500, allow_flipping=False, step_decay=False, use_CLR=False):
    total_length = X.shape[1]
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    eta_evolution = []
    k_list = list(range(0, total_length, n_batch))

    for epoch in tqdm(range(n_epochs)):
        
        # From 0 to the total length of the data set. n_batch makes "batch" increase by n_batch (batch size) for each iteration.
        for batch in range(0, total_length, n_batch):
            max_batch_idx = batch+n_batch
            Y_batch = Y[:,batch:max_batch_idx]
            X_batch = X[:,batch:max_batch_idx]
            if allow_flipping:
                if randint(0,1) > .5:
                    X_batch = np.flip(X_batch, axis=0)
            grad_W1, grad_W2, grad_b1, grad_b2 = compute_gradients(X_batch, Y_batch, W1, W2, b1, b2, lambda_val)
            W1 -= eta*grad_W1
            b1 -= eta*grad_b1
            W2 -= eta*grad_W2
            b2 -= eta*grad_b2

            """ CLR """
            if use_CLR:
                t = k_list.index(batch) + (epoch*n_batch)
                eta = CLR(t, eta_min, eta_max, n_s)
                eta_evolution.append(eta)

                pass

        """ Weight decay """
        if epoch%10 == 0 and epoch!=0 and step_decay:
            eta = .1*eta
            eta_min = .1*eta_min
            eta_max = .1*eta_max
            tqdm.write('Step decay performed.. eta is now {}'.format(eta)) 
                  
        training_accuracy.append(compute_accuracy(X, y, W1, W2, b1, b2))
        training_loss.append(compute_cost(X, Y, W1, W2, b1, b2, lambda_val))
        validation_accuracy.append(compute_accuracy(X_val, y_val, W1, W2, b1, b2))
        validation_loss.append(compute_cost(X_val, Y_val, W1, W2, b1, b2, lambda_val)) 
    return W1, W2, b1, b2, training_accuracy, training_loss, validation_accuracy, validation_loss, eta_evolution

def plot_training_validation(training, validation=None, labels=['Train', 'Validation'], plot_accuracy=True, save_file=False, file_name=''):
    if plot_accuracy:
        plotting = ' Accuracy: '
    else:
        plotting = ' Loss: '

    plt.plot(training, color='red', label=str(labels[0]) + str(plotting) + str(training[-1]))
    if validation != None:
        plt.plot(validation, color='blue', label=str(labels[1]) + str(plotting) + str(validation[-1]))
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(plotting)
    if save_file:
        plt.savefig(file_name)
    plt.show()

def CLR(t, eta_min, eta_max, n_s):
    l = math.floor(t/(2*n_s))

    if t >= 2*n_s*l and t < (2*l + 1)*n_s:
        eta = eta_min + ((t - 2*l*n_s)/n_s)*(eta_max - eta_min)

    elif t >= (2*l + 1)*n_s and t < 2*(l + 1)*n_s:
        eta = eta_max - ((t - (2*l + 1)*n_s)/n_s)*(eta_max - eta_min)
    return eta

if __name__ == '__main__':

    # Variables used to easy decide what to run
    run_grad_test = False
    run_sanity_check = False
    run_CLR_test = False
    run_ex_4 = True

    # Index for test set in the list of data sets 
    test_idx = 0
    # Add all datasets and test set urls for easy access later
    # The training set is at idx 0, so that the indexes correspond to the batch number for the other data sets
    datasets = ['../Datasets/cifar-10-batches-py/test_batch',
                '../Datasets/cifar-10-batches-py/data_batch_1',
                '../Datasets/cifar-10-batches-py/data_batch_2',
                '../Datasets/cifar-10-batches-py/data_batch_3',
                '../Datasets/cifar-10-batches-py/data_batch_4',
                '../Datasets/cifar-10-batches-py/data_batch_5'
                ]

    # Extract the input and ouput data, as well as the one hot encoded output from each data set.
    training_x, training_y, training_encoded = load_batch(datasets[1])
    validation_x, validation_y, validation_encoded = load_batch(datasets[2])    
    test_x, test_y, test_encoded = load_batch(datasets[test_idx])

    # Normalize the data
    training_x_norm, validation_x_norm, test_x_norm = normalize_data(training_x, validation_x, test_x)
    # Number of hidden nodes
    m = 50
    # Generate weights and biases for the first layer
    W1, b1 = init_params((m, training_x.shape[0]), std_dev=1/np.sqrt(training_x.shape[0]))
    
    # Generate weights and biases for the second layer
    W2, b2 = init_params((10, m), std_dev=1/np.sqrt(m))
    # Create lists containing the weights and biases

    ### Testing gradient calculations ###
    if run_grad_test:
        grads_x = training_x_norm[:20,:10]
        grads_encoded = training_encoded[:,:10]

        _, _ = evaluate_classifier(grads_x, W1[:,:20], W2, b1, b2)

        grads_W1, grads_W2, grads_b1, grads_b2 = compute_gradients(grads_x, grads_encoded, W1[:,:20], W2, b1, b2, 0)
        print('Finished computing analytical')
        grads_num_W1, grads_num_W2, grads_num_b1, grads_num_b2 = compute_grads_num(grads_x, grads_encoded, W1[:,:20], W2, b1, b2, 0)
        print('Finished computing numerical')
        grads_W1_test, rel_err1, denom1 = compare_computed_gradients(grads_num_W1, grads_W1)
        print('Comparison 1 complete')
        grads_W2_test, rel_err2, denom2 = compare_computed_gradients(grads_num_W2, grads_W2)
        print('Comparison 2 complete')
        grads_b1_test, rel_err3, denom3 = compare_computed_gradients(grads_num_b1, grads_b1)
        print('Comparison 3 complete')
        grads_b2_test, rel_err4, denom4 = compare_computed_gradients(grads_num_b2, grads_b2)
        print('Comparison 4 complete')  

        print('Grads test on W1 returned: {}, with relative error: {} and denom: {}'.format(grads_W1_test, rel_err1, denom1))
        print('Grads test on W2 returned: {}, with relative error: {} and denom: {}'.format(grads_W2_test, rel_err2, denom2))
        print('Grads test on b1 returned: {}, with relative error: {} and denom: {}'.format(grads_b1_test, rel_err3, denom3))
        print('Grads test on b2 returned: {}, with relative error: {} and denom: {}'.format(grads_b2_test, rel_err4, denom4))
        
        print(rel_err1/denom1)
        print(rel_err2/denom2)
        print(rel_err3/denom3)
        print(rel_err4/denom4)

    ### Sanity check ###
    if run_sanity_check:
        W1, W2, b1, b2, training_accuracy, training_loss, validation_accuracy, validation_loss, eta_evolution = gradient_descent(training_x_norm[:, :100],
                                                                                                                training_encoded[:, :100], 
                                                                                                                training_y[:100],
                                                                                                                validation_x_norm[:, :100],
                                                                                                                validation_encoded[:, :100],
                                                                                                                validation_y[:100],
                                                                                                                n_batch=100,
                                                                                                                eta=1e-5,
                                                                                                                n_epochs=200,
                                                                                                                W1=W1,
                                                                                                                W2=W2,
                                                                                                                b1=b1,
                                                                                                                b2=b2,
                                                                                                                lambda_val=0.01
                                                                                                                )
        
        
        plot_training_validation(training_accuracy, 
                                            validation_accuracy, 
                                            plot_accuracy=True,
                                            save_file=True,
                                            file_name='CLR_acc.png')
        
        plot_training_validation(training_loss, 
                                            validation_loss, 
                                            plot_accuracy=False,
                                            save_file=True,
                                            file_name='CLR_val.png')
    
    
    if run_CLR_test:
        W1, W2, b1, b2, training_accuracy, training_loss, validation_accuracy, validation_loss, eta_evolution = gradient_descent(training_x_norm[:, :],
                                                                                                                training_encoded[:, :], 
                                                                                                                training_y[:],
                                                                                                                validation_x_norm[:, :],
                                                                                                                validation_encoded[:, :],
                                                                                                                validation_y[:],
                                                                                                                n_batch=100,
                                                                                                                eta=1e-5,
                                                                                                                n_epochs=10,
                                                                                                                W1=W1,
                                                                                                                W2=W2,
                                                                                                                b1=b1,
                                                                                                                b2=b2,
                                                                                                                lambda_val=0.01,
                                                                                                                eta_min=1e-5,
                                                                                                                eta_max=1e-1,
                                                                                                                n_s=500,
                                                                                                                use_CLR=True
                                                                                                                )
        
        
        plot_training_validation(training_accuracy, 
                                            validation_accuracy, 
                                            plot_accuracy=True,
                                            save_file=True,
                                            file_name='CLR_acc.png')
        
        plot_training_validation(training_loss, 
                                            validation_loss, 
                                            plot_accuracy=False,
                                            save_file=True,
                                            file_name='CLR_val.png')

        plt.plot(eta_evolution, color='blue', label='ETA over time')
        plt.legend()
        plt.xlabel("Update steps")
        plt.ylabel('ETA')
        plt.savefig('CLR_ETA.png')
        plt.show()
    
    if run_ex_4:
        n_s = 800
        cycles = 3

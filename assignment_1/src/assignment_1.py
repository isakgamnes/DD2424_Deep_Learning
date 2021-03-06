import numpy as np
from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from random import randint

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

def normalize_data(data):
    std_data = np.std(data, axis=0)
    data = data - np.mean(data, axis=0)
    data = data / std_data
    return data

def init_params(size, mean=0, std_dev=0.01):
    W = np.random.normal(mean, std_dev, size)
    b = np.random.normal(mean, std_dev, (size[0],1))
    return W,b

def evaluate_classifier(X, W, b):
    s = np.dot(W, X) + b
    return np.exp(s)/np.sum(np.exp(s), axis=0)

def compute_cost(X, Y, W, b, lambda_val):
    P = evaluate_classifier(X, W, b)
    regularization_term = lambda_val*np.sum(np.square(W))
    D = X.shape[1]
    cross_loss = np.trace(-np.matmul(np.log(P),np.transpose(Y)))
    return 1/D * cross_loss + regularization_term

def compute_accuracy(X, y, W, b):
    y_pred = np.argmax(evaluate_classifier(X, W, b), axis=0)
    return accuracy_score(y, y_pred)

def compute_gradients(X, Y, W, b, lambda_val):
    P = evaluate_classifier(X, W, b)
    batch_gradient = P - Y
    D = X.shape[1]
    grad_W = np.add(np.dot(batch_gradient,np.transpose(X)) / D , 2 * lambda_val * W)
    grad_b = np.dot(batch_gradient,np.ones((D,1))) / D
    return grad_W, grad_b

def compute_grads_num(X, Y, W, b, lambda_val, h=10**(-6)):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = compute_cost(X, Y, W, b, lambda_val)
    
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lambda_val)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2 = compute_cost(X, Y, W_try, b, lambda_val)
            grad_W[i,j] = (c2-c) / h
    return grad_W, grad_b

def compare_computed_gradients(ga, gn, eps=10**(-7)):
    # Default eps value comes from the Standford???s course Convolutional Neural Networks for Visual Recognition recommendation 
    # https://cs231n.github.io/neural-networks-3/#gradcheck
    relative_error = np.abs(ga - gn).sum()
    denom = max(eps, np.abs(ga + gn).sum())
    if relative_error/denom < eps:
        return True
    return False

def gradient_descent(X, Y, y, X_val, Y_val, y_val, n_batch, eta, n_epochs, W, b, lambda_val, allow_flipping=False, step_decay=False):
    total_length = X.shape[1]
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    
    for epoch in tqdm(range(n_epochs)):
        # From 0 to the total length of the data set. n_batch makes "batch" increase by n_batch (batch size) for each iteration.
        for batch in range(0, total_length, n_batch):
            max_batch_idx = batch+n_batch
            X_batch = X[:,batch:max_batch_idx]
            if allow_flipping:
                if randint(0,1) > .5:
                    X_batch = np.flip(X_batch, axis=0)
            grad_W, grad_b = compute_gradients(X_batch, Y[:,batch:max_batch_idx], W, b, lambda_val)
            W -= eta*grad_W
            b -= eta*grad_b

        """ Weight decay """
        if epoch%10 == 0 and epoch!=0 and step_decay:
            eta = .1*eta
            tqdm.write('Step decay performed.. eta is now {}'.format(eta)) 
                  
        training_accuracy.append(compute_accuracy(X, y, W, b))
        training_loss.append(compute_cost(X, Y, W, b, lambda_val))
        validation_accuracy.append(compute_accuracy(X_val, y_val, W, b))
        validation_loss.append(compute_cost(X_val, Y_val, W, b, lambda_val)) 
    return W, b, training_accuracy, training_loss, validation_accuracy, validation_loss

def plot_training_validation(training, validation=None, labels=['Train', 'Validation'], plot_accuracy=True, save_file=False, file_name=''):
    if plot_accuracy:
        plotting = 'Accuracy'
    else:
        plotting = 'Loss'

    plt.plot(training, color='red', label=str(labels[0]) + str(plotting))
    if validation != None:
        plt.plot(validation, color='blue', label=str(labels[1]) + str(plotting))
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(plotting)
    if save_file:
        plt.savefig(file_name)
    plt.show()


if __name__ == '__main__':
    datasets = ['../Datasets/cifar-10-batches-py/data_batch_1',
                '../Datasets/cifar-10-batches-py/data_batch_2',
                '../Datasets/cifar-10-batches-py/data_batch_3',
                '../Datasets/cifar-10-batches-py/data_batch_4',
                '../Datasets/cifar-10-batches-py/data_batch_5'
                ]

    train_x1, train_Y1, train_encoded_Y1 = load_batch(datasets[0])
    train_x2, train_Y2, train_encoded_Y2 = load_batch(datasets[1])
    train_x3, train_Y3, train_encoded_Y3 = load_batch(datasets[2])
    train_x4, train_Y4, train_encoded_Y4 = load_batch(datasets[3])
    train_x5, train_Y5, train_encoded_Y5 = load_batch(datasets[4])

    train_x = np.concatenate((train_x1, train_x2, train_x3, train_x4, train_x5[:,:9000]), axis=1)                                             
    train_Y = np.concatenate((train_Y1, train_Y2, train_Y3, train_Y4, train_Y5[:9000]), axis=0)                                               
    train_encoded_Y = np.concatenate((train_encoded_Y1, train_encoded_Y2, train_encoded_Y3, train_encoded_Y4, train_encoded_Y5[:,:9000]), axis=1)     

    validation_x = train_x5[:,9000:]
    validation_Y = train_Y5[9000:]
    validation_encoded_Y = train_encoded_Y5[:,9000:]

    train_normalized = normalize_data(train_x)
    validation_normalized = normalize_data(validation_x)

    W, b = init_params((10,3072))
    print(W.shape, b.shape)

    lambda_vals = [1.]
    etas = [0.1]
    n_batches = [10]
    for lambda_val in lambda_vals:
        for eta in etas:
            for n_batch in n_batches:
                W, b, training_accuracy, training_loss, validation_accuracy, validation_loss = gradient_descent(train_normalized, 
                                                                                                                train_encoded_Y, 
                                                                                                                train_Y, 
                                                                                                                validation_normalized, 
                                                                                                                validation_encoded_Y, 
                                                                                                                validation_Y, 
                                                                                                                n_batch=n_batch, 
                                                                                                                eta=eta, 
                                                                                                                n_epochs=40, 
                                                                                                                W=W, 
                                                                                                                b=b, 
                                                                                                                lambda_val=lambda_val, 
                                                                                                                allow_flipping=False,
                                                                                                                step_decay=True)

                
                
                acc_file_name = '../figures_grid_search/accuracy_lb_{}_eta_{}_batch_{}.png'.format(str(eta).replace('.','dot'),str(lambda_val).replace('.','dot'),n_batch)
                loss_file_name = '../figures_grid_search/loss_lb_{}_eta_{}_batch_{}.png'.format(eta,lambda_val,n_batch)
                plot_training_validation(training_accuracy, 
                                        validation_accuracy, 
                                        plot_accuracy=True,
                                        save_file=True,
                                        file_name=acc_file_name)
                print('Saved accuracy plot as: {}'.format(acc_file_name))
                plot_training_validation(training_loss, 
                                        validation_loss, 
                                        plot_accuracy=False,
                                        save_file=True,
                                        file_name=loss_file_name)
                print('Saved loss plot as: {}'.format(loss_file_name))
                montage(W)
                _continue = input('Press enter to continue')
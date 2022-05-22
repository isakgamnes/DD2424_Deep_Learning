# Done
def normalize_data(data):
    std_data = np.std(data, axis=0)
    data = data - np.mean(data, axis=0)
    data = data / std_data
    return data

# Done
def init_params(layer_sizes, He=True, BN=True, mean=0, std_dev=0.01):
    n_layers = len(layer_sizes)
    W = []
    b = []
    betas = []
    gammas = []
    include_gamma_beta = False
    
    for layer in range(n_layers-1):
        # Use careful initialization
        if He:
            # He definition:
            # https://machinelearning.wtf/terms/he-initialization/#:~:text=He%20initialization%20initializes%20the%20bias%20vectors%20of%20a,l%20is%20the%20dimension%20of%20the%20previous%20layer.

            # Create random gaussian distribution with 0 mean and 1 variance
            layer_W = np.random.randn(layer_sizes[layer+1], layer_sizes[layer])
            # Mulitply by sqrt(2/nl) where nl is the size of a previous layer to get He init
            layer_W *= np.sqrt(2/layer_sizes[layer])

            # If batch norm, init b as zeros, else use He
            if BN:
                layer_b = np.zeros((layer_sizes[layer+1], 1))
            else:
                layer_b = np.random.randn(layer_sizes[layer+1], 1)
                layer_b *= np.sqrt(2/layer_sizes[layer])

        # Else, use old init
        else:
            W = np.random.normal(mean, std_dev, (layer_sizes[layer+1], layer_sizes[layer]))
            b = np.random.normal(mean, std_dev, (layer_sizes[layer+1],1))



        W.append(layer_W)
        b.append(layer_b)
        if include_gamma_beta:
            gammas.append(np.ones((layer_sizes[layer],1)))
            betas.append(np.zeros((layer_sizes[layer],1)))
        else:
            include_gamma_beta = True
    return W, b, gammas, betas

# Done
def evaluate_classifier(X, W, b, gamma, beta, BN):
    num_layers = len(W)
    # Initialize variables and allocate memory
    layer_X = [X.copy()] + [None]*(num_layers-1)
    S = [None]*(num_layers-1)
    S_hat = [None]*(num_layers-1)
    layer_means = [None]*(num_layers-1)
    layer_vars = [None]*(num_layers-1)

    for layer in range(num_layers-1):
        S[layer] = np.matmul(W[layer], layer_X[layer]) + b[layer]

        if BN:
            layer_means[layer] = S[layer].mean(axis=1).reshape(-1,1)
            layer_vars[layer] = S[layer].var(axis=1).reshape(-1,1)
            
            S_hat[layer] = (S[layer]-layer_means[layer])/(np.sqrt(layer_vars[layer]+1e-10))
            S_tilde = np.multiply(S_hat[layer], gamma[layer]) + beta[layer]
            S_tilde[S_tilde<0] = 0
            layer_X[layer+1] = S_tilde
    
    S_out = np.matmul(W[num_layers-1], layer_X[num_layers-1]) + b[num_layers-1]
    P = np.exp(S_out)/np.sum(np.exp(S_out), axis=0)
    return P, S, S_hat, layer_X[1:], layer_means, layer_vars

# Done
def compute_cost(X, Y, W, b, gamma, beta, lambda_val, BN):
    P, _, _, _, _, _ = evaluate_classifier(X, W, b, gamma, beta, BN)
    square_sum = [np.sum(np.square(w)) for w in W]
    regularization_term = lambda_val*np.sum(square_sum)
    D = X.shape[1]
    cross_loss = np.trace(-np.multiply(Y, np.log(P)))
    return (cross_loss/D) + regularization_term

# Done
def compute_accuracy(X, y, W, b, gamma, beta, BN):
    y_pred = np.argmax(evaluate_classifier(X, W, b, gamma, beta, BN)[0], axis=0)
    return accuracy_score(y, y_pred)

# Done
def compute_gradients(X, Y, W, b, gamma, beta, lambda_val, BN=True):
    P, S, S_hat, layer_X, layer_means, layer_vars = evaluate_classifier(X, W, b, gamma, beta, BN)
    k = len(W)
    N = X.shape[1]
    dJdW = [None]*k
    dJdB = [None]*k

    if BN:
        layer_X = [X.copy()] + layer_X
    else:
        layer_X = [X.copy()] + S

    # Propagate the gradient through the loss and softmax operations
    G_batch = -(Y - P)

    # The gradients of J w.r.t. bias vector bk and weight matrix Wk
    dJdW[k-1] = np.matmul(G_batch, np.transpose(layer_X[k-1]))/N + 2*lambda_val*W[k-1]
    dJdB[k-1] = np.matmul(G_batch, np.ones((N,1)))/N
    
    # Propagate G_batch to the previous layer
    G_batch = np.matmul(np.transpose(W[k-1]), G_batch)
    G_batch = np.multiply(G_batch, (layer_X[k-1] > 0))

    # Allocate memory for gamma and beta gradients
    dJdGamma = [None]*(k-1)
    dJdBeta = [None]*(k-1)

    # For layer = k − 2, k − 3, . . . , 0
    for layer in range(k-2, -1, -1):
        if BN:
            dJdGamma[layer] = np.matmul(np.multiply(G_batch, S_hat[layer]), np.ones((N,1)))/N
            dJdBeta[layer] = np.matmul(G_batch, np.ones((N,1)))/N
            G_batch_BN = np.multiply(G_batch, np.matmul(gamma[layer], np.ones((1,N))))
            G_batch = batch_norm_back_pass(G_batch_BN, S[layer], layer_means[layer], layer_vars[layer])
        
        dJdW[layer] = np.matmul(G_batch, np.transpose(layer_X[layer]))/N + 2*lambda_val*W[layer]
        dJdB[layer] = np.matmul(G_batch, np.ones((N,1)))/N

        if layer > 0:
            G_batch = np.matmul(np.transpose(W[layer]), G_batch)
            G_batch = np.multiply(G_batch, layer_X[layer] > 0)
    return dJdW, dJdB, dJdGamma, dJdBeta, layer_means, layer_vars

# Done
def compute_grads_num(X, Y, W, b, gamma, beta, lambda_val, BN=True, h=1e-5):
    k = len(W)

    dJdW = [None]*k
    dJdB = [None]*k
    dJdGamma = [None]*(k-1)
    dJdBeta = [None]*(k-1)
    
    for layer in range(len(W)):

        grad_b = np.zeros(b[layer].shape)
        
        for i in range(grad_b.shape[0]):
            b_temp = b.copy()
            b_try = b_temp[layer]
            b_try[i] += h
            b_temp[layer] = b_try
            c1 = compute_cost(X, Y, W, b_temp, gamma, beta, lambda_val, BN)
            
            b_temp = b.copy()
            b_try = b_temp[layer]
            b_try[i] -= h
            b_temp[layer] = b_try
            c2 = compute_cost(X, Y, W, b_temp, gamma, beta, lambda_val, BN)
            grad_b[i] = (c2-c1) / h
        dJdB[layer] = grad_b

        grad_W = np.zeros(W[layer].shape)
        for i in range(grad_W.shape[0]):
            for j in range(grad_W.shape[1]):
                W_temp = W.copy()
                W_try = W_temp[layer]
                W_try[i,j] += h
                W_temp[layer] = W_try
                c1 = compute_cost(X, Y, W_temp, b, gamma, beta, lambda_val, BN)
                
                W_temp = W.copy()
                W_try = W_temp[layer]
                W_try[i,j] -= h
                W_temp[layer] = W_try
                c2 = compute_cost(X, Y, W_temp, b, gamma, beta, lambda_val, BN)
                grad_W[i,j] = (c2-c1) / h
        dJdW[layer] = grad_W

    for layer in range(len(W)-1):
        grad_beta = np.zeros(beta[layer].shape)
        for i in range(grad_beta.shape[0]):
            beta_temp = beta.copy()
            beta_try = beta_temp[layer]
            beta_try[i] += h
            beta_temp[layer] = beta_try
            c1 = compute_cost(X, Y, W, b, gamma, beta_temp, lambda_val, BN)
            
            beta_temp = beta.copy()
            beta_try = beta_temp[layer]
            beta_try[i] -= h
            beta_temp[layer] = beta_try
            c2 = compute_cost(X, Y, W, b, gamma, beta_temp, lambda_val, BN)

            grad_beta[i] = (c2-c1) / h
        dJdBeta[layer] = grad_beta

        grad_gamma = np.zeros(gamma[layer].shape)
        for i in range(grad_gamma.shape[0]):
            gamma_temp = gamma.copy()
            gamma_try = gamma_temp[layer]
            gamma_try[i] += h
            gamma_temp[layer] = gamma_try
            c1 = compute_cost(X, Y, W, b, gamma_temp, beta, lambda_val, BN)
            
            gamma_temp = gamma.copy()
            gamma_try = gamma_temp[layer]
            gamma_try[i] -= h
            gamma_temp[layer] = gamma_try
            c2 = compute_cost(X, Y, W, b, gamma_temp, beta, lambda_val, BN)
            
            grad_gamma[i] = (c2-c1)/h
        dJdGamma[layer] = grad_gamma

    return dJdW, dJdB, dJdGamma, dJdBeta

# Done
def compare_computed_gradients(ga, gn, eps=1**(-7)):
    # Default eps value comes from the Standford’s course Convolutional Neural Networks for Visual Recognition recommendation 
    # https://cs231n.github.io/neural-networks-3/#gradcheck
    relative_error = [np.abs(ga_i - gn_i).sum() for ga_i, gn_i in zip(ga, gn)]
    denom = [max(eps, np.abs(ga_i + gn_i).sum()) for ga_i, gn_i in zip(ga, gn)]
    test = [relative_error_i/denom_i < eps for relative_error_i, denom_i in zip(relative_error, denom)]
    return test
    """if any(test):
        return True
    return False"""

# Done
def batch_norm_back_pass(G_batch, S_batch, mean, var):
    n = S_batch.shape[1]
    sigma_1 = (var+1e-10)**(-0.5)
    sigma_2 = (var+1e-10)**(-1.5)
    G1 = np.multiply(G_batch, np.matmul(sigma_1, np.ones((1,n))))
    G2 = np.multiply(G_batch, np.matmul(sigma_2, np.ones((1,n))))
    D = S_batch - np.matmul(mean, np.ones((1,n)))
    c = np.matmul(np.multiply(G2, D), np.ones((n,1)))

    return (G1 - np.matmul(G1, np.ones((n,1))))/n - np.multiply(D, np.matmul(c, np.ones((1,n))))/n

def gradient_descent(X, Y, y, X_val, Y_val, y_val, n_batch, eta, n_epochs, W, b, gamma, beta, lambda_val, alpha, eta_min=1e-5, eta_max=1e-1, n_s=500, step_decay=False, use_CLR=False, BN=True):
    total_length = X.shape[1]
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    eta_evolution = []
    k_list1 = list(range(0, n_s, 1))
    k_list2 = list(range(n_s, 0, -1))
    k_list = np.concatenate([k_list1, k_list2], axis=0)
    cycle_idx = 0

    for epoch in tqdm(range(n_epochs)):
        # From 0 to the total length of the data set. n_batch makes "batch" increase by n_batch (batch size) for each iteration.
        for batch in range(0, total_length, n_batch):
            max_batch_idx = batch+n_batch
            Y_batch = Y[:,batch:max_batch_idx]
            X_batch = X[:,batch:max_batch_idx]
            
            grad_W, grad_b, grad_gamma, grad_beta, layer_means, layer_vars = compute_gradients(X_batch, Y_batch, W, b, gamma, beta, lambda_val, BN)
            

            """ CLR """
            if use_CLR:
                t = k_list[cycle_idx]
                
                cycle_idx += 1
                if cycle_idx >= len(k_list):
                    cycle_idx = 0
                eta = CLR(t, eta_min, eta_max, n_s)
                eta_evolution.append(eta)
            
            try:
                W -= eta*np.array(grad_W)
                b -= eta*np.array(grad_b)
            except np.VisibleDeprecationWarning as e:
                a=1
            if BN:
                try:
                    gamma-=eta*np.array(grad_gamma)
                    beta-=eta*np.array(grad_beta)
                except np.VisibleDeprecationWarning as e:
                    a=2
                if batch == 0 and epoch == 0:
                    average_mean = layer_means
                    average_var = layer_vars
                else:
                    average_mean = [alpha*average_mean[l]+(1-alpha)*layer_means[l] for l in range(len(layer_means))]
                    average_var = [alpha*average_var[l]+(1-alpha)*layer_vars[l] for l in range(len(layer_vars))]

        """ Step decay """
        if epoch%10 == 0 and epoch!=0 and step_decay:
            eta = .1*eta
            eta_min = .1*eta_min
            eta_max = .1*eta_max
            tqdm.write('Step decay performed.. eta is now {}'.format(eta)) 
                  
        training_accuracy.append(compute_accuracy(X, y, W, b, gamma, beta, BN))
        training_loss.append(compute_cost(X, Y, W, b, gamma, beta, lambda_val, BN))
        validation_accuracy.append(compute_accuracy(X_val, y_val, W, b, gamma, beta, BN))
        validation_loss.append(compute_cost(X_val, Y_val, W, b, gamma, beta, lambda_val, BN)) 
    return W, b, gamma, beta, training_accuracy, training_loss, validation_accuracy, validation_loss, eta_evolution, average_mean, average_var

def CLR(t, eta_min, eta_max, n_s):
    l = math.floor(t/(2*n_s))

    if t >= 2*n_s*l and t < (2*l + 1)*n_s:
        eta = eta_min + ((t - 2*l*n_s)/n_s)*(eta_max - eta_min)/n_s

    elif t >= (2*l + 1)*n_s and t < 2*(l + 1)*n_s:
        eta = eta_max - ((t - (2*l + 1)*n_s)/n_s)*(eta_max - eta_min)/n_s
    return eta

def plot_training_validation(training, validation, plot_accuracy=True):
    if plot_accuracy:
        plotting = 'Accuracy'
    else:
        plotting = 'Loss'
    plt.plot(training, color='red', label='Train ' + plotting)
    plt.plot(validation, color='blue', label='Validation ' + plotting)
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(plotting)
    plt.show()
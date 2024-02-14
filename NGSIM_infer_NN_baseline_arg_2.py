


import sys
sys.path.insert(0, 'Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import pickle

if 1:
    np.random.seed(1234)
    tf.set_random_seed(1234)
else:
    np.random.seed(1234)
    tf.random.set_seed(1234)


# para1 = float((sys.argv[1]))  
# para2 = float((sys.argv[2]))
# para3 = float((sys.argv[3]))
# para4 = float((sys.argv[4]))
# para5 = int((sys.argv[5]))

# para1: weight of the data loss of density (rho)
# para2: weight of the data loss of velocity (u)
# para3: weight of the physics loss of desity
# para4: weight of the physics loss of velocity
# para5: pre-train length
para1, para2, para3, para4, para5, para6 = 10, 10, 0, 0, 5000, 16


print("parameters:", para1, para2, para3, para4, para5)

with open("US101_Lane1to5_t1.5s30.pickle",'rb') as f:
    data_pickle = pickle.load(f)

Num_of_LOOP = 14

Loop_dict = {3:	[0,7,20],\
    4:	[0,5,11,20],\
    5:	[0,4,8,13,20],\
    6:	[0,3,7,11,14,20],\
    7:	[0,3,6,9,12,15,20],\
    8:	[0,2,5,8,11,13,16,20],\
    9:	[0,2,4,7,9,12,14,17,20],\
    10:	[0,2,4,6,8,11,13,15,17,20],\
    11:	[0,2,4,6,8,10,12,14,16,18,20],\
    12:	[0,1,3,5,7,9,11,12,14,16,18,20],\
    13:	[0,1,3,5,6,8,10,11,13,15,16,18,20],\
    14:	[0,1,3,4,6,7,9,11,12,14,15,17,18,20],\
    15:	[0,1,2,4,5,7,8,10,11,13,14,16,17,19,20],\
    16:	[0,1,2,4,5,6,8,9,11,12,13,15,16,17,19,20],\
    18:	[0,1,2,3,4,6,7,8,9,11,12,13,14,15,17,18,19,20]}

if 1:
    Para_dict = {3:	[0.666666667, 13.1470821,  27.6],\
    4:[0.666666667, 	13.20616838, 	32.9],\
    5:[0.666666667, 	13.25222039, 	29.7],\
    6:[0.666666667, 	13.38781334, 	28.8],\
    7:[0.666666667, 	13.3973927, 	29.4],\
    8:[0.666666667, 	13.38967928, 	30.5],\
    9:[0.666666667, 	13.44997944, 	28.8],\
    10:[0.666666667, 	13.43056139, 	28.2],\
    11:[0.666666667, 	13.50359525, 	27.1],\
    12:[0.666666667, 	13.49605115, 	28.6],\
    13:[0.666666667, 	13.45704845, 	29.1],\
    14:[0.666666667, 	13.51057273, 	27.5],\
    15:[0.666666667, 	13.46840866, 	29.5],\
    16:[0.666666667, 	13.48429297, 	28.8],\
    18:[0.666666667, 	13.52053988, 	28.1]}
else:
    Para_dict = {3:[	0.475927875	,	17.27383125	,	27.6	],\
            4:[	0.468811167	,	17.51858601	,	32.9	],\
            5:[	0.464741287	,	17.77895198	,	29.7	],\
            6:[	0.46446889	,	17.8177041	,	28.8	],\
            7:[	0.455617006	,	18.17839975	,	29.4	],\
            8:[	0.458914869	,	18.05895764	,	30.5	],\
            9:[	0.45249452	,	18.35041852	,	28.8	],\
            10:[	0.45436323	,	18.28289647	,	28.2	],\
            11:[	0.450527574	,	18.48492336	,	27.1	],\
            12:[	0.454128303	,	18.32940351	,	28.6	],\
            13:[	0.454362407	,	18.31077979	,	29.1	],\
            14:[	0.451493967	,	18.45601689	,	27.5	],\
            15:[	0.453846411	,	18.34504919	,	29.5	],\
            16:[	0.451410843	,	18.45661909	,	28.8	],\
            18:[	0.450768173	,	18.49836423	,	28.1	]}

#10:[	0.45436323	,	18.28289647	,	28.2	],\

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, X3, rho, u, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
        
        
        self.x = X[:,0:1]
        #self.x2 = X2[0:1,:]#X2[:,0:1]
        self.t = X[:,1:2]
        self.rho = rho
        self.u = u
        
        #self.xl = X2[:,0:1]
        #self.xu = X2[:,0:1] + (1 + 1.0/239.0) # another boundary with additional gap at x=241
        #self.t2 = X2[:,1:2]
        
        self.x3 = X3[:,0:1]
        self.t3 = X3[:,1:2]
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize parameters
        rho_max, u_max, tau = Para_dict[Num_of_LOOP]
        Para1 = para1 
        #self.rho_max = tf.constant([rho_max], dtype=tf.float32) # rho_max
        #self.u_max = tf.constant([u_max], dtype=tf.float32) # u_max
        #self.tau = tf.constant([tau], dtype=tf.float32)
        
        self.rho_max = tf.Variable([rho_max], dtype=tf.float32) # rho_max
        self.u_max = tf.Variable([u_max], dtype=tf.float32) # u_max
        self.tau = tf.Variable([tau], dtype=tf.float32)
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]]) # self.x.shape[1] =1, None indicates that the first dimension, corresponding to the batch size, can be of any size
        #self.x_tf2 = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        self.x3_tf = tf.placeholder(tf.float32, shape=[None, self.x3.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        #self.t2_tf = tf.placeholder(tf.float32, shape=[None, self.t2.shape[1]])
        self.t3_tf = tf.placeholder(tf.float32, shape=[None, self.t3.shape[1]])
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        #self.x_tf_l = tf.placeholder(tf.float32, shape=[None, self.xl.shape[1]])
        #self.x_tf_u = tf.placeholder(tf.float32, shape=[None, self.xu.shape[1]])
        
        self.rho_pred, self.u_pred = self.net_rho_u(self.x_tf, self.t_tf)
        self.f_rho_pred, self.f_u_pred = self.net_f(self.x3_tf, self.t3_tf)
        
        
        self.resid_rho = tf.reduce_mean(tf.square(self.f_rho_pred))
        self.resid_u = tf.reduce_mean(tf.square(self.f_u_pred))
        
        self.loss = Para1*tf.reduce_mean(tf.square(self.rho_tf - self.rho_pred)) + para2*tf.reduce_mean(tf.square(self.u_tf - self.u_pred))\
                    + para3*tf.reduce_mean(tf.square(self.f_rho_pred)) + para4*tf.reduce_mean(tf.square(self.f_u_pred))
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,#Maximum number of iterations
                                                                           'maxfun': 50000, #Maximum number of function evaluations
                                                                           'maxcor': 50, # number of limited memory matric
                                                                           'maxls': 50, 
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer_Adam = tf.train.AdamOptimizer()#learning_rate = 0.0005)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer() # after this variables hold the values you told them to hold when you declare them will be made
        self.sess.run(init) # initialization run

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1): # need the -1 because the first and last number in defining the nn is input and output
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases): # NP
        
        num_layers = len(weights) + 1 # need +1, because # of weights is one dim smaller
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  # this is all element-wise operation, centralize and standardize the input data
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        #Y = tf.exp(tf.add(tf.matmul(H, W), b)) # the final layer has no activation func, so it has to be separetly processed
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
            
    def net_rho_u(self, x, t):  
        rho_and_u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases) # concatenate is needed for making [[x0,t0],[x1,t1],...]
        rho = rho_and_u[:,0:1]
        u = rho_and_u[:,1:2]
        return rho, u 
        
    
    def U_eq(self, rho):
        return self.u_max * (1 - rho/self.rho_max)
    
    def net_f(self, x, t): # physics-informed part
        
        rho, u = self.net_rho_u(x,t)
        U_eq = self.u_max * (1 - rho/self.rho_max)
        h = self.u_max * rho/self.rho_max
        
        
        ############# f_rho ##############
        rho_t = tf.gradients(rho, t)[0]
        rho_x = tf.gradients(rho, x)[0]
        rho_xx = tf.gradients(rho_x, x)[0]
        rho_time_u = rho * u
        rho_time_u_x = tf.gradients(rho_time_u, x)[0]
        f_rho =  rho_t + rho_time_u_x  - 0.001*rho_xx
        
        ############# f_u ##############
        u_h = u + h
        u_h_t = tf.gradients(u_h, t)[0]
        uu_h_x =  u * tf.gradients(u_h, x)[0]
        f_u =  self.tau * (u_h_t + uu_h_x) -  (U_eq - u)
        #f_u = (u_h_t + uu_h_x) - 10*(U_eq - u)
       
        return f_rho, f_u # the residuals for both rho and u
    
    
    def callback(self, loss, lambda_1, lambda_2, lambda_3):
        #print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)))
        print('Loss: %e, rho_max: %.6f, u_max: %.6f, tau: %.6f' % (loss, lambda_1, lambda_2, lambda_3))
    
    
        
    def train(self, nIter):
        #tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.rho_tf: self.rho,   self.x_tf_l: self.xl, self.x_tf_u: self.xu}
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t,\
         self.x3_tf: self.x3, self.t3_tf: self.t3,\
         self.rho_tf: self.rho, self.u_tf: self.u} # , self.x_tf_u: self.xu}
        
        start_time = time.time()
        for it in range(nIter): # why we need to train this??
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0: # adam training first.. WHY?
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                rhomax_value = self.sess.run(self.rho_max)
                u_max_value = self.sess.run(self.u_max)
                tau_value = self.sess.run(self.tau)
                
                print('It: %d, Loss: %.3e, rho_max: %.6f, u_max: %.6f, tau: %.6f' % 
                      (it, loss_value, rhomax_value, u_max_value, tau_value))
                start_time = time.time()
                
        start_time = time.time()
        if 1:
            self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.rho_max, self.u_max, self.tau], # fetch some variables to forward to loss_callback
                                loss_callback = self.callback) # take the value of loss, lambda1 and lambda2 out #this training will continue until epi exceed. for physics problem, maybe L-BFGS is better.
        
        elapsed = time.time() - start_time
        print('Time: %.2f' % (elapsed))

        
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1],  self.t_tf: X_star[:,1:2]}
        rho_star, u_star = self.sess.run([self.rho_pred, self.u_pred], tf_dict)
        
        return rho_star, u_star
    
    def predict_f(self, X_star):
        
        
        tf_dict = {self.x3_tf: X_star[:,0:1],  self.t3_tf: X_star[:,1:2]}
        rho_star, u_star = self.sess.run([self.f_rho_pred, self.f_u_pred], tf_dict)#self.sess.run([self.rho_pred, self.u_pred], tf_dict)
        
        
        return rho_star, u_star




if __name__ == "__main__": 
    
    
    N_u = 22000#3000
    
    
    layers = [2, 20, 20, 40, 80, 80, 40, 20, 20, 2]
    
    
    print(data_pickle.keys()) # dict_keys(['vMat', 'rhoMat', 'qMat', 'para_gsd_fitted_rhomax', 'para_gsd_predefined_rhomax', 't', 's'])
    print(len(data_pickle['rhoMat']), len(data_pickle['rhoMat'][20]))
    print(len(data_pickle['s']), len(data_pickle['t'])) # s:21 t:1770
    
    xx = np.array(data_pickle['s'])
    tt = np.array(data_pickle['t'])
    rhoMat = np.array([np.array(ele) for ele in data_pickle['rhoMat']])
    vMat = np.array([np.array(ele) for ele in data_pickle['vMat']])
    
    X, T = np.meshgrid(xx,tt)
    print(len(X),len(X[0]))# 21 by 1770
    
    N_u = int(len(X)*len(X[0])*0.8)
    
    x = X.flatten()[:,None]# 21*1770 by 1
    t = T.flatten()[:,None]# 21*1770 by 1
    Exact_rho = rhoMat.T # 1770 by 21
    Exact_v = vMat.T
    
    print(len(t), len(t[0]))
    print(len(x), len(x[0]))
    print(len(Exact_rho), len(Exact_rho[0]))
    print(len(Exact_v), len(Exact_v[0]))
    
    
    X_star = np.hstack((x, t)) # hstack is column wise stack, 21*1770 by 2
    rho_star = Exact_rho.flatten()[:,None] # not 21*1770 by 1 => 21*1770 by 1
    v_star = Exact_v.flatten()[:,None]
    
    
    print(len(X_star),len(X_star[0]), len(rho_star),len(v_star),len(v_star[0]))
    
    # Doman bounds
    lb = X_star.min(0) # [29.62194824 42.00757576]
    ub = X_star.max(0) # [622.06091311 2695.9862013 ]
    
    print(lb)
    print(ub)
    
    print(X_star.shape[0])
    
    N_loop = Loop_dict[Num_of_LOOP]
    print(N_loop)
    
    
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    noise = 0.0   
    idx = np.random.choice(X_star.shape[0], N_u, replace=False) # N_u = 22000 out of 37170 for Auxiliary points
    idx2 = []
    
    
    for i in range(1770): # for observations on the loops
        base = i*21
        index = [base + ele for ele in N_loop]
        idx2 += index
    
    
    
    print(len(rho_star))
    print(len(v_star))
    print(len(idx2), len(X_star))
    X_rho_train = X_star[idx2,:] # [x, 0.0] and 100 points from left bound selected
    X_rho_colocat = X_star[idx,:]
    rho_train = rho_star[idx2,:]
    v_train = v_star[idx2,:]
    #for ele in X_rho_train: print(ele)
    print(len(idx),len(idx2))
    
    model = PhysicsInformedNN(X_rho_train, X_rho_colocat, rho_train, v_train, layers, lb, ub)
    
    itera = para5
    
    model.train(itera)
    
    rho_pred, v_pred = model.predict(X_star)
    f_pred = model.predict_f(X_star)
    
    error_rho = np.linalg.norm(rho_star-rho_pred,2)/np.linalg.norm(rho_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    
    
    RHO_pred = griddata(X_star, rho_pred.flatten(), (X, T), method='cubic')
    RHO_org = griddata(X_star, Exact_rho.flatten(), (X, T), method='cubic')
    
    U_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    U_org = griddata(X_star, Exact_v.flatten(), (X, T), method='cubic')
    
    print('Error rho: %e' % (error_rho))    
    print('Error v: %e' % (error_v))    
    print(Num_of_LOOP)
    print(N_loop, itera)
    print("parameters:", para1, para2, para3, para4)
    
    
    '''
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    
    noise = 0.01        
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(10000)
    
    u_pred, f_pred = model.predict(X_star)
        
    lambda_1_value_noisy = model.sess.run(model.lambda_1)
    lambda_2_value_noisy = model.sess.run(model.lambda_2)
    #lambda_2_value_noisy = np.exp(lambda_2_value_noisy)
            
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu)/nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))                           
    '''
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    fig, ax = newfig(1.0, 1.4)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    #h = ax.imshow(RHO_pred.T, interpolation='nearest', cmap='rainbow', 
    h = ax.imshow(RHO_pred.T, interpolation='nearest', cmap='jet', vmin=0, vmax=0.37,
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    #ax.plot(X_rho_train[:,1], X_rho_train[:,0], 'kx', label = 'Data (%d points)' % (rho_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    #ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[320]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$rho(t,x)$', fontsize = 10)
    
    ############real##############
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0-0.06, bottom=0+0.06, left=0.15, right=0.85, wspace=0.0)
    ax = plt.subplot(gs2[:, :])
    
    #h = ax.imshow(RHO_org.T, interpolation='nearest', cmap='rainbow', 
    h = ax.imshow(RHO_org.T, interpolation='nearest', cmap='jet', vmin=0, vmax=0.37,
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    #ax.plot(X_rho_train[:,1], X_rho_train[:,0], 'kx', label = 'Data (%d points)' % (rho_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    #ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[320]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('exact $rho(t,x)$', fontsize = 10)
    
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.02, bottom=1.0-2.0/3.0+0.07, left=0.1, right=0.9, wspace=0.5)
    Exact = Exact_rho
    
    """
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[2,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,RHO_pred[2,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$rho(t,x)$')    
    ax.set_title('$t = 0.0781$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,0.95])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[5,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,RHO_pred[5,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$rho(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,0.95])
    ax.set_title('$t = 0.234$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[10,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,RHO_pred[10,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$rho(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,0.95])
    ax.set_title('$t = 1.0$', fontsize = 10)
    """
    
    
    
    '''
    ####### Row 3: Identified PDE ##################    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
    
    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(0.1,0.1,s)
       
    '''
    savefig('./figures/ARZ_RHO_inference')
    
    
    
    ######################################################################
    ############################# Plotting222 ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 1.4)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    #h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='jet', vmin=0, vmax=15,
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    #ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[320]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ############real##############
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0-0.06, bottom=0+0.06, left=0.15, right=0.85, wspace=0.0)
    ax = plt.subplot(gs2[:, :])
    
    #h = ax.imshow(U_org.T, interpolation='nearest', cmap='rainbow', 
    h = ax.imshow(U_org.T, interpolation='nearest', cmap='jet', vmin=0, vmax=15,
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    #ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    #ax.plot(t[320]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('exact $u(t,x)$', fontsize = 10)
    
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.02, bottom=1.0-2.0/3.0+0.07, left=0.1, right=0.9, wspace=0.5)
    Exact = Exact_v
    
    """
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[2,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[2,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.0781$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,1.01])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[6,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[6,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,1.01])
    ax.set_title('$t = 0.234$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[10,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[10,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.01,1.01])
    ax.set_title('$t = 1.0$', fontsize = 10)
    """
    
    savefig('./figures/ARZ_U_inference')
     
    




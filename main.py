import caco
from tqdm import tqdm
import numpy as np


# dim: number of vertices in graph
# size: sample size
# q: y values
# t_type: type of observation points, 'balanced' / 'unbalanced'
# m_type: type of model, one of ['Model I Tree', 'Model I Hub', 'Model II Tree', 'Model II Hub', 'Model III Tree', 'Model III Hub']
# k_type: type of kernel function, 'brownian' / 'rbf'
dim = 20
size = 300
q = [0.1, 0.25, 0.5, 0.75, 0.9]
t_type = 'balanced'
m_type = 'Model I Tree'
k_type = 'rbf'


# generate observe time points and samples
observe_point, g_t = caco.generate_t(size, t_type, 1200)
samples = caco.gen_sample(observe_point, 50, 'Model I Tree', dim, g_t, 'rbf')
X = samples[0]  # random functions X
Y = samples[1]  # covariate Y
E = samples[2]  # edge matrix E


# calculate intermidiate quantities
X_c = caco.cal_coor(samples[0], observe_point, t_type, g_t, 'rbf')
K_X = caco.kernel_x_mat(samples[0], X_c, observe_point, g_t, 'rbf')
K_2 = caco.kernel_x_inv(K_X)


# calculate norms of operators and plot the ROC curve
norm_y = list()
for i in tqdm(range(len(q))):
    K_Y, gamma = caco.kernel_y_mat(samples[1])
    precision_mat = caco.precision_operator_norm(np.quantile(Y, q[i]), K_X, K_2, K_Y, samples[1], gamma)
    norm_y.append(precision_mat)
caco.plot_roc(np.mean(np.asarray(norm_y), axis=0), samples[2])


# An easy way to make repeated simulations:
# Repeat 50 times when:
# dim = 10, sample size = 100, y = 0.1,
# samples are 'balanced', model = 'Model I Hub', kernel function is 'brownian'.
# Return 50 AUC values.
caco.repeated_exp(50, 100, 10, 0.1, 'balanced', 'Model I Hub', 'brownian')

# Considering computational time, we suggest dim and size not be too large. 








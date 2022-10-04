import math
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.linalg import sqrtm


def brownian_kernel(s, t):
    return min(s, t)


def rbf_kernel(s, t, g):
    return math.exp(0 - g * (s - t) * (s - t))


def kernel_t_mat(t_1, t_2, g, kernel_type):
    if kernel_type == 'brownian':
        l_1 = len(t_1)
        l_2 = len(t_2)
        out = list()
        for i in range(l_1):
            out_i = list()
            for j in range(l_2):
                out_i.append(brownian_kernel(t_1[i], t_2[j]))
            out.append(out_i)
        out = np.asmatrix(out)
        return out
    elif kernel_type == 'rbf':
        l_1 = len(t_1)
        l_2 = len(t_2)
        out = list()
        for i in range(l_1):
            out_i = list()
            for j in range(l_2):
                out_i.append(rbf_kernel(t_1[i], t_2[j], g))
            out.append(out_i)
        out = np.asmatrix(out)
        return out


def generate_t(n, data_type, length):
    if data_type == 'unbalanced':
        out = list()
        for i in range(n):
            pool = np.linspace(0.01, 1, 100, endpoint=True)
            out_i = np.random.choice(pool, 50, replace=False)
            out_i.sort()
            out.append(out_i)
    elif data_type == 'balanced':
        out = np.linspace(0, length/50, length, endpoint=False)
        out = [out] * n
    else:
        print('data_type should be balanced or unbalanced!')

    g = 0
    temp = np.unique(np.concatenate(out))
    N = len(temp)
    for i in range(N):
        for j in range(i, N):
            g = g + abs(temp[i] - temp[j])
    g = 2 / (N * (N - 1)) * g
    return [out, g]


def gen_sample(t, m, graph_type, dim, g_t, kernel_type):
    n = len(t)
    ot = np.random.rand(m)
    k = list()
    for i in range(n):
        k.append(kernel_t_mat(t[i], ot, g_t, kernel_type))

    y = np.random.rand(n)
    j = m

    if graph_type == 'Model I Tree':
        out = list()
        cov = list()
        for i in range(n):
            y_i = y[i]
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                    if i == 0:
                        cov_l = [1] + [0] * (dim - 1)
                else:
                    parent = int((l + 1) / 2) - 1
                    xi = np.asmatrix(np.random.randn(j)).T
                    p_node = np.multiply(x[:, parent], x[:, parent])
                    if l % 2 == 0:
                        x_l = np.matmul(k[i], xi) + (1 - math.sqrt(y_i)) * p_node
                    else:
                        x_l = np.matmul(k[i], xi) + math.sqrt(y_i) * p_node
                    x = np.append(x, x_l, axis=1)
                    if i == 0:
                        cov_l = [0] * dim
                        cov_l[parent] = 1
                        cov_l[l] = 1
                if i == 0:
                    cov.append(cov_l)
            out.append(x)
        cov = np.asarray(cov)
        cov = cov + cov.T - np.identity(dim, dtype=int)
        return[out, y, cov]
    elif graph_type == 'Model I Hub':
        out = list()
        for i in range(n):
            y_i = y[i]
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                elif l % 5 == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x_l = np.matmul(k[i], xi)
                    x = np.append(x, x_l, axis=1)
                else:
                    parent = int(l/5) * 5
                    xi = np.asmatrix(np.random.randn(j)).T
                    p_node = np.multiply(1 + x[:, parent], 1 + x[:, parent])
                    if l % 2 == 0:
                        x_l = 0.25 * np.matmul(k[i], xi) + (1 - math.sqrt(y_i)) * p_node
                    else:
                        x_l = 0.25 * np.matmul(k[i], xi) + math.sqrt(y_i) * p_node
                    x = np.append(x, x_l, axis=1)
            out.append(x)
        block = np.asarray([[1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1]])
        cov = block
        q = int(dim/5)
        for i in range(q - 1):
            cov = np.append(cov, np.zeros((5 * (i + 1), 5), dtype=int), axis=1)
            temp = np.append(np.zeros((5, 5 * (i + 1)), dtype=int), block, axis=1)
            cov = np.append(cov, temp, axis=0)
        r = dim - q * 5
        if r == 0:
            return [out, y, cov]
        else:
            cov = np.append(cov, np.zeros((5 * q, r), dtype=int), axis=1)
            if r == 1:
                temp = np.append(np.zeros((r, 5 * q), dtype=int), np.identity(1, dtype=int))
            else:
                temp = np.identity(r, dtype=int)
                temp[0, :] = np.ones((1, r), dtype=int)
                temp[:, 0] = np.ones((1, r), dtype=int)
                temp = np.append(np.zeros((r, 5 * q), dtype=int), temp, axis=1)
            cov = np.append(cov, temp, axis=0)
            return [out, y, cov]
    elif graph_type == 'Model II Tree':
        out = list()
        cov = list()
        for i in range(n):
            y_i = y[i]
            one = np.asmatrix(np.ones(len(t[0]))).T
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                    if i == 0:
                        cov_l = [1] + [0] * (dim - 1)
                else:
                    parent = int((l + 1) / 2) - 1
                    xi = np.asmatrix(np.random.randn(j)).T
                    if l % 2 == 0:
                        x_l = np.multiply(np.matmul(k[i], xi), one + (1 - math.sqrt(y_i)) * x[:, parent])
                    else:
                        x_l = np.multiply(np.matmul(k[i], xi), one + math.sqrt(y_i) * x[:, parent])
                    x = np.append(x, x_l, axis=1)
                    if i == 0:
                        cov_l = [0] * dim
                        cov_l[parent] = 1
                        cov_l[l] = 1
                if i == 0:
                    cov.append(cov_l)
            out.append(x)
        cov = np.asarray(cov)
        cov = cov + cov.T - np.identity(dim, dtype=int)
        return[out, y, cov]
    elif graph_type == 'Model II Hub':
        out = list()
        for i in range(n):
            y_i = y[i]
            one = np.asmatrix(np.ones(len(t[i]))).T
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                elif l % 5 == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x_l = np.matmul(k[i], xi)
                    x = np.append(x, x_l, axis=1)
                else:
                    parent = int(l/5) * 5
                    xi = np.asmatrix(np.random.randn(j)).T
                    if l % 2 == 0:
                        x_l = np.multiply(np.matmul(k[i], xi), one + (1 - math.sqrt(y_i)) * x[:, parent])# / max(abs(x[:, parent]))[0, 0])
                    else:
                        x_l = np.multiply(np.matmul(k[i], xi), one + math.sqrt(y_i) * x[:, parent])# / max(abs(x[:, parent]))[0, 0])
                    x = np.append(x, x_l, axis=1)
            out.append(x)
        block = np.asarray([[1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1]])
        cov = block
        q = int(dim/5)
        for i in range(q - 1):
            cov = np.append(cov, np.zeros((5 * (i + 1), 5), dtype=int), axis=1)
            temp = np.append(np.zeros((5, 5 * (i + 1)), dtype=int), block, axis=1)
            cov = np.append(cov, temp, axis=0)
        return[out, y, cov]
    elif graph_type == 'Model III Tree':
        out = list()
        cov = list()
        for i in range(n):
            y_i = y[i]
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                    if i == 0:
                        cov_l = [1] + [0] * (dim - 1)
                else:
                    parent = int((l + 1) / 2) - 1
                    xi = np.asmatrix(np.random.randn(j)).T
                    if l % 2 == 0:
                        x_l = 0.25 * np.matmul(k[i], xi) + (1 - math.sqrt(y_i)) * x[:, parent]
                    else:
                        x_l = 0.25 * np.matmul(k[i], xi) + math.sqrt(y_i) * x[:, parent]
                    x = np.append(x, x_l, axis=1)
                    if i == 0:
                        cov_l = [0] * dim
                        cov_l[parent] = 1
                        cov_l[l] = 1
                if i == 0:
                    cov.append(cov_l)
            out.append(x)
        cov = np.asarray(cov)
        cov = cov + cov.T - np.identity(dim, dtype=int)
        return[out, y, cov]
    elif graph_type == 'Model III Hub':
        out = list()
        for i in range(n):
            y_i = y[i]
            for l in range(dim):
                if l == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x = np.matmul(k[i], xi)
                elif l % 5 == 0:
                    xi = np.asmatrix(np.random.randn(j)).T
                    x_l = np.matmul(k[i], xi)
                    x = np.append(x, x_l, axis=1)
                else:
                    parent = int(l/5) * 5
                    xi = np.asmatrix(np.random.randn(j)).T
                    if l % 2 == 0:
                        x_l = 0.25 * np.matmul(k[i], xi) + (1 - math.sqrt(y_i)) * x[:, parent]
                    else:
                        x_l = 0.25 * np.matmul(k[i], xi) + math.sqrt(y_i) * x[:, parent]
                    x = np.append(x, x_l, axis=1)
            out.append(x)
        block = np.asarray([[1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0],
                            [1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 1]])
        cov = block
        q = int(dim/5)
        for i in range(q - 1):
            cov = np.append(cov, np.zeros((5 * (i + 1), 5), dtype=int), axis=1)
            temp = np.append(np.zeros((5, 5 * (i + 1)), dtype=int), block, axis=1)
            cov = np.append(cov, temp, axis=0)
        return[out, y, cov]


def cal_coor(x, t, t_type, g_t, kernel_type):
    t_merged = t[0]
    for i in range(1, len(t)):
        t_merged = np.append(t_merged, t[i], axis=0)
    t_merged = np.unique(t_merged)
    l = len(t_merged)

    out = list()
    n = len(x)
    p = x[0].shape[1]
    if t_type == 'balanced':
        temp = kernel_t_mat(t[0], t[0], g_t, kernel_type)
        lam = np.real(max(np.linalg.eigvalsh(temp)))
        temp = np.linalg.inv(temp + lam / (len(t[0]) ** (1 / 5)) * np.identity(len(t[0])))
        for i in range(n):
            out_i = np.matmul(temp, x[i])
            x_i = np.asmatrix(np.zeros((l, p), dtype=type(out_i[0, 0])))
            x_i[np.where(t_merged == t[i][:, None])[-1], :] = out_i
            out.append(x_i)
    else:
        for i in range(n):
            temp = kernel_t_mat(t[i], t[i], g_t, kernel_type)
            lam = np.real(max(np.linalg.eigvalsh(temp)))
            temp = np.linalg.inv(temp + lam / (len(t[i]) ** (1 / 5)) * np.identity(len(t[i])))
            out_i = np.matmul(temp, x[i])
            x_i = np.asmatrix(np.zeros((l, p), dtype=type(out_i[0, 0])))
            x_i[np.where(t_merged == t[i][:, None])[-1], :] = out_i
            out.append(x_i)
    return out


def kernel_x_mat(x, c, t, g_t, kernel_type):
    n = len(x)
    dim = x[0].shape[1]

    t_merged = t[0]
    for i in range(1, len(t)):
        t_merged = np.append(t_merged, t[i], axis=0)
    t_merged = np.unique(t_merged)

    u_1 = min(t_merged)
    u_2 = max(t_merged)
    l = 99
    t_integral = np.linspace(u_1, u_2, l, endpoint=True)
    h = (u_2 - u_1)/l
    d = h / 3 * np.diag([1] + [4, 2] * int((l - 3)/2) + [4, 1])

    k = kernel_t_mat(t_integral, t_merged, g_t, kernel_type)
    metric = np.matmul(d, k)
    metric = np.matmul(k.T, metric)

    gam = list()
    temp = list()
    for p in range(dim):
        m_p = np.asmatrix(np.zeros((len(t_merged), 1), dtype=int))
        for i in range(n):
            m_p = np.append(m_p, c[i][:, p], axis=1)
        m_p = np.delete(m_p, 0, axis=1)

        G = np.matmul(m_p.T, np.matmul(metric, m_p))

        temp_k = list()
        g = 0
        for i in range(n):
            temp_ki = list()
            for j in range(i, n):
                norm = G[i, i] - G[i, j] - G[j, i] + G[j, j]
                temp_ki.append(norm)
                g = g + math.sqrt(norm)
            temp_k.append(temp_ki)
        temp.append(temp_k)
        g = 2 / (n * (n - 1)) * g
        g = 1 / (g * g)
        gam.append(g)

    out = list()
    for p in range(dim):
        out_p = list()
        for i in range(n):
            k_i = [0] * i
            for j in range(i, n):
                k_ij = math.exp(- gam[p] * temp[p][i][j-i])
                k_i = k_i + [k_ij]
            out_p.append(k_i)
        out_p = np.asmatrix(out_p)
        out_p = (out_p + out_p.T) - np.diag(np.diagonal(out_p))
        out.append(out_p)
    return out


def kernel_x_inv(k_x):
    out = list()
    n = k_x[0].shape[0]
    for i in range(len(k_x)):
        lam_i = np.real(max(np.linalg.eigvals(k_x[i])))
        k_i_inv = np.linalg.inv(k_x[i] + lam_i / (n ** (1 / 5)) * np.identity(n))
        out.append(np.matmul(k_i_inv, k_x[i]))
    return out


def kernel_y_mat(y):
    n = len(y)
    g = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            g = g + abs(y[i] - y[j])
    g = 2 / (n * (n - 1)) * g
    g = 1 / (g * g)

    out = list()
    for i in range(n):
        row = [0] * i
        for j in range(i, n):
            row = row + [math.exp(- g * (y[i] - y[j]) * (y[i] - y[j]))]
        out.append(row)
    out = np.asmatrix(out)
    out = out + out.T - np.diag(np.diagonal(out))
    return [out, g]


def precision_operator_norm(y, k_x, k_2, k_y, sample_y, gam_y):
    n = k_x[0].shape[0]
    p = len(k_x)
    y_vec = list()

    for a in range(len(sample_y)):
        y_vec.append(math.exp(- gam_y * (y - sample_y[a]) * (y - sample_y[a])))
    y_vec = np.asmatrix(y_vec).T

    lam = np.real(max(np.linalg.eig(k_y)[0]))
    k_y_inv = np.linalg.inv(k_y + lam / (n ** (1/5)) * np.identity(n))
    v = np.matmul(k_y_inv, y_vec)
    mu = np.matmul(v, v.T)
    L = np.asmatrix(np.diag(v.T.tolist()[0]))

    t1 = time.time()
    s = list()
    k_1 = list()
    for i in range(p):
        temp_1 = np.matmul(L - mu, k_x[i])
        k_1.append(temp_1)
        s.append(np.real(np.matmul(k_2[i], temp_1)))
    t2 = time.time()
    print(t2 - t1)

    singular = list()
    for i in range(p):
        singular.append(max(np.linalg.eigvals(s[i])))
    sing = np.real(max(singular))

    s_inv_sqr = list()
    for i in range(p):
        s_inv_sqr_i = np.linalg.inv(s[i] + sing/(n ** (1/5)) * np.identity(n))
        s_inv_sqr.append(np.real(sqrtm(s_inv_sqr_i)))
    t3 = time.time()
    print(t3 - t2)

    A = list()
    B = list()
    I = list()
    for i in range(p):
        A_i = np.matmul(np.matmul(k_1[i], s_inv_sqr[i]), k_2[i])
        B_i = np.matmul(k_1[i], k_2[i])
        A.append(A_i)
        B.append(B_i)
        I.append(np.linalg.pinv(B_i - np.matmul(A_i, A_i)))
    t35 = time.time()
    print(t35 - t3)

    core = np.asmatrix(np.identity(n))
    AI = list()
    for i in range(p):
        temp = np.matmul(A[i], I[i])
        core = core + np.matmul(temp, A[i])
        AI.append(np.matmul(temp, k_1[i]))
    core = np.linalg.pinv(core)
    t4 = time.time()
    print(t4 - t35)

    core_AI = list()
    for i in range(p):
        core_AI.append(np.matmul(core, AI[i]))
    t45 =time.time()
    print(t45 - t4)

    pre = list()
    for i in range(p):
        IA = np.matmul(I[i], A[i])
        IA = np.matmul(k_2[i], IA)
        pre_i = list()
        for j in range(p):
            if i > j:
                pre_i.append(0)
            else:
                pre_i.append(np.linalg.norm(np.matmul(IA[i], core_AI[j]), ord=2))
        pre.append(pre_i)
    t5 = time.time()
    print(t5 - t45)
    return np.asarray(pre)


def upper_tri(x):
    d_1 = x.shape[0]
    d_2 = x.shape[1]
    out = list()
    for i in range(d_1):
        for j in range(d_2):
            if i <= j:
                out.append(x[i, j])
            else:
                continue
    return out


def plot_roc(t, c):
    pre_mat = t
    cov_mat = c

    y_pred = upper_tri(pre_mat)
    y_test = upper_tri(cov_mat)

    fpr, tpr, thr = roc_curve(y_test, y_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def repeated_exp(times, n, dim, y, t_type, d_type, k_type):
    out = list()
    for i in tqdm(range(times)):
        a, t = generate_t(n, t_type, 50)
        b = gen_sample(a, 50, d_type, dim, t, k_type)
        c = cal_coor(b[0], a, t_type, t, k_type)

        dim = b[0][0].shape[1]

        d = kernel_x_mat(b[0], c, a, t, 'brownian')
        d_inv = kernel_x_inv(d)
        e, g = kernel_y_mat(b[1])

        p = precision_operator_norm(y, d, d_inv, e, b[1], g)

        y_p = upper_tri(p)
        y_t = upper_tri(b[2])

        fpr, tpr, thr = roc_curve(y_t, y_p, drop_intermediate=False)
        out.append(round(auc(fpr, tpr), 5))
        a, b, c, d, e, f, s, t, p = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return out


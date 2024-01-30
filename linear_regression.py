import math
import copy
import matplotlib.pyplot as plt
import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


def compute_cost(x, y, w, b):
    # m = 'Number of features'
    m = len(x)

    cost_sum = 0

    # initialized Array(list) of f_wb
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb = (x[i] * w) + b

        # Square error function | Loss function
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]

    # initialized derivative term
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = (x[i] * w) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    # mean values of derivative terms
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    j_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, j_history, p_history  # return w and J,w history for graphing

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2


w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# ----------------------------------------------------------------------

# Graph 1
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))

# 플롯 세팅
ax1.plot(J_hist[:100], linewidth=2, c='r')
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:], linewidth=2)

# 타이틀
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")

# 라벨링
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')

ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')

# 그리드 추가
ax1.grid(True, linestyle='--')
ax2.grid(True, linestyle='--')


# Graph 2
# w와 b에 대한 값의 범위 설정
w_values = np.linspace(-100, 400, 100)  # 예를 들어 -100에서 600 사이
b_values = np.linspace(-400, 400, 100)  # 예를 들어 -100에서 600 사이

# 그리드 생성
W, B = np.meshgrid(w_values, b_values)
Z = np.zeros_like(W)

# 각 그리드 포인트에 대해 비용 계산
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])

# 등고선 그래프 그리기
plt.figure(figsize=(8, 6))
CS = plt.contour(W, B, Z, cmap='Paired', levels=[50, 1000, 5000, 10000, 25000, 50000])
plt.clabel(CS, inline=1, fontsize=10)

min_cost_index = np.unravel_index(np.argmin(Z), Z.shape)

W_min = W[min_cost_index]
B_min = B[min_cost_index]

plt.axvline(W_min, color='grey', linestyle='--', linewidth=1)
plt.axhline(B_min, color='grey', linestyle='--', linewidth=1)

plt.title('Cost function contour for linear regression')
plt.xlabel('w (weight)')
plt.ylabel('b (bias)')


plt.show()


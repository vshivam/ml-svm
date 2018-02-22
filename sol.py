from scipy.optimize import minimize

import numpy
import random
import matplotlib.pyplot as plt
import math

orderofkernel = 7
sigma = 2


def rbf_kernel(i, j):
    val = numpy.linalg.norm(inputs[i] - inputs[j]) ** 2
    return math.exp(-val / (2 * (sigma ** 2)))


def rbf_kernel_2(vector1, vector2):
    val = numpy.linalg.norm(vector1 - vector2) ** 2
    return math.exp(-val / (2 * (sigma ** 2)))


def linear_kernel(i, j):
    # print x
    # print y
    return numpy.dot(inputs[i], inputs[j])


def linear_kernel_2(vector1, vector2):
    return numpy.dot(vector1, vector2)


def quadratic_kernel(i, j):
    val = linear_kernel(i, j)
    return (val + 1) ** orderofkernel


def quadratic_kernel_2(vector1, vector2):
    return (linear_kernel_2(vector1, vector2) + 1) ** orderofkernel


kernel = quadratic_kernel

kernel2 = quadratic_kernel_2


def draw(classA, classB):
    plt.axis('equal')


training_class_a = []
training_class_b = []

seed = numpy.random.randint(0, 99)


def generate_training_data():
    numpy.random.seed(100)
    classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [-1.5, 1.5],
         numpy.random.randn(10, 2) * 0.2 + [-1.5, 1.5]))
    classA[19] = numpy.array([-1.5, 0.25])
    classB = numpy.random.randn(20, 2) * 0.2 + [-0.5, 0.0]

    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
         -numpy.ones(classB.shape[0])))

    N = inputs.shape[0]  # number of rows (samples)
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB


def generate_test_data():
    numpy.random.seed(seed)
    classA = numpy.concatenate(
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
         -numpy.ones(classB.shape[0])))

    N = inputs.shape[0]  # number of rows (samples)
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB


inputs, targets, training_class_a, training_class_b = generate_training_data()

# print inputs.shape
# print targets.shape
N = inputs.shape[0]

# print "N: " + str(N)


'''
P = numpy.zeros((N, N))
for i in range(0, N):
    for j in range(0, N):
        P[i][j] = targets[i] * targets[j] * quadratic_kernel(inputs[i], inputs[j])

# print P
'''


def zerofun(alpha):
    # print alpha
    sum = 0
    for i in range(0, N):
        sum = sum + alpha[i] * targets[i]
    return sum


constraints = {"type": "eq", "fun": zerofun}


def objective(alpha):
    alpha_sum = numpy.sum(alpha)
    sum = 0
    for i in range(0, N):
        for j in range(0, N):
            sum = sum + alpha[i] * alpha[j] * targets[i] * targets[j] * kernel(i, j)
    L = alpha_sum - 0.5 * sum
    # print L
    return L * (-1)


C = 0.005
bounds = [(0, C) for b in range(N)]
# print bounds
alpha_init = numpy.zeros(N)
ret = minimize(objective, alpha_init, bounds=bounds, constraints=constraints)
print ret
# print ret['x']
# print ret['success']
optimum_alphas = ret['x']

l = []


class Output:
    def __init__(self, alpha, input, target):
        self.alpha = alpha
        self.input = input
        self.target = target


for idx, alpha in enumerate(optimum_alphas):
    if alpha < 10 ** (-5):
        optimum_alphas[idx] = 0
    else:
        sum = 0
        output = Output(alpha, inputs[idx], targets[idx])
        l.append(output)

for i in l:
    print i.alpha
    print i.input
    print i.target
    print "\n"

if len(l)==0:
    print "No SOLUTION FOUND!"
    exit(0)
else:
    print "Solution found"

idx_selected_support_vector = 0
xk = l[idx_selected_support_vector].input
tk = l[idx_selected_support_vector].target
b = 0;

for i in l:
    b = b + i.alpha * i.target * kernel2(i.input, xk)
b = b - tk

print b


def indicator(s):
    ind = 0
    for i in l:
        ind = ind + i.alpha * i.target * kernel2(i.input, s)
    return ind - b


def indicator_2(x1, x2):
    input = numpy.array([x1, x2])
    ind = 0
    for i in l:
        ind = ind + i.alpha * i.target * kernel2(i.input, input)
    return ind - b


classA = []
classB = []
new_test_data, test_target, trash_a, trash_b = generate_test_data()
for idx, val in enumerate(new_test_data):
    test_target[idx] = indicator(val)
    if (test_target[idx] > 0):
        classA.append(val)
    else:
        classB.append(val)

print test_target

print training_class_a

print training_class_b

print classA

print classB

'''
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'k.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'y.')
'''
plt.plot([p[0] for p in training_class_a],
         [p[1] for p in training_class_a],
         'b.')
plt.plot([p[0] for p in training_class_b],
         [p[1] for p in training_class_b],
         'r.')

for i in l:
    if (i.target == 1):
        plt.plot(i.input[0], i.input[1],
                 'bo')
    else:
        plt.plot(i.input[0], i.input[1],
                 'ro')

print l

xgrid = numpy.linspace(-5, 5, 50)
ygrid = numpy.linspace(-5, 5, 50)
grid = numpy.zeros((50, 50))
vector_s = numpy.zeros((50, 50))

grid = numpy.array([[indicator_2(x, y) for y in ygrid] for x in xgrid])
grid = numpy.transpose(grid)

plt.contour(xgrid, ygrid, grid, (-1, 0.0, 1.0), colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))
plt.show()

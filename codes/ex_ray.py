import ray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import time

cpu_nums = 16
ray.init(num_cpus=cpu_nums)


def generate_data():
    x, y = make_classification(n_samples=50000, n_features=200, n_classes=2)
    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


@ray.remote
def lr_eval(index):
    x, y = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(x_train, y_train)
    score = lr.score(x_test, y_test)
    return [index, score]


def ray_eval():
    ts = time.time()
    result_id = [lr_eval.remote(i) for i in range(cpu_nums)]
    res = ray.get(result_id)
    print(res)
    print(f'ray running time: {time.time() - ts}')
    ray.shutdown()


def lr_eval_local():
    ts = time.time()
    for index in range(cpu_nums):
        x, y = generate_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        lr = LogisticRegression(max_iter=2000)
        lr.fit(x_train, y_train)
        score = lr.score(x_test, y_test)
        print(f'{index}, score: {score}')
    print(f'serial running time: {time.time() - ts}')


if __name__ == '__main__':
    ray_eval()
    lr_eval_local()

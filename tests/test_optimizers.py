
from src.dlmpc import optimizer

if __name__ == '__main__':
    # optimizer(optimizer_name='adam', learning_rate=0.1, du_bound=0.1, exponential_decay=True)
    optimizer(optimizer_name='sgd', learning_rate=0.1, du_bound=0.1, exponential_decay=True)

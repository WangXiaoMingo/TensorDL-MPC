from src.dlmpc import NonNegative, BoundedConstraint
import tensorflow as tf
if __name__ == '__main__':
    weight = tf.constant((-1.0, 1.0))
    # tf.Tensor([-0.  1.], shape=(2,), dtype=float32)
    print(NonNegative()(weight))

    u_bounds = [-0.5, 0.5]
    u = tf.Variable(tf.random.uniform((5,1)),dtype=tf.float32)
    bounded_constraint = BoundedConstraint(lower_bound=u_bounds[0], upper_bound=u_bounds[1])
    print(bounded_constraint(u))
    #tf.Tensor([[0.5       ]
     # [0.5       ]
     # [0.33824527]
     # [0.5       ]
     # [0.240731  ]], shape=(5, 1), dtype=float32)

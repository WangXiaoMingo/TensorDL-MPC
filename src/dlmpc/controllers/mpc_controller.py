# dlmpc/controllers/mpc_controller.py

# Software Copyright Notice

#   This file is part of DL-MPC
#
#   DL-MPC: A toolbox for deep learning-based nonlinear model predictive control
#
#   Copyright (c) 2024, Xiaoming Wang. All rights reserved
#
#   This software (including but not limited to all modules, files, and code) is developed and owned by Xiaoming Wang.
#   Unauthorized distribution, copying, modification, or redistribution of this software in any form is prohibited.
#
#   This software is freely available for academic research activities, including but not limited to paper writing,
#   academic discussions, and academic conferences.
#
#   Please note that this software is not intended for commercial use, including but not limited to commercial projects,
#   products, and services. If you require the use of this software for commercial purposes, please contact Xiaoming Wang
#   to obtain the appropriate license.
#
#   This software may contain third-party software components, the use and distribution of which are subject to the
#   respective third-party license agreements. If you need to use third-party software components,
#   please ensure compliance with their license agreements.
#
#   If you have any questions about the software or require further assistance,
#   please contact Xiaoming Wang support team. e-mail: wangxiaoming19951@163.com
#
#   Last updated on June, 2024
#   Author: Xiaoming Wang

import numpy as np
import tensorflow as tf
from ..constraints.constraint import BoundedConstraint
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)

class MPCController:
    def __init__(self, model, prediction_horizon, control_horizon, Q, R, ly, lu, dim_u, du_bounds,u_bounds, optimizer):
        """
        Initialize the DeepLearningMPCController.

        Parameters:
        model: A trained neural network model for predicting system dynamics.
        prediction_horizon: The time domain for prediction.
        control_horizon: The time domain for control.
        Q: The state weight matrix.
        R: The control weight matrix.
        ly: The size of the past y values.
        lu: The size of the past u values, excluding the current moment.
        dim_u: The number of control variables.
        du_bounds: The lower and upper bounds for du.
        u_bounds: The lower and upper bounds for u.
        optimizer: The optimizer, 'sgd' by default.
        """
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.ly = ly
        self.lu = lu
        self.dim_u = dim_u
        self.current_state = None
        self.lower_bound_du = du_bounds[0]
        self.upper_bound_du = du_bounds[1]
        self.lower_bound = u_bounds[0]
        self.upper_bound = u_bounds[1]
        self.optimizer = optimizer
        self.error = 0
        self.learning_rate = self.optimizer.learning_rate
        self.u_sequence = tf.Variable(initial_value=np.zeros((self.control_horizon, 1)),
                                      name='u', trainable=True, dtype=tf.float32)

    def rolling_predict(self, state_y, state_u, error):
        predict_y = tf.constant([[0]], dtype=tf.float32)
        # Use TensorFlow's vectorized operations for prediction
        for k in range(self.prediction_horizon):
            # Select the control input based on the control horizon
            x_current = tf.concat([state_y, state_u], axis=1)
            current_u = tf.reshape(
                self.u_sequence[tf.minimum(k, self.control_horizon - 1)], [1, 1, 1])
            # Predict the next state
            y_pred = self.model([x_current, current_u]) + error
            # Update the states
            state_y = tf.concat([state_y, tf.reshape(y_pred, [1, 1, 1])], axis=1)[:, -self.ly:, :]
            state_u = tf.concat([state_u, current_u], axis=1)[:, -self.lu:, :]
            # Accumulate the prediction results
            predict_y = tf.concat([predict_y, y_pred], axis=0)
        # Return the last 'prediction_horizon' number of predictions
        return tf.reshape(predict_y[-self.prediction_horizon:], (self.prediction_horizon, 1))

    def cost_function(self,predict_y,y_ref):
        '''
         Compute the cost function based on the reference and predicted states.

        Parameters:
        - y_ref (tf.Tensor): The reference state tensor, against which the predictions are compared.
        - predict_y (tf.Tensor): The tensor representing the predicted state from the model.

        Returns:
        - tf.Tensor: The computed cost, which is a scalar tensor representing the error between the reference and predicted states.
       '''
        cost = tf.transpose(predict_y - y_ref) @ self.Q @ (predict_y - y_ref)
        return cost

    def optimize(self, error, state_y, state_u, y_ref,iterations=1000, tol=1e-8,function_type=None):
        """
        Optimize the control input sequence.

        Parameters:
        u0: Initial control input, type is tf.Tensor.
        state_y: Initial state, type is tf.Tensor.
        state_u: Initial control input, type is tf.Tensor. (Note: This parameter seems to be a duplicate of 'u0' and should be reviewed for accuracy.)
        u_sequence: Control input sequence, type is tf.Tensor.
        y_ref: Reference state, type is tf.Tensor.

        Returns:
        The optimized control input sequence, type is tf.Tensor.
        """

        # Initialize the number of iterations and the previous cost
        # Ensure that u_sequence is a trainable variable

        J_prev = -1
        state_y = tf.expand_dims(state_y,axis=0)
        state_u = tf.expand_dims(state_u,axis=0)
        for epoch in range(iterations):
            J = self.compute_gradients_update(error,state_y, state_u,y_ref)
            if np.abs(J - J_prev) and J <= J_prev < tol :  #
                return J, self.u_sequence.numpy(), epoch+1
            J_prev = J
        return J.numpy(), self.u_sequence.numpy(), epoch+1

    @tf.function
    def compute_gradients_update(self,error, state_y,state_u, y_ref):
        """
        Calculate the gradients for model updating based on prediction error and state variables.

        Parameters:
        error: The prediction error, indicating the discrepancy between the predicted and actual outcomes, type is tf.float32.
        state_y: The vector representing the current system state variables, type is tf.float32.
        state_u: The vector representing the current control inputs, type is tf.float32.
        y_ref: The reference or target state values that the system is aiming to achieve, type is tf.float32.

        Returns:
        The gradient value, a tf.float32 tensor representing the gradient of the loss function with respect to the model's weights.
        """
        # Implement the logic for gradient computation
        # Use the loss function to calculate the gradients based on the prediction error and the difference between current and target states
        # Utilize TensorFlow's automatic differentiation to compute the gradients

        with tf.GradientTape() as tape:
            # Predict the state sequence using the rolling prediction method
            y_pred_sequence = self.rolling_predict(state_y, state_u, error)
            # Calculate the cost based on the predicted and reference states
            cost = self.cost_function(y_ref, y_pred_sequence)
        # Calculate the gradients of the cost function with respect to the control input sequence
        gradients = tape.gradient(cost, self.u_sequence)
        delta_u = 2 * self.learning_rate * tf.linalg.inv(
            tf.eye(self.control_horizon) - 2 * self.learning_rate * self.R) @ gradients
        # # 控制增量约束, if need
        # delta_u = BoundedConstraint(self.lower_bound_du, self.upper_bound_du)(delta_u)

        # Apply constraints to the control input sequence to ensure it remains within bounds
        u_sequence = BoundedConstraint(self.lower_bound, self.upper_bound)(self.u_sequence.assign_sub(delta_u))
        self.u_sequence.assign(u_sequence)
        # also can be use as following
        # self.u_sequence.assign_sub(delta_u)
        # self.ensure_constraints()
        return cost

    def ensure_constraints(self):
        for k in range(self.control_horizon):
            if self.u_sequence[k, 0] > self.upper_bound:
                self.u_sequence[k, 0].assign(self.upper_bound)
            if self.u_sequence[k, 0] < self.lower_bound:
                self.u_sequence[k, 0].assign(self.lower_bound)

    def online_correction(self, predicted_state, measured_state):
        """
        Online correction of the predicted state based on the measured state.

        Parameters:
        predicted_state: The current predicted state of the system, of type tf.float32.
        measured_state: The current measured state of the system, of type tf.float32.

        Returns:
        The corrected state after online calibration, of type tf.float32.
        """
        # Implement the online correction logic
        # Adjust the predicted state using the measured state to improve accuracy

        '''see close_loop_compuatation.py'''
        self.error = measured_state - predicted_state
        return self.error

    def update_model(self, model):
        """
        Update the model.

        Parameters:
        model: The model to be updated with the gradients. The type of 'model' should be specified here.

        Returns:
        Optionally, specify what the function returns, if applicable.
        """
        # Implement the model update logic
        # Apply the gradients to the model's parameters to update the model
        self.model = model

    def reset(self,prediction_horizon, control_horizon, Q, R, ly, lu, dim_u, du_bounds,u_bounds, optimizer):
        """
        reset parameters。
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.Q = tf.convert_to_tensor(Q, dtype=tf.float32)
        self.R = tf.convert_to_tensor(R, dtype=tf.float32)
        self.ly = ly
        self.lu = lu
        self.dim_u = dim_u
        self.current_state = None
        self.du_bounds = du_bounds
        self.lower_bound_du = du_bounds[0]
        self.upper_bound_du = du_bounds[1]
        self.lower_bound = u_bounds[0]
        self.upper_bound = u_bounds[1]
        self.optimizer = optimizer




class MPC:
    def __init__(self,control_horizon,predictive_horizon=1,model=1):
        self.control_horizon = control_horizon
        self.model = model
        self.predictive_horizon = predictive_horizon

    def predict(self):
        print(self.control_horizon)
        print(self.predictive_horizon)
        return self.predictive_horizon+1


MPC.control_horizon=1
MPC.predictive_horizon=1

print(MPC.predict(MPC(1)))



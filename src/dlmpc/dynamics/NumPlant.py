
def plant_model(x, u):
    # u = u0 + u     # 两种求解方法
    y = 0.5 *x + u**2
    return y
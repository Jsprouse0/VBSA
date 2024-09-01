def rosenbrock_v0(*args):
    a = 1
    if len(args) % 2:
        raise ValueError('rosenbrock_v0 requires an even number of arguments')
    return sum(100 * (args[i]**2 - args[i+1])**2 + (args[i] - a)**2 for i in range(0, len(args), 2))

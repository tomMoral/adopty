

def stop_on_value(threshold_cost, costs):
    return costs[-1] < threshold_cost


def stop_on_no_decrease(tol, costs):
    return costs[-2] - costs[-1] < tol

import argparse


def checkFieldSize(size: str):
    size = int(size)
    if size <= 4:
        raise argparse.ArgumentTypeError("Field size must be greater than 4")
    return size

def checkNumCandys(num: int):
    num = int(num)
    if num <= 3:
        raise argparse.ArgumentTypeError("Number of candys must be greater than 3")
    return num

def checkDesiredReward(desired_reward: str):

    try:
        desired_reward = float(desired_reward)
    except:
        raise argparse.ArgumentTypeError("Invalid desired_reward. It must be a float value.")

    if desired_reward < 0:
        raise argparse.ArgumentTypeError("Invalid desired_reward. It can not be negativ")

    return desired_reward

def checkMode(mode: str):
    """
    Check if mode is "0" or "1"
    Keyword arguments:
        mode -- Must be "0" or "1" otherwise an exception will be thrown
    Return:
        mode 
    """
    if mode != "0" and mode != "1":
        raise argparse.ArgumentTypeError("Invalid mode option. Use \"0\" = game window or \"1\" = game window with plots")

    return mode

class dummy_context_mgr():
    """
    A null object required for a conditional with statement
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False
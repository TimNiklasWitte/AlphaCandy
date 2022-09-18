from PPO_Agent import*
import argparse

def checkFieldSize(size: str):
    size = int(size)
    if size <= 4:
        raise argparse.ArgumentTypeError("Field size must be greater than 4")
    return size

def checkNumCandys(num: int):
    num = int(num)
    if num <= 4:
        raise argparse.ArgumentTypeError("Number of candys must be greater than 4")
    return num

def main():

    # Set up ArgumentParser
    parser = argparse.ArgumentParser(description="Setup the desired CandyCrush/Training sconfiguration")
    parser.add_argument("--size", help="Set the field size.", type=checkFieldSize, required=True)
    parser.add_argument("--num", help="Set the number of candys.", type=checkNumCandys, required=True)

    args = parser.parse_args()

    agent = Agent_PPO("CandyCrushGym",args.size,args.num)
	
    agent.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
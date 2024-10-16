import os
import sys
sys.path.append(os.path.join(os.getcwd(), "diffiner"))
#from dotenv import load_dotenv

from src.run import run


#load_dotenv()


def main():
    run()


if __name__ == "__main__":
    main()
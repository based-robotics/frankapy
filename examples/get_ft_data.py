from frankapy import FrankaArm
from time import sleep


if __name__ == "__main__":
    fa = FrankaArm()

    try:
        while True:
            print("FT wrench is", fa.get_ft_wrench())
            sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    
from dotenv import load_dotenv
from utils.env_utils import doublecheck_env, doublecheck_pkgs

def main():
    print("Hello from abstract-gen!")

    load_dotenv()

    # Check and print results
    doublecheck_env(".env")  # check environmental variables
    doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)  # check packages


if __name__ == "__main__":
    main()

from src.pipeline import run_pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    run_pipeline()
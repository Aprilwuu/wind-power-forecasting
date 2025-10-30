from src.utils.config import load_config
from src.data.load import load_data
from src.train.trainer import train_model

def main():
    print(">>> starting run_track1")
    cfg = load_config("configs/track1.yaml")
    df = load_data(cfg["data"]["path"])
    print(" Data loaded successfully. Shape =", df.shape)
    train_model(df, cfg["data"]["target"], cfg)

if __name__ == "__main__":
    main()
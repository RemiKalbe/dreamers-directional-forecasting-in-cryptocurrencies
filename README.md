# Directional Forecasting in Cryptocurrencies

## Setup

### Using Pixi (Recommended)
This project uses [Pixi](https://pixi.sh/) as its package manager. To get started:

1. Install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Install dependencies:
```bash
pixi install
```

### Manual Installation
Alternatively, you can install the required packages manually using pip or your preferred package manager. Check `pyproject.toml` for the complete list of dependencies.

## Dataset

The dataset is precomputed and split into multiple .parquet files. Due to size limitations of github, they need to be downloaded separately:

- `features_selected_X_test_df.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/features_selected_X_test_df.parquet) (235MB)
- `features_selected_X_val_df.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/features_selected_X_val_df.parquet) (109MB)
- `selected_features_X_train_unnormalized.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/selected_features_X_train_unnormalized.parquet) (438MB)
- `test_unnormalized.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/test_unnormalized.parquet) (3.07GB)
- `val_unnormalized.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/val_unnormalized.parquet) (5.88GB)
- `train_unnormalized.parquet` - [Download](https://r2.remi.boo/dreamers-directional-forecasting-in-cryptocurrencies/train_unnormalized.parquet) (1.45GB)

### Computing the Dataset
While you can compute the dataset from scratch, please note that some features are computationally intensive. For example, the wavelet transform feature alone took approximately 3 hours to compute on a modern machine.

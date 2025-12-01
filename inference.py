from pathlib import Path
from typing import Union, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Paths to artifacts
# ------------------------------------------------------------------
MODEL_PATH = Path("models/ctr_model.pth")
CAT_VOCAB_PATH = Path("models/cat_vocab.pkl")
CAT_NUM_CLASSES_PATH = Path("models/cat_num_classes.pkl")
SCALER_PATH = Path("models/numeric_scaler.pkl")
FINAL_CATEGORICAL_PATH = Path("models/final_categorical.pkl")
FINAL_NUMERIC_PATH = Path("models/final_numeric.pkl")

BASE_TIME = pd.Timestamp("2016-06-01")
RawInput = Union[pd.DataFrame, pd.Series, dict]


# ------------------------------------------------------------------
# Model (matches notebook)
# ------------------------------------------------------------------
def get_emb_dim(n_cat: int) -> int:
    return min(50, int(round(n_cat ** 0.5) * 2))


class CTRModel(nn.Module):
    def __init__(
        self,
        cat_num_classes,
        num_numeric_features: int,
        emb_dropout: float = 0.1,
        hidden_dims=(128, 64),
    ):
        super().__init__()

        self.cat_cols = list(cat_num_classes.keys())

        # Embeddings
        self.emb_layers = nn.ModuleDict()
        emb_out_dims = []
        for col, n_classes in cat_num_classes.items():
            emb_dim = get_emb_dim(n_classes)
            self.emb_layers[col] = nn.Embedding(
                num_embeddings=n_classes,
                embedding_dim=emb_dim,
                padding_idx=0,
            )
            emb_out_dims.append(emb_dim)

        total_emb_dim = sum(emb_out_dims)
        input_dim = total_emb_dim + num_numeric_features

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(emb_dropout))
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev, 1)

    def forward(self, cat_inputs: torch.Tensor, num_inputs: torch.Tensor) -> torch.Tensor:
        emb_list = []
        for i, col in enumerate(self.cat_cols):
            emb = self.emb_layers[col](cat_inputs[:, i])
            emb_list.append(emb)

        emb_cat = torch.cat(emb_list, dim=1)
        x = torch.cat([emb_cat, num_inputs], dim=1)
        x = self.mlp(x)
        logits = self.output_layer(x).squeeze(1)
        return logits


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class CTRInferencePipeline:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        cat_vocab_path: Path = CAT_VOCAB_PATH,
        cat_num_classes_path: Path = CAT_NUM_CLASSES_PATH,
        scaler_path: Path = SCALER_PATH,
        final_cat_path: Path = FINAL_CATEGORICAL_PATH,
        final_num_path: Path = FINAL_NUMERIC_PATH,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ---- load preprocessing artifacts ----
        self.cat_vocab = joblib.load(cat_vocab_path)
        full_cat_num_classes = joblib.load(cat_num_classes_path)
        self.scaler = joblib.load(scaler_path)
        self.final_categorical = joblib.load(final_cat_path)
        self.final_numeric = joblib.load(final_num_path)

        # numeric columns actually used when fitting the scaler
        if hasattr(self.scaler, "feature_names_in_"):
            self.scaler_numeric_cols = list(self.scaler.feature_names_in_)
        else:
            # fallback – use FINAL_NUMERIC length
            self.scaler_numeric_cols = list(self.final_numeric)

        # means used by scaler during fit (for filling missing)
        self.numeric_means = dict(
            zip(self.scaler_numeric_cols, self.scaler.mean_)
        )

        # restrict cat_num_classes to final categoricals only
        self.cat_num_classes = {
            col: full_cat_num_classes[col] for col in self.final_categorical
        }

        # ---- load model state dict & infer hidden sizes ----
        state_dict = torch.load(model_path, map_location=self.device)
        hidden_dims = self._infer_hidden_dims_from_state_dict(state_dict)

        self.model = CTRModel(
            cat_num_classes=self.cat_num_classes,
            num_numeric_features=len(self.final_numeric),
            emb_dropout=0.2,
            hidden_dims=hidden_dims,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    # --------------- helpers ----------------
    @staticmethod
    def _infer_hidden_dims_from_state_dict(state_dict) -> Tuple[int, ...]:
        linear_layers = []
        for k, v in state_dict.items():
            m = re.match(r"mlp\.(\d+)\.weight$", k)
            if m:
                idx = int(m.group(1))
                out_features = v.shape[0]
                linear_layers.append((idx, out_features))
        if not linear_layers:
            raise RuntimeError("Could not infer hidden_dims from state_dict.")
        linear_layers.sort(key=lambda x: x[0])
        return tuple(out for _, out in linear_layers)

    @staticmethod
    def _to_dataframe(x: RawInput) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        if isinstance(x, pd.Series):
            return x.to_frame().T
        if isinstance(x, dict):
            return pd.DataFrame([x])
        raise TypeError(
            f"Unsupported input type {type(x)}. Use dict / Series / DataFrame."
        )

    # --------------- feature engineering ----------------
    def _ensure_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # event_hour / event_dayofweek / event_time_dt from event_timestamp
        if "event_timestamp" in df.columns:
            if "event_time_dt" not in df.columns:
                df["event_time_dt"] = BASE_TIME + pd.to_timedelta(
                    df["event_timestamp"], unit="s"
                )
            if "event_hour" not in df.columns:
                df["event_hour"] = df["event_time_dt"].dt.hour.astype("int8")
            if "event_dayofweek" not in df.columns:
                df["event_dayofweek"] = df["event_time_dt"].dt.dayofweek.astype("int8")

        # publish times to datetime
        if "event_doc_publish_time" in df.columns:
            df["event_doc_publish_time"] = pd.to_datetime(
                df["event_doc_publish_time"], errors="coerce"
            )
        if "ad_doc_publish_time" in df.columns:
            df["ad_doc_publish_time"] = pd.to_datetime(
                df["ad_doc_publish_time"], errors="coerce"
            )

        # document ages
        if (
            "event_doc_age_hours" not in df.columns
            and "event_time_dt" in df.columns
            and "event_doc_publish_time" in df.columns
        ):
            df["event_doc_age_hours"] = (
                df["event_time_dt"] - df["event_doc_publish_time"]
            ).dt.total_seconds() / 3600.0
            df["event_doc_age_hours"] = df["event_doc_age_hours"].fillna(0)

        if (
            "ad_doc_age_hours" not in df.columns
            and "event_time_dt" in df.columns
            and "ad_doc_publish_time" in df.columns
        ):
            df["ad_doc_age_hours"] = (
                df["event_time_dt"] - df["ad_doc_publish_time"]
            ).dt.total_seconds() / 3600.0
            df["ad_doc_age_hours"] = df["ad_doc_age_hours"].fillna(0)

        # same_topic flag
        if (
            "same_topic" not in df.columns
            and "event_doc_top_topic_id" in df.columns
            and "ad_doc_top_topic_id" in df.columns
        ):
            df["same_topic"] = (
                df["event_doc_top_topic_id"] == df["ad_doc_top_topic_id"]
            ).astype("int8")

        return df

    # --------------- preprocessing ----------------
    def preprocess(self, raw_input: RawInput):
        df = self._to_dataframe(raw_input)

        # 1) engineered features
        df = self._ensure_engineered_features(df)

        # 2) ensure all scaler numeric cols exist; missing -> training mean
        for col in self.scaler_numeric_cols:
            if col in df.columns or col in self.numeric_means:
                df[col] = float(self.numeric_means[col])

        # 3) scale using numpy array to avoid feature-name mismatch checks
        vals = df[self.scaler_numeric_cols].values.astype("float32")
        scaled = self.scaler.transform(vals)
        df.loc[:, self.scaler_numeric_cols] = scaled

        # 4) categorical -> indices
        for col in self.final_categorical:
            vocab = self.cat_vocab[col]
            s = df[col].astype(str).fillna("UNK")
            unk_idx = vocab.get("<UNK>", 0)
            df[col + "_idx"] = s.map(lambda v: vocab.get(v, unk_idx)).astype("int64")

        cat_idx_cols = [c + "_idx" for c in self.final_categorical]

        # 5) numeric tensor uses FINAL_NUMERIC (subset of scaled cols)

        cat_tensor = torch.tensor(
            df[cat_idx_cols].values,
            dtype=torch.long,
            device=self.device,
        )
        num_tensor = torch.tensor(
            df[self.final_numeric].values.astype("float32"),
            dtype=torch.float32,
            device=self.device,
        )

        return cat_tensor, num_tensor

    # --------------- inference ----------------
    @torch.no_grad()
    def predict_proba(self, raw_input: RawInput) -> np.ndarray:
        cat_x, num_x = self.preprocess(raw_input)
        logits = self.model(cat_x, num_x)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, raw_input: RawInput, threshold: float = 0.3):
        probs = self.predict_proba(raw_input)
        preds = (probs >= threshold).astype("int64")
        if probs.size == 1:
            return int(preds[0]), float(probs[0])
        return preds, probs


# --- add this near the bottom of inference.py ---

# create a single global pipeline instance to reuse across requests
_pipeline = CTRInferencePipeline()

def predict_ctr_one(raw_row: dict):
    """
    Convenience wrapper for single-row inference.

    Parameters
    ----------
    raw_row : dict
        Raw feature dict (keys are column names).

    Returns
    -------
    (prob, label)
        prob  : float  -> predicted click-through probability
        label : int    -> 0/1 prediction using default threshold 0.5
    """
    label, prob = _pipeline.predict(raw_row)
    return float(prob), int(label)


# ------------------------------------------------------------------
# Quick local test
# ------------------------------------------------------------------
if __name__ == "__main__":
    pipe = CTRInferencePipeline()

    df = pd.read_csv('data/train_merged.csv')
    row = df.sample(1).iloc[0]
    raw_row = row.to_dict()
    label, prob = pipe.predict(raw_row)
    print(f"Predicted clicked label: {label}, probability: {prob:.4f}")


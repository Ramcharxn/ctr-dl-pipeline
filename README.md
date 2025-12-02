# Real-Time CTR Prediction System (PyTorch + FastAPI + AWS EKS)

This project is an **end-to-end Click-Through Rate (CTR) prediction system** that:

- Trains a deep learning model on the **Outbrain Ad Click Prediction dataset**
- Exposes a real-time **`/predict`** API via **FastAPI**
- Runs on **AWS EKS** using self-managed **EC2 worker nodes** in an **Auto Scaling Group (ASG)**
- Uses **Docker + ECR** as the deployment backbone

It is designed to resemble the **ad ranking / CTR scoring services** used in modern ad platforms.

---

## Key Features

- **PyTorch CTR model** with embeddings + MLP
- **Feature-rich pipeline** (temporal, frequency, CTR aggregates, topic match)
- **Hyperparameter tuning** with **Optuna**
- **Evaluation** using **ROC-AUC & LogLoss**
- **FastAPI inference service** (Dockerized)
- **AWS-native deployment** with:
  - EKS control plane
  - Self-managed EC2 worker nodes via ASG
  - ECR for container images
  - Kubernetes **LoadBalancer** to expose the public `/predict` endpoint

---

## Tech Stack

### ML & Data
- Python 3.12
- Pandas, NumPy
- PyTorch
- Scikit-Learn (scaling, splitting)
- LightGBM (feature importance / selection)
- Optuna (hyperparameter optimization)

### Serving & Infrastructure
- FastAPI, Uvicorn
- Docker
- **AWS ECR** – container registry
- **AWS EKS** – Kubernetes control plane
- **Amazon EC2 + Auto Scaling Group** – self-managed worker nodes
- **AWS Elastic Load Balancer** – Distribute traffic
- AWS IAM, AWS CloudShell

---

## End-to-End Pipeline (Single Flow)

```text
Outbrain Data
  → Feature Engineering & Preprocessing
  → LightGBM Feature Selection
  → Train & Tune PyTorch CTR Model (Optuna)
  → Save Artifacts (model + vocab + scaler + feature lists)
  → Build Docker Image with FastAPI Inference
  → Push Image to AWS ECR
  → Deploy on AWS EKS (control plane) + EC2 ASG (workers)
  → Expose /predict via Kubernetes LoadBalancer (ELB)
```

## Repository Structure

```text
.
├── app.py                    # FastAPI application (API endpoints)
├── inference.py              # Model loading, preprocessing, prediction logic
├── model_def.py              # PyTorch CTR model definition
├── models/
│   ├── ctr_model.pth         # Trained PyTorch model
│   ├── cat_vocab.pkl         # Categorical vocabulary mapping
│   ├── cat_num_classes.pkl   # Number of classes per categorical feature
│   ├── numeric_scaler.pkl    # Fitted StandardScaler for numeric features
│   ├── final_categorical.pkl # Final list of selected categorical features
│   ├── final_numeric.pkl     # Final list of selected numeric features
├── Dockerfile                # Docker image for FastAPI + PyTorch inference
├── requirements.txt          # Python dependencies
├── config                    
│    ├──ctr-deployment.yaml       # Kubernetes Deployment (ECR)
│    └──ctr-service.yaml       # Kubernetes Service (LoadBalancer)
└── README.md                 # This file
```


---

## Data & Features (Very Short Overview)

### Dataset
Based on the **Outbrain Click Prediction** dataset, mainly using:
- `clicks_train.csv`
- `events.csv`
- `promoted_content.csv`
- `documents_meta.csv`
- `documents_topics.csv`

### Features (Summary Only)
- Joins on keys like `display_id`, `ad_id`, `document_id`, `uuid`, `campaign_id`, `publisher_id`
- Core features:
  - Temporal (hour, day of week, part of day)
  - Document age (`event_doc_age_hours`, `ad_doc_age_hours`)
  - Frequency counts (user/ad/campaign/publisher impressions)
  - Historical CTR aggregates (ad/campaign/publisher)
  - Simple topic match flag

### Preprocessing (Summary Only)
- Clip and `log1p` heavy-tailed numeric features
- Handle missing values (numeric → 0, categorical → `<UNK>`)
- Categorical encoding via vocabularies (`<UNK> = 0` → `cat_vocab.pkl`, `cat_num_classes.pkl`)
- Numeric scaling via `StandardScaler` → `numeric_scaler.pkl`
- 80/20 stratified train/validation split

---

## Model & Training

### CTR Model (PyTorch)
- Embeddings for each categorical feature
- Concatenation of:
  - All embeddings
  - Selected numeric features
- MLP head (e.g. 256 → 128 → 1) with:
  - ReLU activations
  - Dropout regularization

### Loss & Optimization

- **Loss**: `BCEWithLogitsLoss` (with `pos_weight` to handle severe label imbalance)
- **Optimizer**: Adam
- **Typical training config**:
  - Batch size ≈ 1024
  - Epochs ≈ 5–10 (early stopping based on validation performance)

### Feature Selection (LightGBM)

- Train a LightGBM model on a subsample of the data
- Use feature importance (gain) to identify and drop zero / near-zero importance features
- Persist final selected feature lists:
  - `final_categorical.pkl`
  - `final_numeric.pkl`

### Hyperparameter Tuning (Optuna)

**Tuned hyperparameters include:**
- Learning rate
- Hidden layer sizes & number of MLP layers
- Dropout rate
- Weight decay (L2 regularization)
- Batch size
- Embedding dimension multiplier

Optuna runs multiple trials on a data subset → the best trial configuration is then retrained on the **full** training set.

### Evaluation

- **Primary metric**: **ROC-AUC** ≥ 0.80+ on validation set
- **Secondary metrics**: LogLoss, Accuracy, Precision, Recall
- Confusion matrix & probability calibration analysis to evaluate behavior under heavy class imbalance

All metrics are computed on a stratified hold-out validation set (80/20 split).

## Inference Artifacts

All runtime dependencies required for live prediction are stored in the `models/` folder:

- `ctr_model.pth` — trained PyTorch model weights
- `cat_vocab.pkl` — categorical feature → integer vocabulary mappings
- `cat_num_classes.pkl` — number of unique categories per categorical feature
- `numeric_scaler.pkl` — fitted `StandardScaler` for numeric features
- `final_categorical.pkl` — final list of selected categorical columns
- `final_numeric.pkl` — final list of selected numeric columns

These files are loaded once at startup by `app.py` / `inference.py`.

## FastAPI Service

### Endpoints

**GET** `/health`  
Health check → returns `{"status": "ok"}`

**POST** `/predict`  
- Input: JSON payload containing the required features (IDs, timestamps, geo, etc.)
- Output: CTR probability + optional binary prediction

## Running Locally

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Start FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 3. Test endpoints
http://localhost:8000/health
http://localhost:8000/predict
```

## Docker
```bash
# Build image
docker build -t ctr-service:latest .

# Run container
docker run -p 8000:8000 ctr-service:latest
```

## AWS Deployment (EKS + Auto Scaling Group + ECR)

### 1. Push Image to ECR

```bash
# Create repository (run once)
aws ecr create-repository --repository-name ctr-service

# Login to ECR
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com

# Tag and push
docker tag ctr-service:latest <account-id>.dkr.ecr.<your-region>.amazonaws.com/ctr-service:latest
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/ctr-service:latest
```

### 2. EKS Cluster (Control Plane)

- Create an EKS cluster (via AWS Console, eksctl or AWS CLI)  
- Attach IAM role with `AmazonEKSClusterPolicy`  
- No managed node group needed — EKS is used purely as the control plane

### 3. Self-Managed Worker Nodes (EC2 + Auto Scaling Group)

- Create a Launch Template with:
  - EKS-optimized AMI
  - Instance profile that includes the following IAM policies:
    - `AmazonEKSWorkerNodePolicy`
    - `AmazonEC2ContainerRegistryReadOnly`
    - `AmazonEKS_CNI_Policy`
- Create an Auto Scaling Group using that Launch Template
- Nodes automatically join the EKS cluster as Kubernetes worker nodes
- These nodes pull images from ECR and run the FastAPI + PyTorch pods

### 4. Deploy to EKS

Run from AWS CloudShell or any machine with `kubectl` configured and access to your EKS cluster:

```bash
kubectl apply -f ctr-deployment.yaml
```

`ctr-deployment.yaml` defines: A **Deployment** (replicas, container image, ports)
`ctr-service.yaml` defines:    A **Service** with `type: LoadBalancer` (automatically creates an AWS ELB)

#### Get the external endpoint

```bash
kubectl get service ctr-service
```

#### Then call
```bash
http://<elb-hostname>/health
http://<elb-hostname>/predict
```

## Architecture Diagram (Text)

```text
Developer / Repo
      |
      ↓
EC2 Builder (Docker)
   ├─ docker build
   └─ docker push
      ↓
AWS ECR (ctr-service image)
      ↓ (pull)
   AWS EKS                 ← (Control Plane only)
      |
      ↓ (schedules pods)
Auto Scaling Group (EC2)
   (Self-managed worker nodes)
      |
      ↓ (run pods)
FastAPI + PyTorch Pod
      |
      ↓
Kubernetes Service (type: LoadBalancer)
      |
      ↓
AWS Elastic Load Balancer (ELB)
      |
      ↓
Client / User / App
```

## Cleanup (Avoid Unnecessary AWS Costs)

When you’re done, delete the following resources (in this order):
- EKS cluster  
- Auto Scaling Group + EC2 worker instances  
- Kubernetes LoadBalancer Service (automatically removes the ELB)  
- ECR repository (if no longer needed)  
- Any unused IAM roles / instance profiles  

This will stop all billing for compute, networking and storage related to the project.

## Future Improvements

- Real-time streaming ingestion (Kafka / Kinesis)  
- Automated retraining pipeline (EMR / SageMaker)  
- A/B testing between multiple model versions  
- Online feature store integration (e.g., Feast, Redis)  
- Full CI/CD for Docker images + Kubernetes manifests (GitHub Actions / CodePipeline)  
- Monitoring & alerting (Prometheus + Grafana, CloudWatch)  
- Model explainability (SHAP values in /predict response)  
- Horizontal Pod Autoscaling + Cluster Autoscaler for cost-efficient scaling  




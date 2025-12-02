# 📘 Real Time CTR Prediction System (PyTorch + FastAPI + AWS EKS)

This project implements an end-to-end **Click-Through Rate (CTR)** prediction system inspired by real-world ad ranking pipelines. It includes data preprocessing, deep learning model training, real-time inference via FastAPI, Docker containerization, and deployment on **AWS EKS with self-managed EC2 worker nodes**.

---

## 🚀 Key Features
- **PyTorch CTR Model** using embeddings + MLP  
- **Feature engineering pipeline** (categorical encoding, numeric scaling, CTR aggregates, temporal features)  
- **LightGBM feature selection**  
- **Hyperparameter tuning with Optuna**  
- **FastAPI real-time inference API**  
- **Docker + AWS ECR containerized deployment**  
- **AWS EKS control plane + EC2 ASG worker nodes**  
- **Kubernetes LoadBalancer for `/predict` endpoint**  

---

## 📊 Dataset Overview
Based on the **Outbrain Ad Click Prediction** dataset.

Core files:
- `clicks_train.csv`
- `events.csv`
- `promoted_content.csv`
- `documents_meta.csv`
- `documents_topics.csv`

> Detailed dataset exploration, statistics, and record counts are available in `training.ipynb`.

---

### 🧠 Model Overview

A production-grade **Click-Through Rate (CTR) Prediction** model built with **PyTorch**, designed for high-scale advertising/recommendation systems.

#### 1. **Embedding Layers** (Categorical Features)
- Handles **high-cardinality** features (user_id, item_id, etc.)
- Learnable dense embeddings (32–128 dim depending on frequency)
- Embedding regularization via L2 and dropout

#### 2. **Numeric Feature Processing**
- Continuous features standardized with **StandardScaler**
- Heavy-tailed count features transformed with **log1p**
- Missing values imputed with median/-1 sentinel

#### 3. **Deep Architecture** (MLP)
- Multi-layer perceptron with **ReLU** activations
- Hidden layers: configurable (e.g., 512→256→128)
- **Dropout** (0.1–0.5) for regularization
- BatchNorm after each linear layer

#### 4. **Loss Function**
- **BCEWithLogitsLoss** (numerically stable)
- **Class imbalance handling** via positive-class weighting:
  ```python
  pos_weight = (neg_samples / pos_samples)
  ```

### 5. Feature Selection with LightGBM (Smart Pruning)

- Trained a **LightGBM** model end-to-end on the raw dataset  
- Extracted feature importance using both **gain** and **split** metrics  
- Applied multi-stage filtering:  
  - Removed features with **zero importance** across 5-fold CV  
  - Kept only top ~150 features (percentile ≥ 95)  
  - Manual allow-list for business-critical fields  
- Result: **~90% noise reduction** while retaining >99.8% of predictive power  

### 6. Hyperparameter Optimization with Optuna

- **100–300 trials** using Tree-structured Parzen Estimator (TPE)  
- Optimized for **validation AUC** (early stopping @ 5 epochs no improvement)  
**Best Trial Result**  
- Validation AUC: **0.7964** ↑ (+0.016 vs baseline)  
- Training time: ~18 min/epoch on single V100  
- Inference latency: **~1.2 ms** per sample (CPU)

Ready for production deployment with TorchServe / SageMaker / Triton.

### Saved model artifacts:
- `ctr_model.pth`
- `cat_vocab.pkl`
- `cat_num_classes.pkl`
- `numeric_scaler.pkl`
- `final_categorical.pkl`
- `final_numeric.pkl`

These are loaded automatically at FastAPI startup.

---

## 📦 Repository Structure
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
│    ├──ctr-deployment.yaml   # Kubernetes Deployment (ECR)
│    └──ctr-service.yaml      # Kubernetes Service (LoadBalancer)
└── README.md                 # This file
```

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

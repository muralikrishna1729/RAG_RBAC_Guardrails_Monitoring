# ☁️ AWS Free Tier Deployment Guide ($0/Month): RBAC RAG System

This guide walks you through deploying the **RBAC RAG Chatbot System** completely within the **AWS Free Tier** ($0/month), avoiding any credit usage or unexpected billing.

---

## 📌 Free Tier Architecture Overview

| Component | AWS Resource | Free Tier Allowance | Your Config | Cost |
|---|---|---|---|---|
| **Compute** | EC2 (`t2.micro` or `t3.micro`) | 750 hours/month | 1 vCPU, 1 GiB RAM | **$0.00** |
| **Storage** | EBS General Purpose (gp3) | 30 GB total | 30 GB root storage | **$0.00** |
| **Virtual Memory** | Linux Swap File | Uses free EBS space | 3 GB Swap (4 GB total RAM) | **$0.00** |
| **Data Transfer** | Inbound / Outbound | 100 GB/month egress | Web traffic | **$0.00** |

> [!IMPORTANT]
> **Why Swap Memory is Mandatory on Free Tier (`t2.micro` / `t3.micro`)**:  
> `t2.micro` instances have **1 GiB physical RAM**. Loading PyTorch CPU, `sentence-transformers` (`all-MiniLM-L6-v2`), `cross-encoder`, ChromaDB, and Streamlit requires ~1.5 - 2.5 GB of RAM. Without a **Swap File**, Linux will kill the container with an `Out Of Memory (OOM)` error. Adding a 3GB Swap file on your free 30GB EBS disk expands total usable RAM to 4GB for $0.

---

## 🚀 Step-by-Step Free Tier EC2 Deployment

### Step 1: Launch your Free Tier EC2 Instance
1. Log into **AWS Management Console** → Go to **EC2** → Click **Launch Instance**.
2. **Name**: `rbac-rag-freetier`
3. **Application & OS Image (AMI)**: **Ubuntu Server 24.04 LTS** (marked *"Free tier eligible"*).
4. **Instance Type**: Select **`t2.micro`** (or **`t3.micro`** depending on your AWS region's free tier eligibility).
5. **Key Pair**: Select or click **Create new key pair** (`rbac-rag-key.pem`).
6. **Network Settings (Security Group)**:
   - Check **Allow SSH traffic from** -> *My IP* (or Anywhere).
   - Check **Allow HTTP traffic from the internet** (Port 80).
   - Check **Allow HTTPS traffic from the internet** (Port 443).
7. **Configure Storage**:
   - Change `8 GiB` to **`30 GiB`** (30 GB is 100% Free under AWS Free Tier).

Click **Launch Instance**.

---

### Step 2: Connect to Instance & Enable 3GB Swap File (Crucial)

SSH into your instance:
```bash
chmod 400 rbac-rag-key.pem
ssh -i rbac-rag-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>
```

Run these commands to allocate 3 GB Swap space from your free 30GB storage:
```bash
# Create 3GB swap file
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify total memory (Physical RAM + Swap = ~4GB)
free -h
```

---

### Step 3: Install Docker & Docker Compose
```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker & Docker Compose plugin
sudo apt-get install -y docker.io docker-compose-v2

# Grant ubuntu user Docker privileges
sudo usermod -aG docker ubuntu
newgrp docker
```

---

### Step 4: Clone Repository & Create `.env`

```bash
git clone https://github.com/muralikrishna1729/RAG_RBAC_Guardrails_Monitoring.git
cd RAG_RBAC_Guardrails_Monitoring

# Create environment configuration
cat << 'EOF' > .env
GROQ_API_KEY=your_actual_groq_api_key
JWT_SECRET_KEY=change-this-to-a-random-secret-key-32-chars
LANGCHAIN_TRACING_V2=false
COMPANY_DOMAIN=company.com

# Memory optimization for 1-vCPU Free Tier
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
EOF
```

---

### Step 5: Build & Run Containers ($0 Cost)

```bash
# Launch application containers using Docker Compose
docker compose up -d --build

# Monitor container build & launch status
docker compose ps
docker compose logs -f
```

---

### 🌐 Step 6: Access Your Live Application

Open your browser and navigate to:
- **Streamlit Web UI**: `http://<YOUR_EC2_PUBLIC_IP>` (Port 80 handled via Nginx)
- **FastAPI REST API Docs**: `http://<YOUR_EC2_PUBLIC_IP>/api/docs`

---

## 💡 AWS Free Tier Cost Protection Checklist

To ensure you stay **100% free** and never incur unexpected charges:

1. **Keep EBS at or under 30 GB**: AWS allows up to 30 GB of gp2/gp3 EBS volume for free per month.
2. **Stay within 750 Hours**: 1 `t2.micro` instance running 24/7 uses 744 hours in a 31-day month, fitting perfectly within the 750 free hours limit.
3. **Avoid Unattached Elastic IPs**: If you allocate an Elastic IP (EIP), keep it attached to your running EC2 instance. Unattached EIPs cost ~$0.005/hr.
4. **Billing Alarm ($1 Threshold)**:
   - AWS Console → **Billing and Cost Management** → **Preferences** → Check **Receive Free Tier Usage Alerts** and **Receive Billing Alerts**.
   - Create a CloudWatch Alarm for **$1.00** threshold to receive an email if any charge ever occurs.

---

## 🛠️ Management & Troubleshooting

### Stopping Instance when not in use (Optional)
If you want to conserve hours or stop testing, stop the instance in AWS EC2 Console. Stopping an instance halts compute hours ($0) while retaining your disk and files.

### Restarting Containers after Reboot
Docker containers are configured with `restart: unless-stopped` and will automatically resume whenever your EC2 instance boots up.

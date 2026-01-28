# Deployment Guide

This guide provides detailed instructions for deploying the Handwriting Recognition API in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Production Considerations](#production-considerations)
7. [Monitoring Setup](#monitoring-setup)
8. [Scaling and Load Balancing](#scaling-and-load-balancing)
9. [Security Configuration](#security-configuration)
10. [Backup and Recovery](#backup-and-recovery)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores (4+ cores recommended for production)
- **Memory**: 4GB+ RAM (8GB+ recommended for production)
- **Storage**: 10GB+ available space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Dependencies

- Python 3.8+
- Docker and Docker Compose
- Git
- curl (for testing)

### Optional Dependencies

- NVIDIA Docker runtime (for GPU acceleration)
- NVIDIA GPU with CUDA support
- Kubernetes cluster (for K8s deployment)

## Local Development

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd handwriting-recognition-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# API Configuration
HANDWRITING_LOG_LEVEL=DEBUG
HANDWRITING_DEBUG=true
HANDWRITING_HOST=0.0.0.0
HANDWRITING_PORT=8000

# OCR Configuration
HANDWRITING_OCR_USE_GPU=false
HANDWRITING_OCR_LANG=en
HANDWRITING_OCR_USE_ANGLE_CLS=true

# Image Processing
HANDWRITING_MAX_IMAGE_SIZE=10485760
HANDWRITING_MAX_IMAGE_DIMENSIONS=4096,4096

# Performance
HANDWRITING_MAX_CONCURRENT_REQUESTS=5
HANDWRITING_REQUEST_TIMEOUT=30

# Monitoring
HANDWRITING_ENABLE_METRICS=true
HANDWRITING_METRICS_PORT=9090
```

### 3. Run Development Server

```bash
# Run the API with auto-reload
python -m uvicorn handwriting_api.api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# Or use the provided script
python scripts/run_dev.py
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Test with sample image
curl -X POST "http://localhost:8000/recognize" \
  -F "image=@tests/fixtures/sample_handwriting.jpg" \
  -F "return_confidence=true"
```

## Docker Deployment

### Basic Docker Deployment

```bash
# Build the Docker image
docker build -t handwriting-api:latest .

# Run the container
docker run -d \
  --name handwriting-api \
  -p 8000:8000 \
  -e HANDWRITING_LOG_LEVEL=INFO \
  -e HANDWRITING_OCR_USE_GPU=false \
  handwriting-api:latest
```

### Docker Compose Deployment

#### Production Environment

```bash
# Start production services
docker-compose up -d handwriting-api redis prometheus grafana nginx

# Check service status
docker-compose ps

# View logs
docker-compose logs -f handwriting-api
```

#### GPU-Enabled Environment

```bash
# Install NVIDIA Docker runtime first
# https://github.com/NVIDIA/nvidia-docker

# Start GPU-enabled services
docker-compose --profile gpu up -d

# Verify GPU is available
docker exec handwriting-api-gpu nvidia-smi
```

#### Development Environment

```bash
# Start development services
docker-compose --profile dev up -d

# The API will auto-reload on code changes
```

### Docker Configuration Options

#### Multi-Stage Build

The Dockerfile includes multiple stages:
- `builder`: Dependency installation
- `production`: Optimized production image
- `development`: Development environment with hot reload
- `gpu`: GPU-enabled production image

#### Environment Variables

Configure via environment variables or `.env` file:

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  handwriting-api:
    environment:
      - HANDWRITING_LOG_LEVEL=INFO
      - HANDWRITING_MAX_IMAGE_SIZE=20971520  # 20MB
      - HANDWRITING_MAX_CONCURRENT_REQUESTS=20
      - HANDWRITING_REQUEST_TIMEOUT=60
    volumes:
      - ./custom-models:/app/models
      - ./logs:/app/logs
```

## Cloud Deployment

### AWS Deployment

#### ECS with Fargate

```bash
# Create ECR repository
aws ecr create-repository --repository-name handwriting-api

# Build and push image
docker build -t handwriting-api:latest .
docker tag handwriting-api:latest <aws-account>.dkr.ecr.<region>.amazonaws.com/handwriting-api:latest
docker push <aws-account>.dkr.ecr.<region>.amazonaws.com/handwriting-api:latest

# Deploy using ECS task definition
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json
aws ecs create-service --service-name handwriting-api --task-definition handwriting-api:1 --desired-count 2
```

#### ECS Task Definition (aws/ecs-task-definition.json)

```json
{
  "family": "handwriting-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "handwriting-api",
      "image": "<aws-account>.dkr.ecr.<region>.amazonaws.com/handwriting-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "HANDWRITING_LOG_LEVEL", "value": "INFO"},
        {"name": "HANDWRITING_MAX_CONCURRENT_REQUESTS", "value": "10"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/handwriting-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Application Load Balancer Setup

```bash
# Create target group
aws elbv2 create-target-group \
  --name handwriting-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id <vpc-id> \
  --health-check-path /health

# Create ALB
aws elbv2 create-load-balancer \
  --name handwriting-api-alb \
  --subnets <subnet-1> <subnet-2> \
  --security-groups <security-group>

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn <alb-arn> \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=<tg-arn>
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Google Container Registry
docker build -t gcr.io/your-project/handwriting-api:latest .
docker push gcr.io/your-project/handwriting-api:latest

# Deploy to Cloud Run
gcloud run deploy handwriting-api \
  --image gcr.io/your-project/handwriting-api:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 1 \
  --allow-unauthenticated \
  --set-env-vars "HANDWRITING_LOG_LEVEL=INFO,HANDWRITING_MAX_CONCURRENT_REQUESTS=10"
```

#### GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create handwriting-api-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --zone us-central1-a

# Deploy to GKE
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Azure Deployment

#### Container Instances

```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name handwriting-api \
  --image your-registry/handwriting-api:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --dns-name-label handwriting-api \
  --environment-variables \
    HANDWRITING_LOG_LEVEL=INFO \
    HANDWRITING_MAX_CONCURRENT_REQUESTS=10
```

#### AKS Deployment

```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name handwriting-api-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Deploy to AKS
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Kubernetes Deployment

### Basic Kubernetes Setup

```bash
# Create namespace
kubectl create namespace handwriting-api

# Apply configurations
kubectl apply -f k8s/ -n handwriting-api

# Check deployment status
kubectl get pods -n handwriting-api
kubectl get services -n handwriting-api
kubectl get ingress -n handwriting-api
```

### Kubernetes Manifests

#### Deployment (k8s/deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: handwriting-api
  labels:
    app: handwriting-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: handwriting-api
  template:
    metadata:
      labels:
        app: handwriting-api
    spec:
      containers:
      - name: handwriting-api
        image: handwriting-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: HANDWRITING_LOG_LEVEL
          value: "INFO"
        - name: HANDWRITING_OCR_USE_GPU
          value: "false"
        - name: HANDWRITING_MAX_CONCURRENT_REQUESTS
          value: "10"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: tmp
          mountPath: /app/tmp
      volumes:
      - name: logs
        emptyDir: {}
      - name: tmp
        emptyDir: {}
```

#### Service (k8s/service.yaml)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: handwriting-api-service
  labels:
    app: handwriting-api
spec:
  selector:
    app: handwriting-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

#### Ingress (k8s/ingress.yaml)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: handwriting-api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rate-limit: "10"
    nginx.ingress.kubernetes.io/rate-limit-window: "1s"
    nginx.ingress.kubernetes.io/client-max-body-size: "11m"
spec:
  rules:
  - host: handwriting-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: handwriting-api-service
            port:
              number: 80
  tls:
  - hosts:
    - handwriting-api.yourdomain.com
    secretName: handwriting-api-tls
```

### Helm Chart Deployment

```bash
# Install Helm chart
helm install handwriting-api ./helm/handwriting-api \
  --namespace handwriting-api \
  --set image.tag=latest \
  --set replicaCount=3 \
  --set resources.requests.memory=1Gi \
  --set resources.limits.memory=2Gi

# Upgrade deployment
helm upgrade handwriting-api ./helm/handwriting-api \
  --namespace handwriting-api \
  --set image.tag=v1.1.0
```

## Production Considerations

### Resource Requirements

| Environment | CPU | Memory | Storage | Network |
|-------------|-----|--------|---------|---------|
| Development | 1 core | 2GB | 10GB | 1 Gbps |
| Production (Small) | 2 cores | 4GB | 50GB | 1 Gbps |
| Production (Medium) | 4 cores | 8GB | 100GB | 10 Gbps |
| Production (Large) | 8+ cores | 16GB+ | 500GB+ | 10+ Gbps |

### Scaling Strategies

#### Horizontal Scaling
```bash
# Kubernetes HPA
kubectl autoscale deployment handwriting-api \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n handwriting-api
```

#### Vertical Scaling
```yaml
# Update resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Performance Tuning

#### OCR Engine Optimization
```env
# Tune OCR parameters
HANDWRITING_OCR_USE_ANGLE_CLS=true
HANDWRITING_OCR_USE_GPU=true
HANDWRITING_MODEL_CACHE_SIZE=2
HANDWRITING_ENABLE_MODEL_WARMUP=true
```

#### Image Processing Optimization
```env
# Optimize for throughput
HANDWRITING_MAX_IMAGE_SIZE=20971520  # 20MB
HANDWRITING_MAX_CONCURRENT_REQUESTS=20
HANDWRITING_REQUEST_TIMEOUT=60
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'handwriting-api'
    static_configs:
      - targets: ['handwriting-api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the provided dashboard JSON:
```bash
# Access Grafana
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/handwriting-api.json
```

### Key Metrics to Monitor

1. **Request Metrics**
   - Request rate (requests/second)
   - Request duration (p50, p95, p99)
   - Error rate (4xx, 5xx errors)

2. **OCR Performance**
   - OCR processing time
   - Recognition accuracy
   - Model loading time

3. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network throughput

4. **Business Metrics**
   - Images processed per minute
   - Batch processing efficiency
   - User satisfaction (if tracked)

## Scaling and Load Balancing

### Nginx Load Balancing

```nginx
upstream handwriting_api {
    least_conn;
    server handwriting-api-1:8000 weight=3;
    server handwriting-api-2:8000 weight=2;
    server handwriting-api-3:8000 weight=1;
    
    keepalive 32;
}

server {
    location / {
        proxy_pass http://handwriting_api;
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Auto-scaling Configuration

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: handwriting-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: handwriting-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### AWS Auto Scaling

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name handwriting-api-template \
  --launch-template-data file://aws/launch-template.json

# Create auto scaling group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name handwriting-api-asg \
  --launch-template LaunchTemplateName=handwriting-api-template \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 3 \
  --target-group-arns <target-group-arn>
```

## Security Configuration

### Network Security

#### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw enable
```

#### Security Groups (AWS)
```bash
# Create security group
aws ec2 create-security-group \
  --group-name handwriting-api-sg \
  --description "Security group for handwriting API"

# Allow HTTP/HTTPS
aws ec2 authorize-security-group-ingress \
  --group-name handwriting-api-sg \
  --protocol tcp \
  --port 80 \
  --source-group <load-balancer-sg>
```

### Application Security

#### API Authentication (Optional)
```python
# Add to FastAPI app
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your token validation logic
    pass

@app.post("/recognize", dependencies=[Depends(verify_token)])
async def recognize_handwriting(...):
    # Your endpoint logic
```

#### Rate Limiting
```nginx
# Enhanced rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=batch_limit:10m rate=1r/s;

limit_conn_zone $binary_remote_addr zone=addr:10m;
limit_conn_zone $server_name zone=server:10m;
```

### SSL/TLS Configuration

#### Let's Encrypt SSL
```bash
# Install certbot
certbot certonly --webroot -w /var/www/html -d yourdomain.com

# Nginx SSL configuration
server {
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
}
```

## Backup and Recovery

### Data Backup

#### Application Logs
```bash
# Create backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/handwriting-api/$DATE"

mkdir -p $BACKUP_DIR
docker exec handwriting-api tar -czf - /app/logs | cat > $BACKUP_DIR/logs.tar.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/logs.tar.gz s3://your-backup-bucket/handwriting-api/$DATE/
```

#### Configuration Backup
```bash
# Backup configuration files
tar -czf handwriting-api-config-$(date +%Y%m%d).tar.gz \
  docker-compose.yml \
  nginx.conf \
  monitoring/ \
  k8s/
```

### Disaster Recovery

#### Recovery Procedures

1. **Service Failure**
   ```bash
   # Restart services
   docker-compose restart
   
   # Or in Kubernetes
   kubectl rollout restart deployment/handwriting-api -n handwriting-api
   ```

2. **Data Loss**
   ```bash
   # Restore from backup
   aws s3 cp s3://your-backup-bucket/handwriting-api/latest/logs.tar.gz ./
   docker exec handwriting-api tar -xzf logs.tar.gz -C /
   ```

3. **Complete Environment Recovery**
   ```bash
   # Rebuild and redeploy
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Monitoring Alerts

Set up alerts for critical conditions:

```yaml
# Prometheus alerting rules
groups:
- name: handwriting-api
  rules:
  - alert: HighErrorRate
    expr: rate(handwriting_api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes{name="handwriting-api"} / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage detected"
```

This deployment guide provides comprehensive instructions for deploying the Handwriting Recognition API in various environments. Choose the deployment method that best fits your infrastructure and requirements.
version: '3.8'

services:
  backend:
    image: ${DOCKER_HUB_USERNAME}/stock-prediction-backend:latest
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,${EC2_PUBLIC_IP}
      - DATABASE_HOST=${RDS_ENDPOINT}
      - DATABASE_NAME=Stock_Predictor
      - DATABASE_USER=postgres
      - DATABASE_PASSWORD=${DB_PASSWORD}
    restart: always
    logging:
      driver: "awslogs"
      options:
        awslogs-region: "${AWS_REGION}"
        awslogs-group: "stock-prediction"
        awslogs-stream: "backend"
  
  frontend:
    image: ${DOCKER_HUB_USERNAME}/stock-prediction-frontend:latest
    ports:
      - "80:80"
    restart: always
    logging:
      driver: "awslogs"
      options:
        awslogs-region: "${AWS_REGION}"
        awslogs-group: "stock-prediction"
        awslogs-stream: "frontend"
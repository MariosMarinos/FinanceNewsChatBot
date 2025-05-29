# 📦 Streamlit App – Docker Setup Guide

This guide explains how to set up and run the Streamlit app using Docker and Docker Compose.
---

## 🚀 Prerequisites

If using Ubuntu make sure the following are installed on the machine:

### 1. Install Docker
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. Install Docker Compose Plugin (v2)
```bash
sudo apt install docker-compose-plugin -y
```

> ✅ Test installation:
```bash
docker --version
docker compose version
```

---

## 📁 Project Structure

```
project/
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── main.py              # your Streamlit app
│   └── ...
```

---

## ⚙️ 1. Build and Start the App

In the project root directory:

```bash
docker compose up --build 
```
or 
```bash
docker-compose up --build
```
if using previous version.

This will:
- Build the Docker image
- Start the Streamlit app inside a container
- Expose it at **http://localhost:8501**

---

## 🛑 2. Stop the App

To stop the app and clean up containers:

```bash
docker compose down
```

---

## ✅ Optional: Run Without Rebuilding

Once built, you can start the app again faster:

```bash
docker compose up
```

---

## 🧪 Debugging

- Check logs: `docker compose logs`
- Rebuild from scratch: `docker compose up --build --force-recreate`
- Shell into container: `docker exec -it <container_id> bash`

---
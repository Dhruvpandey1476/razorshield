# RazorShield -- single-container deployment.
# Build: docker build -t razorshield .
# Run:   docker run -p 8000:8000 -e RAZORPAY_WEBHOOK_SECRET=... razorshield
#
# The pipeline runs once at build time so the image starts with
# artifacts/ already populated -- no first-request delay, and every
# deploy is reproducible from the same pinned dependency versions.
FROM python:3.12-slim

WORKDIR /app

# System deps for matplotlib (SHAP plot rendering) and xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate data, train the model, run the agent on all spike events --
# baked into the image so the container is immediately useful on start.
RUN chmod +x run_pipeline.sh && ./run_pipeline.sh

EXPOSE 8000

# Render/Railway/Fly all set $PORT; default to 8000 for local docker run.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]

# mixtape demo on a Hugging Face Space (Docker SDK, free CPU tier: 2 vCPU, 16 GB RAM).
# The playlist data is NOT in this image: the app downloads it at start-up from the
# private dataset repo named by MIXTAPE_DATA_REPO, using the HF_TOKEN Space secret.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only torch keeps the image small; then the rest of the CPU stack
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user numpy pandas pyarrow scipy scikit-learn networkx pyyaml tqdm \
        "streamlit>=1.40" "sentence-transformers>=3.0" huggingface_hub

# bake the title encoder into the image so start-up doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user configs ./configs
COPY --chown=user app ./app
RUN pip install --no-cache-dir --user --no-deps .

ENV MIXTAPE_BACKEND=cpu
EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--server.enableXsrfProtection", "false", "--server.enableCORS", "false", \
     "--browser.gatherUsageStats", "false"]

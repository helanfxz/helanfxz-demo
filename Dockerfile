FROM python:3.12-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8010 \
    AIGC_DISABLE_LLM=1 \
    AIGC_DISABLE_VIDEO_MODEL=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY prompt_skill_library ./prompt_skill_library
COPY docs ./docs
COPY task_creation_demo_app.py video_task_module.py commerce_style_templates.py ./

RUN python -m pip install --retries 10 --timeout 120 --upgrade pip \
    && python -m pip install --retries 10 --timeout 120 --prefer-binary -e .

EXPOSE 8010

CMD ["python", "task_creation_demo_app.py"]

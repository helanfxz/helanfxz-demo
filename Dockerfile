FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8010 \
    AIGC_DISABLE_LLM=1 \
    AIGC_DISABLE_VIDEO_MODEL=1

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY agent ./agent
COPY prompt_skill_library ./prompt_skill_library
COPY docs ./docs
COPY task_creation_demo_app.py video_task_module.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 8010

CMD ["python", "task_creation_demo_app.py"]

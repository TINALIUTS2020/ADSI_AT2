FROM tensorflow/tensorflow:2.12.0

ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG WORKING_DIRECTORY=default

ARG PORT=80
ARG CURRENT_CONTAINER=api_dev


ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=OFF
ENV PIP_DISABLE_VERSION_CHECK=1

ENV PORT=$PORT
ENV CURRENT_CONTAINER=$CURRENT_CONTAINER

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
RUN apt-get update && apt-get install -y git



USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
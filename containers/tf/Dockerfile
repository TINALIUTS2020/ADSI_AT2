FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG WORKING_DIRECTORY=default

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=OFF
ENV PIP_DISABLE_VERSION_CHECK=1

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
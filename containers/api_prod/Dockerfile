FROM tensorflow/tensorflow:2.12.0

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001
ARG WORKING_DIRECTORY=/app
ARG PORT=80
ARG CURRRENT_CONTAINER=api_prod

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=OFF
ENV PIP_DISABLE_VERSION_CHECK=1

ENV PORT=$PORT
ENV CURRRENT_CONTAINER=${CURRRENT_CONTAINER}

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

COPY ./containers/api_prod/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

COPY ./app /app
COPY ./src /app/src
RUN python -m pip install /app
RUN ["chmod", "+x", "/app/run.sh"]
USER $USERNAME

WORKDIR /app

CMD /app/run.sh
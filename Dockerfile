# Use the official Python 3.8 slim image as the base image
FROM python:3.11-slim

# install nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# install node 22
RUN nvm install 22
# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Expose port 8080 for the FastAPI
EXPOSE 8080

CMD  bash -c 'uv run app.py'
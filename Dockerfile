FROM python:3.12-slim

# Install uv for lightning-fast builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Copy the pyproject.toml and README (required for standard hatchling builds)
COPY pyproject.toml README.md ./

# Copy the core library source code
COPY src/ ./src/

# Install the seroepi package and all its [app] dependencies into the system python using uv
RUN uv pip install --system .[app]

# Copy the actual Shiny application directory
COPY app/ ./app/

# Expose the default Shiny port
EXPOSE 8000

# Run the Shiny application
CMD ["shiny", "run", "app:app", "--host", "0.0.0.0", "--port", "8000"]

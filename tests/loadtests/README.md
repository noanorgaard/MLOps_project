# Load Testing for AI vs Human Classifier API

This directory contains load testing utilities for the FastAPI application.

## Tools Used

### 1. Locust (Interactive Load Testing)
[Locust](https://locust.io/) is a modern load testing tool that provides a web UI for monitoring and controlling tests.

### 2. Pytest + Requests (Automated Load Testing)
Traditional pytest-based load tests that can be integrated into CI/CD pipelines.

## Running Load Tests

### Prerequisites

Ensure the required packages are installed:
```bash
uv add locust requests
```

Make sure your API is running:
```bash
# Start the API locally
uv run uvicorn ai_vs_human.api:app --host 0.0.0.0 --port 8000
```

Or using Docker:
```bash
docker build -f dockerfiles/api.dockerfile -t ai-vs-human-api .
docker run -p 8000:8000 --env-file .env ai-vs-human-api
```

### Method 1: Locust (Recommended for exploration)

Start the Locust web interface:
```bash
uv run locust -f tests/loadtests/locustfile.py --host=http://localhost:8000
```

Then open your browser to http://localhost:8000 and:
1. Set the number of users to simulate (e.g., 10, 50, 100)
2. Set the spawn rate (users per second, e.g., 5)
3. Click "Start swarming"

The web UI will show real-time statistics including:
- Requests per second (RPS)
- Response times (min, max, avg, median, P95, P99)
- Failure rate
- Number of concurrent users

#### Running Locust Headless

For automated testing without the web UI:
```bash
# Run with 50 users, spawning 5 per second, for 2 minutes
uv run locust -f tests/loadtests/locustfile.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 2m \
    --headless \
    --html reports/load_test_report.html
```

#### Stress Testing with Locust

Use the `StressTestUser` class for aggressive load testing:
```bash
uv run locust -f tests/loadtests/locustfile.py \
    --host=http://localhost:8000 \
    StressTestUser
```

## Load Test Scenarios

### 1. Normal User Behavior (`AIvsHumanUser` in locustfile.py)
Simulates typical user behavior with:
- 70% prediction requests
- 20% health checks
- 10% root endpoint access
- 1-3 second wait between requests

### 2. Stress Testing (`StressTestUser` in locustfile.py)
Aggressive testing with:
- Rapid consecutive requests (0.1-0.5s between requests)
- Minimal wait time
- Tests system limits and error handling

## Performance Metrics

The tests track several key metrics:

- **Response Time**: Time taken for the API to respond
  - Average, min, max, P95, P99
- **Throughput**: Requests per second (RPS)
- **Success Rate**: Percentage of successful requests
- **Concurrency**: Number of simultaneous users/requests
- **Error Rate**: Percentage of failed requests

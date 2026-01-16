# Project description (ai_vs_human)

## Overall goal of the project
To detect if images are AI generated or not.

##  What data are you going to run on
We have found our data on Kaggle:
https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art

The metadata is:
- Paintings, photography and AI generated
- 970 images; 536 AI images and 434 Human images
- Formats: .jpg, .png and .jpeg
- 503.78 MB (Version 1)

## What models do you expect to use
ResNet18, finetuned.

## Setup

### Environment Configuration

**Required:** Create a `.env` file in the project root with your Weights & Biases API key:

```bash
cp .env.template .env
```

Then edit `.env` and add your actual API key:
```
WANDB_API_KEY=your_key_here
```

Get your API key from: https://wandb.ai/authorize

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── loadtests/           # Load testing suite
│   │   ├── locustfile.py
│   │   ├── test_load.py
│   │   └── README.md
│   ├── performancetests/
│   ├── unittests/
│   │   ├── test_api.py
│   │   ├── test_data.py
│   │   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Testing

### Unit Tests
```bash
# Run all tests with coverage
uv run invoke test
```

### Load Testing

The project includes comprehensive load testing for the API:

```bash
# Install load testing dependencies
uv sync

# Start the API first
uv run invoke api

# Then run load tests (in another terminal)

# Option 1: Locust headless mode (automated)
uv run invoke load-test-locust --users=50 --rate=5 --time=2m

# Option 2: Locust web UI (interactive)
uv run invoke load-test-locust-web
# Then open http://localhost:8089 in your browser
```

For detailed information about load testing scenarios see [tests/loadtests/README.md](tests/loadtests/README.md).
Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).



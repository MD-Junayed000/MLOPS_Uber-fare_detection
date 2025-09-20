# src/zenml_run.py
from zenml import pipeline, step

@step
def hello():
    return "ok"

@pipeline
def demo():
    hello()

if __name__ == "__main__":
    demo()()

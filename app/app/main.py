"""The entrypoint for the application."""

from fastapi import FastAPI


app = FastAPI(title="Video LLM-based search engine API")


@app.get("/")
def main_route():
    return {"message": "Welcome to the Video LLM-based search engine API!"}
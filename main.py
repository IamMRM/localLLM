import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "start:app",
        reload=True,
        host="0.0.0.0",
        port=8000
    )
    
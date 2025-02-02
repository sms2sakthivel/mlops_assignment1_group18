from fastapi import FastAPI
from Service import router
from fastapi.middleware.cors import CORSMiddleware

# Step 1: Create a FastAPI app
app = FastAPI()

app.add_middleware(CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],)

app.include_router(router.router)

@app.get("/")
async def root():
    return {"message": "Group 18 Assignment 1 API"}
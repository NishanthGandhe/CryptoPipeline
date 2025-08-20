"""
FastAPI wrapper for CryptoPipeline backend
Converts the Python scripts into web API endpoints for deployment
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CryptoPipeline API",
    description="AI-powered cryptocurrency price prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_status = {
    "is_running": False,
    "last_run": None,
    "last_error": None,
    "progress": ""
}

class PipelineResponse(BaseModel):
    success: bool
    message: str
    timestamp: str
    details: Dict[str, Any] = None
    error: Dict[str, Any] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CryptoPipeline API is running",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Get current pipeline status"""
    return {
        "is_running": pipeline_status["is_running"],
        "last_run": pipeline_status["last_run"],
        "last_error": pipeline_status["last_error"],
        "progress": pipeline_status["progress"]
    }

async def run_python_script(script_name: str) -> tuple[str, str, int]:
    """Run a Python script asynchronously"""
    try:
        process = await asyncio.create_subprocess_exec(
            "python", script_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app"  # Railway working directory
        )
        
        stdout, stderr = await process.communicate()
        return stdout.decode(), stderr.decode(), process.returncode
    except Exception as e:
        return "", str(e), 1

async def run_pipeline_background():
    """Run the complete data pipeline in background"""
    try:
        pipeline_status["is_running"] = True
        pipeline_status["progress"] = "Starting data ingestion..."
        logger.info("Starting data pipeline...")
        
        stdout, stderr, returncode = await run_python_script("ingest.py")
        if returncode != 0:
            raise Exception(f"Ingestion failed: {stderr}")
        
        logger.info("Data ingestion completed successfully")
        pipeline_status["progress"] = "Data ingestion complete. Starting ML training..."
        
        stdout, stderr, returncode = await run_python_script("generate_insight.py")
        if returncode != 0:
            raise Exception(f"Insight generation failed: {stderr}")
        
        logger.info("ML training and prediction completed successfully")
        pipeline_status["progress"] = "Pipeline completed successfully"
        pipeline_status["last_run"] = datetime.utcnow().isoformat()
        pipeline_status["last_error"] = None
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        pipeline_status["last_error"] = str(e)
        pipeline_status["progress"] = f"Pipeline failed: {str(e)}"
    finally:
        pipeline_status["is_running"] = False

@app.post("/refresh-data", response_model=PipelineResponse)
async def refresh_data(background_tasks: BackgroundTasks):
    """Trigger the complete data pipeline refresh"""
    
    if pipeline_status["is_running"]:
        raise HTTPException(
            status_code=409, 
            detail="Pipeline is already running. Please wait for completion."
        )
    
    background_tasks.add_task(run_pipeline_background)
    
    return PipelineResponse(
        success=True,
        message="Data pipeline started successfully",
        timestamp=datetime.utcnow().isoformat(),
        details={
            "estimated_duration": "10-15 minutes",
            "steps": ["Data ingestion", "ML model training", "Forecast generation"],
            "note": "Check /status endpoint for progress updates"
        }
    )

@app.post("/refresh-data-sync", response_model=PipelineResponse)
async def refresh_data_sync():
    
    if pipeline_status["is_running"]:
        raise HTTPException(
            status_code=409, 
            detail="Pipeline is already running. Please wait for completion."
        )
    
    try:
        pipeline_status["is_running"] = True
        pipeline_status["progress"] = "Starting data ingestion..."
        logger.info("Starting synchronous data pipeline...")
        
        # Step 1: Data Ingestion
        stdout1, stderr1, returncode1 = await run_python_script("ingest.py")
        if returncode1 != 0:
            raise Exception(f"Ingestion failed: {stderr1}")
        
        logger.info("Data ingestion completed successfully")
        pipeline_status["progress"] = "Data ingestion complete. Starting ML training..."
        
        # Step 2: ML Model Training and Prediction
        stdout2, stderr2, returncode2 = await run_python_script("generate_insight.py")
        if returncode2 != 0:
            raise Exception(f"Insight generation failed: {stderr2}")
        
        logger.info("ML training and prediction completed successfully")
        pipeline_status["progress"] = "Pipeline completed successfully"
        pipeline_status["last_run"] = datetime.utcnow().isoformat()
        pipeline_status["last_error"] = None
        
        return PipelineResponse(
            success=True,
            message="Data pipeline completed successfully",
            timestamp=datetime.utcnow().isoformat(),
            details={
                "duration": "Pipeline completed synchronously",
                "steps_completed": ["Data ingestion", "ML model training", "Forecast generation"],
                "ingestion_output": stdout1[-500:] if stdout1 else "No output",
                "ml_output": stdout2[-500:] if stdout2 else "No output"
            }
        )
        
    except Exception as e:
        logger.error(f"Synchronous pipeline failed: {e}")
        pipeline_status["last_error"] = str(e)
        pipeline_status["progress"] = f"Pipeline failed: {str(e)}"
        
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
    finally:
        pipeline_status["is_running"] = False

@app.post("/ingest-only")
async def ingest_only():
    if pipeline_status["is_running"]:
        raise HTTPException(
            status_code=409, 
            detail="Pipeline is already running"
        )
    
    try:
        pipeline_status["is_running"] = True
        pipeline_status["progress"] = "Running data ingestion..."
        
        stdout, stderr, returncode = await run_python_script("ingest.py")
        
        if returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {stderr}"
            )
        
        return PipelineResponse(
            success=True,
            message="Data ingestion completed successfully",
            timestamp=datetime.utcnow().isoformat(),
            details={"stdout": stdout[-1000:], "stderr": stderr[-500:] if stderr else None}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline_status["is_running"] = False

@app.post("/generate-forecasts")
async def generate_forecasts():
    if pipeline_status["is_running"]:
        raise HTTPException(
            status_code=409, 
            detail="Pipeline is already running"
        )
    
    try:
        pipeline_status["is_running"] = True
        pipeline_status["progress"] = "Generating forecasts..."
        
        stdout, stderr, returncode = await run_python_script("generate_insight.py")
        
        if returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Forecast generation failed: {stderr}"
            )
        
        return PipelineResponse(
            success=True,
            message="Forecast generation completed successfully",
            timestamp=datetime.utcnow().isoformat(),
            details={"stdout": stdout[-1000:], "stderr": stderr[-500:] if stderr else None}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline_status["is_running"] = False

@app.get("/health")
async def health_check():
    try:
        # Check if we can import required modules
        import psycopg
        import pandas as pd
        import xgboost as xgb
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "psycopg": "available",
                "pandas": "available", 
                "xgboost": "available"
            },
            "environment": {
                "python_version": os.sys.version,
                "working_directory": os.getcwd()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

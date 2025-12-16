from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import asyncio
from typing import List, Dict
import json

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ControlParams(BaseModel):
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 1.0
    dt: float = 0.01

class NestedSystem:
    def __init__(self, num_loops: int = 3):
        self.loops = [ControlParams() for _ in range(num_loops)]
        self.states = [{"output": 0.0, "integral": 0.0, "prev_error": 0.0} 
                      for _ in range(num_loops)]
        
    def compute_step(self, target: float) -> List[Dict]:
        """Run one simulation step through all loops"""
        results = []
        current_target = target
        
        for i, (params, state) in enumerate(zip(self.loops, self.states)):
            # PID calculation
            error = current_target - state["output"]
            state["integral"] += error * params.dt
            derivative = (error - state["prev_error"]) / params.dt
            
            pid_output = (
                params.kp * error +
                params.ki * state["integral"] +
                params.kd * derivative
            )
            
            # First-order system response
            tau = 0.1 * (2 ** i)  # Each loop gets progressively slower
            alpha = params.dt / (tau + params.dt)
            state["output"] = alpha * pid_output + (1 - alpha) * state["output"]
            state["prev_error"] = error
            
            results.append({
                "loop": i,
                "output": state["output"],
                "error": error,
                "target": current_target
            })
            
            current_target = state["output"]  # Pass to next loop
            
        return results

system = NestedSystem()

@app.get("/")
def read_root():
    return {"message": "Control System Backend"}

@app.post("/update-loop/{loop_index}")
async def update_loop(loop_index: int, params: ControlParams):
    system.loops[loop_index] = params
    return {"status": "updated"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate at 20Hz
            await asyncio.sleep(0.05)
            step_result = system.compute_step(1.0)
            
            # Prepare audio data
            audio_data = []
            for i, result in enumerate(step_result):
                freq = 220 + result["output"] * 880
                audio_data.append({
                    "loop": i,
                    "frequency": freq,
                    "gain": 0.1 + abs(result["error"]) * 0.3
                })
            
            await websocket.send_json({
                "control_signals": step_result,
                "audio": audio_data,
                "timestamp": asyncio.get_event_loop().time()
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

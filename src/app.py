from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch, os
from train import MyCNN
 
ml = {}
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    model = MyCNN()
    model_path = os.path.join(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'), 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    ml['model'] = model
    yield
    ml.clear()
 
app = FastAPI(lifespan=lifespan)
 
class PredictRequest(BaseModel):
    image: list  # [C, H, W] pixel values
 
@app.get('/ping')
def health():                                   # SageMaker health check
    return {'status': 'healthy'} if ml else HTTPException(503)
 
@app.post('/invocations')
def predict(req: PredictRequest):               # SageMaker routes all requests here
    tensor = torch.tensor(req.image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(ml['model'](tensor), dim=1).squeeze()
    return {'predicted_class': int(probs.argmax()), 'confidence': float(probs.max())}

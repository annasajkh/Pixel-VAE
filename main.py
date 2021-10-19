from vae import *

from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
from matplotlib import cm

import numpy as np
import uvicorn
import threading
import requests
import time

def main_thread():
  app = FastAPI()

  model = VAE()
  model.load_state_dict(torch.load("pixel_vae_model.pkl", map_location=torch.device("cpu")))

  @app.get("/")
  def home():
    return "Alive"

  @app.get("/generate")
  async def main():
    img = Image.fromarray(np.uint8(model.decode(torch.randn(1, 300)).detach().reshape(64, 64, 3) * 255))
    img.resize((300,300), Image.NEAREST).save("generated.png")
    return FileResponse("generated.png")

  if __name__ == "__main__":
    while True:
      try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
      except:
        pass

def ping_thread():
  while True:
    requests.get("https://Pixel-VAE.annasvirtual.repl.co")
    time.sleep(3)

threading.Thread(target=main_thread).start()
threading.Thread(target=ping_thread).start()
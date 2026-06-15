# test_model.py

from Brain_Tumour_Segmentation import inference, utils
from model import load_model

model = load_model(
    "weights/best_attunet.pth"
)

print("Model Loaded Successfully")

# cd ~/Desktop/Projects/Brain_Tumour_Segmentation
# verify :  pwd
#           ls

#pwd
#ls 

# should see : app.py
#              model.py
#              inference.py
#              utils.py
#              templates/
#              static/
#              weights/

# python -c "import torch; print(torch.__version__)"
# should see : 2.12.0

# now run python app.py
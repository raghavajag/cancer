from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64

app = Flask(__name__)

class CancerClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(CancerClassifier, self).__init__()
        self.model = models.resnet50(weights=None)
        # Modify first conv layer for single-channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

class EnsemblePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.classes = [
            'lung_benign', 'lung_malignant',
            'breast_benign', 'breast_malignant',
            'cervical_benign', 'cervical_malignant'
        ]
        
        # Define model paths
        self.model_paths = {
            'r': 'best_model_r.pth',
            'g': 'best_model_g.pth',
            'b': 'best_model_b.pth',
            'gray': 'best_model_gray.pth'
        }
        
        # Initialize models
        try:
            self.models = {}
            for key, path in self.model_paths.items():
                self.models[key] = self.load_model(path)
            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def load_model(self, model_path):
        """Load a model from file"""
        try:
            model = CancerClassifier()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            model.model.load_state_dict(new_state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            raise

    def process_image(self, image, channel=None):
        """Process image for prediction"""
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if channel == 'r':
                channel_image = image.getchannel('R')
            elif channel == 'g':
                channel_image = image.getchannel('G')
            elif channel == 'b':
                channel_image = image.getchannel('B')
            else:  # Grayscale
                channel_image = image.convert('L')
            
            return self.transform(channel_image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def predict(self, image):
        """Make predictions using ensemble of models"""
        try:
            with torch.no_grad():
                predictions = {}
                # Get predictions from each model
                for channel, model in self.models.items():
                    input_tensor = self.process_image(image, channel)
                    output = model(input_tensor)
                    predictions[channel] = torch.argmax(output, dim=1).item()
                
                # Perform majority voting
                pred_list = list(predictions.values())
                final_pred = max(set(pred_list), key=pred_list.count)
                votes = pred_list.count(final_pred)
                
                # Calculate confidence for top 3 predictions
                pred_counts = {}
                for pred in pred_list:
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                
                # Sort by vote count
                sorted_preds = sorted(pred_counts.items(), key=lambda x: -x[1])
                
                results = []
                for pred, count in sorted_preds[:3]:
                    results.append({
                        'class': self.classes[pred],
                        'probability': (count / len(self.models)) * 100,
                        'votes': count
                    })
                
                return results
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

# Initialize predictor
try:
    predictor = EnsemblePredictor()
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        try:
            # Read and process the image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get predictions
            predictions = predictor.predict(image)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return render_template('index.html', 
                                 predictions=predictions,
                                 image=img_str)
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
import torch
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image


class PureCLIPClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.diagnostic_texts = [
            "Normal cognitive function with clear speech patterns and coherent storytelling",
            "Impaired cognitive function with disrupted speech patterns and incoherent storytelling indicating dementia"
        ]

    def load_spectrograms(self, spectrograms_folder="./DementiaBank/spectrograms", output_size=(224, 224)):
        spectrograms = {'Control': {}, 'Dementia': {}}
        
        for folder_type in ['Control', 'Dementia']:
            folder_path = f"{spectrograms_folder}/{folder_type}"
            
            png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

            print(f"Loading {len(png_files)} {folder_type} spectrograms...")
            
            for png_file in png_files:
                file_id = png_file.replace('.png', '')
                image_path = f"{folder_path}/{png_file}"
                
                try:
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize(output_size)
                    spectrograms[folder_type][file_id] = img
                except Exception as e:
                    print(f"Error loading {png_file}: {e}")
        
        print(f"Loaded {len(spectrograms['Control'])} control + {len(spectrograms['Dementia'])} dementia spectrograms")
        return spectrograms
    
    def classify_with_descriptions(self, spectrograms):
        results = []
        
        for category in spectrograms:
            for filename, spectrogram in spectrograms[category].items():
                inputs = self.clip_processor(
                    text=self.diagnostic_texts, 
                    images=spectrogram, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = inputs.to(self.device)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                prediction = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs, dim=1).values.item()
                
                results.append({
                    'filename': filename,
                    'true_label': category,
                    'predicted_label': 'Control' if prediction == 0 else 'Dementia',
                    'confidence': confidence,
                    'probabilities': probs.detach().cpu().numpy()
                })
        
        return results


def main():
    clip_classifier = PureCLIPClassifier()
    spectrograms = clip_classifier.load_spectrograms()
    results = clip_classifier.classify_with_descriptions(spectrograms)
    
    correct = sum(1 for r in results if r['true_label'] == r['predicted_label'])
    total = len(results)
    print(f"Pure CLIP Accuracy: {correct/total:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
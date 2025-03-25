# Usage
### Generate the image and speech data from the text data
```bash
python dataset_preprocessing.py --to_modality image
python dataset_preprocessing.py --to_modality speech
```
### Preprocess the image and speech data to embedding with clip and whisper encoder, train the image and speech aligner
```bash
python train_aligner.py --align_modality image
python train_aligner.py --align_modality speech
```
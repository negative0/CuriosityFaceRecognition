## RESTful API for celebrity face recognition

### Dependencies:
python3
numpy  
opencv-python  
pillow  
tensorflow  
flask  

### Endpoints:
"/": A webpage for inference of an image, file param should be passed with the image

"/api/v1.0/task": Get results of the prediction in JSON format

#### Using curl for inference:  
```bash
curl -F "file=@t.jpeg" http://localhost:5003/api/v1.0/task    
```
t.jpeg is the name of the file


Output Format:
```json
{
  "Ok": "File Uploaded", 
  "result": {
    "priyanka chopra": 0.999988317489624, 
    "shahrukh khan": 1.1627473213593476e-05
  }
}
```

### Trained to recognize following faces:
1. Shahrukh Khan
1. Priyanka chopra

## To run 

Run the run.py file with python3
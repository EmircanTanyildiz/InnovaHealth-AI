from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import os
import pickle

app = Flask(__name__, template_folder='/Users/emircantanyildiz/Desktop/Python/python_temelleri/MicrosoftProjects')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model_path = '/Users/emircantanyildiz/Desktop/Python/python_temelleri/MicrosoftProjects/model.p'
model = pickle.load(open(model_path, 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(file_path)
        
        img = Image.open(file_path)
        img = img.resize((15, 15))

        img_array = np.array(img)

        
        if img_array.ndim == 3 and img_array.shape[2] == 3: 
            img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            img_gray = img_array 


        img_flatten = img_gray.flatten().reshape(1, -1)

        img_flatten = img_flatten / 255.0
        

        prediction = model.predict(img_flatten)
        
        result = "Sağlıklı" if prediction[0] == 0 else "Tümörlü"
        
        return render_template('result.html', result=result)  

if __name__ == '__main__':
    app.run(debug=True)

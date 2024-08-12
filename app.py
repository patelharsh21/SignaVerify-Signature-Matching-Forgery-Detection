

from flask import Flask, render_template, request
from flask_cors import CORS
from utils import process
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    note = ''
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            note = 'Please upload both images.'
        else:
            image1 = request.files['image1'].read()
            image2 = request.files['image2'].read()

            if image1 and image2:
                # Process the images
                res = process(image1, image2)
                if res == 1:
                    note = 'Forged Signature'
                else:
                    note = 'Real Signature'

    return render_template('home.html', note=note)

if __name__ == '__main__':
    app.run(debug=True)

import flask
import urllib.request
import os
import werkzeug

from flask import Flask, flash, request, redirect, url_for, render_template


app = Flask(_name_)
 
UPLOAD_FOLDER ='/Users/nitin/Desktop/SD_project/Img/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    print("enteered")
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/',methods=['GET','POST'])
def intialdisplay():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("no file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename=='':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            flash('Image success uploaded')
            f1=(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return render_template('display1.html',filename=file,file1=f1,msg="done")
    else:
        return render_template("initial.html",msg="notdone") 




if _name_ == "_main_":
    app.run(debug=True)

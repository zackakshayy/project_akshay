__author__ = 'kai'

from flask import Flask, render_template, request
from collections import defaultdict
import requests, json
import pickle
import pandas as pd
import numpy as np
# import csv
import os

app = Flask(__name__)      

def prediction(loaded_model,inp):
    X_test=inp[:-1]
    Y_test=inp[-1]

    ynew = loaded_model.predict(np.array(X_test).reshape(1,-1))
    ynew_prob = loaded_model.predict_proba(np.array(X_test).reshape(1,-1))
    #print('inp',ynew, ynew_prob)
    
    prob1 = round(ynew_prob[0][0]*100,1)
    prob2 = round(ynew_prob[0][1]*100,1)
    
    return ynew[0], prob1, prob2

@app.route('/getRatios',methods=['POST'])
def returnRatios():
    return render_template(request.form['cname']+'.html')

@app.route('/')
def index():
    df = pd.read_csv("ui_test.csv")
    cnames = df['cname']
    count = 0
    retstr = ''
    temp = open("templates/index2.html","r").read()
    filename = 'comp'
    loaded_model = pickle.load(open(filename, 'rb'))
    with open("templates/ret.html", "w+") as retf:
        retf.write("<b></b>")
    attlist = ["Attr1","Attr3","Attr4","Attr5","Attr8","Attr9","Attr12","Attr13","Attr15","Attr16","Attr19","Attr20","Attr21","Attr22","Attr24","Attr27","Attr28","Attr30","Attr32","Attr36","Attr37","Attr41","Attr47","Attr51","Attr53","Attr59","Attr65","Attr66"]
    formlist = ["net profit / total assets","working capital / total assets","current assets / short-term liabilities","[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365","book value of equity / total liabilities","sales / total assets","gross profit / short-term liabilities","(gross profit + depreciation) / sales","(total liabilities * 365) / (gross profit + depreciation)","(gross profit + depreciation) / total liabilities","gross profit / sales","(inventory * 365) / sales","sales (n) / sales (n-1)","profit on operating activities / total assets","gross profit (in 3 years) / total assets","profit on operating activities / financial expenses","working capital / fixed assets","(total liabilities - cash) / sales","(current liabilities * 365) / cost of products sold","total sales / total assets","(current assets - inventories) / long-term liabilities","total liabilities / ((profit on operating activities + depreciation) * (12/365))","(inventory * 365) / cost of products sold","short-term liabilities / total assets","equity / fixed assets","long-term liabilities / equity","sales/working capital","EBDA/(Total Liabilities) + ((Depreciation + Tax + Interest))/(Working Capital)"]
    for company in df.values.tolist():
        cname = company[-1]
        if os.path.exists('templates/'+cname+'.html'):
            os.remove('templates/'+cname+'.html')
        tabstr = '''
                <div class="section-top-border">
                        <h3 class="mb-30 title_color">Table</h3>
                        <div class="progress-table-wrap">
                            <div class="progress-table">
                                <div class="table-head">
                                    <div class="serial">Attribute Name</div>
                                    <div class="percentage">Value</div>
                                </div>'''
        inp = company[:29]
        for i in range(len(attlist)):
            tabstr += '''
                    <div class="table-row">
                                    <div class="serial">{0}</div>
                                    <div class="percentage">{1}</div>
                                </div>'''.format(formlist[i], round(company[i], 5))
        cl, p1, p2=prediction(loaded_model,inp)
        tabstr += '''
                <div class="table-row">
                    <form method='POST' action='/predict'>
                        <input type='hidden' name='cname' value='{0}'>
                        <button class="btn sub-btn"><h4>SUBMIT</h4></button>
                    </form>
                </div>
                </div></div></div>
                '''.format(cname)
        tabstr = temp.format(tabstr)
        with open('templates/'+cname+'.html', 'w+') as tabfile:
            tabfile.write(tabstr)
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    outstr = '''
                <div class="row m0 right_dir">
                <div class="col-lg-6 p0">
                    <div class="success_img">
                        <img src="static/img/{0}.jpg" alt="">
                    </div>
                </div>
                <div class="col-lg-6 p0">
                    <div class="mission_text">
                            <a href="#" class="genric-btn primary e-large"><h4>{1}</h4></a>
                    </div>
                </div>
            </div>
        '''
    df = pd.read_csv("ui_test.csv")
    loaded_model = pickle.load(open('comp', 'rb'))
    cl = 0
    p1 = 0
    p2 = 0
    for company in df.values.tolist():
        cname = company[-1]
        if cname == request.form['cname']:
            cl, p1, p2=prediction(loaded_model,company[:29])
            break
    if cl == 0:
        outstr += outstr.format('happy_smiley', "The model has predicted non-bankruptcy with a probability of " + str(p1))
    else:
        outstr += outstr.format('sad_smiley', "The model has predicted bankruptcy with a probability of " + str(p2))
    return open('templates/predicted.html', 'r').read().replace("--outstr--", outstr)


@app.route('/64ratios')
def the64ratios():
    retstr = open('templates/64ratios.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/proposedmodel')
def proposedmodel():
    retstr = open('templates/proposedmodel.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/ratiocategorization')
def ratiocategorization():
    retstr = open('templates/ratiocategorization.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/featurecorrelation')
def featurecorrelation():
    retstr = open('templates/featurecorrelation.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/kfold')
def kfold():
    retstr = open('templates/kfold.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/top10features')
def top10features():
    retstr = open('templates/top10features.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/correlation')
def correlation():
    retstr = open('templates/correlation.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/featureexplanation')
def featureexplanation():
    retstr = open('templates/featureexplanation.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/testcases')
def testcases():
    retstr = open('templates/testcases.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/references')
def references():
    retstr = open('templates/references.html', 'r').read()
    return open("templates/predicted.html", "r").read().replace("--outstr--", retstr)

@app.route('/industryA')
def industryA():
    return render_template("industryA.html")

@app.route('/industryB')
def industryB():
    return render_template("industryB.html")

@app.route('/industryC')
def industryC():
    return render_template("industryC.html")

@app.route('/industryD')
def industryD():
    return render_template("industryD.html")



if __name__ == '__main__':
	app.run()
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Kn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_confusion_matrix
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import matplotlib 
matplotlib.use('Agg')
import os
import io
import base64
from flask import Flask, jsonify
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
app= Flask(__name__)

UPLOAD_FOLDER='static/upload/'
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER



@app.route("/")
def home():
    
    return  render_template("home.html")
    
@app.route("/numpy")
def numpy():
   
    return  render_template("numpy.html")  
    
@app.route("/pandas")
def pandas():
   
    return  render_template("pandas.html")  

@app.route("/matplotlib")
def matplotlib():
   
    return  render_template("matplotlib.html")     
@app.route("/seaborn")
def seaborn():
   
    return  render_template("seaborn.html")  
    
@app.route("/scipy")
def scipy():
   
    return  render_template("scipy.html")   
@app.route("/sklearn")
def sklearn():
   
    return  render_template("sklearn.html") 
@app.route("/knn")
def knn():
   
    return  render_template("knn.html") 

@app.route("/svm")
def svm():
   
    return  render_template("svm.html") 

@app.route("/logisticregression")
def logisticregression():
   
    return  render_template("logisticregression.html") 

@app.route("/randomforest")
def randomforest():
   
    return  render_template("randomforest.html")

@app.route("/decisiontree")
def decisiontree():
   
    return  render_template("decisiontree.html")
@app.route("/output")
def output():
   
    return  render_template("output.html")
@app.route('/kclus')
def kclus():

   return render_template('kclus.html')

@app.route("/allcompare")
def allcompare():
   
    return  render_template("allcompare.html")

@app.route("/dbscan")
def dbscan():
   
    return  render_template("dbscan.html")

@app.route('/create-plot', methods=['POST'])
def createplot():
   
    pldata1=request.form['pldata1']
    pldata2=request.form['pldata2']
    ls=request.form['ls']
    lw=request.form['lw']
    color=request.form['color']
    marker=request.form['marker']
    pldata1_array = [int(num) for num in pldata1.split(',')]
    pldata2_array = [int(num) for num in pldata2.split(',')]
    print(pldata1_array)
    ax= plt.plot(pldata1_array,pldata2_array,c=color,ls=ls,marker=marker,lw=lw)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgpl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgpl)
  
@app.route('/create-scatterplot', methods=['POST'])
def createsplot():
   
    xdata=request.form['xdata']
    ydata=request.form['ydata']
    colorr=request.form['colorr']
    cmap=request.form['cmap']
    alpha=float(request.form['alpha'])
    xdataarray = [int(num) for num in xdata.split(',')]
    ydataarray = [int(num) for num in ydata.split(',')]
    colorr = [int(num) for num in colorr.split(',')]
    colrr=np.array(colorr)
    size=request.form['size']
    size = [int(num) for num in size.split(',')]
    ax= plt.scatter(xdataarray,ydataarray,c=colrr,s=size,cmap=cmap,alpha=alpha)
    plt.colorbar()
    plt.xticks(rotation=90)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgspl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgspl)
   
@app.route('/create-pie', methods=['POST'])
def createpie():
   
    xdata=request.form['datapi']
    colorr=request.form['colorpi']
    labels=request.form['labels']
   
    datapi = [int(num) for num in xdata.split(',')]
    colorr = [num for num in colorr.split(',')]
    labels = [num for num in labels.split(',')]
    explode=request.form['explode']
    explode = [float(num) for num in explode.split(',')]
    ax= plt.pie(datapi,colors=colorr,labels=labels,explode=explode,autopct='%1.1f%%',counterclock='false',shadow='true')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    imgspl = image_memory.decode('utf-8')
           
    return render_template('matplotoutput.html',image=imgspl)

@app.route('/create3d',methods=['POST'])
def create3d():
    x=int(request.form['x'])
    y=int(request.form['y'])
    z=int(request.form['z'])
    x_data2=np.arange(x,y,z)
    y_data2=np.arange(x,y,z)
    ax=plt.axes(projection="3d")
    x,y=np.meshgrid(x_data2,y_data2)
    z=np.sin(x)*np.cos(y)
    ax.plot_surface(x,y,z,cmap="plasma")
    buffer67 = io.BytesIO()
    plt.savefig(buffer67, format="png")
    buffer67.seek(0)
    image_memory = base64.b64encode(buffer67.getvalue())
    im13 = image_memory.decode('utf-8')
 
    return render_template('matplotoutput.html',image=im13)

@app.route('/normaldistribution',methods=['POST'])    
def normal_distribution():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    

    n=int(request.form['n'])
    y=int(request.form['y'])
    if y==1:
        sns.distplot(random.normal(size=n))
    else:
        sns.distplot(random.normal(size=n),hist=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img4=img) 

@app.route('/binomialdistribution',methods=['POST'])    
def binomial_distribution():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    p=int(request.form['p'])
    q=int(request.form['q'])
    r=float(request.form['r'])
    sns.distplot(random.binomial(n=q, p=r, size=p), hist=True, kde=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img5=img)

@app.route('/poissondistribution',methods=['POST'])    
def poisson_distribution():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    a=int(request.form['a'])
    b=int(request.form['b'])
    
    sns.distplot(random.poisson(lam=b, size=a), kde=False)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img6=img) 
@app.route('/logisticdistribution',methods=['POST'])    
def logistic_distribution():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n=int(request.form['n'])
   
    
    sns.distplot(random.logistic(size=n), hist=False)


    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img7=img)

@app.route('/comparsion',methods=['POST'])    
def Comparison():
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    z=int(request.form['z'])
    q=int(request.form['q'])
    c=int(request.form['c'])
    r=float(request.form['r'])
    
    sns.distplot(random.normal(size=z), hist=False, label='normal')
    sns.distplot(random.binomial(n=c, p=r, size=z), hist=False, label='binomial')
    sns.distplot(random.poisson(lam=q, size=z), hist=False, label='poisson')
    sns.distplot(random.logistic(size=z), hist=False, label='logistic')


    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img=image_memory.decode('utf-8')    


    
    return render_template("seaborn.html",img8=img)

@app.route('/createseries', methods=['POST'])
def create_dataseries():
    my_dict = {}
    df1 = request.form['df1']
    df1_list = [ n for n in df1.split(',')]
    data = request.form['data']
    data_list = [ n for n in data.split(',')]
    for i in range( 0,len(df1_list),1):
                   my_dict[df1_list[i]]= data_list[i]              
    

    datas=pd.Series(my_dict,index=df1_list)
    return render_template('pandas.html',datasi=datas.index,datas=datas)
        
@app.route('/csv', methods=['POST'])
def csv():
    # df2 = str(request.form['name1'])
    upfile=request.files['uploadcsv']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)          
    return render_template('pandaroutput.html', tables=[read.to_html()], titles=[''])

@app.route("/agglomerative")
def agglo():
   
    return  render_template("agglo.html")

@app.route('/agglomerative',methods=['POST'])
def agglomerative():

    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)

    
    print(path)

    df= pd.read_csv(path)
    print(df.describe())

    print(df.info())

    print(df.head())

    #Encoding Variables and Feature Engineering
    #Let's start by dividing the Age into groups that vary in 10, so that we have 20-30, 30-40, 40-50, and so on. Since our youngest customer is 15, we can start at 15 and end at 70, which is the age of the oldest customer in the data. Starting at 15, and ending at 70, we would have 15-20, 20-30, 30-40, 40-50, 50-60, and 60-70 intervals.

    intervals = [15, 20, 30, 40, 50, 60, 70]
    col = df['Age']
    df['Age Groups'] = pd.cut(x=col, bins=intervals)
    df1 = pd.get_dummies(df)#At the moment, we have two categorical variables, Age and Genre, which we need to transform into numbers to be able to use in our model. There are many different ways of making that transformation - we will use the Pandas get_dummies() method that creates a new column for each interval and genre and then fill its values with 0s and 1s- this kind of operation is called one-hot encoding.
    sns.scatterplot(x=df['Annual Income (k$)'],y=df['Spending Score (1-100)'])
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img000=image_memory.decode('utf-8')
    # now trying to make clusters using pca
    df1 = df1.drop(['Age'], axis=1)
    pca = PCA(n_components=10)
    pca.fit_transform(df1)
    pca.explained_variance_ratio_.cumsum()

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df1)

    pc1_values = pcs[:,0]
    pc2_values = pcs[:,1]
    sns.scatterplot(x=pc1_values, y=pc2_values)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img001=image_memory.decode('utf-8')

    plt.figure(figsize=(10, 7))
    plt.title("Customers Dendrogram")

    # Selecting Annual Income and Spending Scores by index
    selected_data = df1.iloc[:, 1:3]
    clusters = shc.linkage(selected_data,
                method='ward',
                metric="euclidean")
    shc.dendrogram(Z=clusters)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img002=image_memory.decode('utf-8')
    #Implementing an Agglomerative Hierarchical Clustering
    clustering_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    clustering_model.fit(selected_data)
    print(clustering_model.labels_)

    data_labels = clustering_model.labels_
    sns.scatterplot(x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    data=selected_data,
                    hue=data_labels,
                    palette="rainbow").set_title('Labeled Customer Data')

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img003=image_memory.decode('utf-8')

    #This is our final clusterized data. You can see the color-coded data points in the form of five clusters.
    #
    # The data points in the bottom right (label: 0, purple data points) belong to the customers with high salaries but low spending. These are the customers that spend their money carefully.
    #
    # Similarly, the customers at the top right (label: 2, green data points), are the customers with high salaries and high spending. These are the type of customers that companies target.
    #
    # The customers in the middle (label: 1, blue data points) are the ones with average income and average spending. The highest numbers of customers belong to this category. Companies can also target these customers given the fact that they are in huge numbers.
    #
    # The customers in the bottom left (label: 4, red) are the customers that have low salaries and low spending, they might be attracted by offering promotions.
    #
    # And finally, the customers in the upper left (label: 3, orange data points) are the ones with high income and low spending, which are ideally targeted by marketing.
    clustering_model_pca = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    clustering_model_pca.fit(pcs)

    data_labels_pca = clustering_model_pca.labels_

    sns.scatterplot(x=pc1_values,
                    y=pc2_values,
                    hue=data_labels_pca,
                    palette="rainbow").set_title('Labeled Customer Data Reduced with PCA')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img004=image_memory.decode('utf-8')

    
    
    return render_template('agglo.html',img0001=img001,img0002=img002,img0003=img003,img0004=img004,img0000=img000)

 
@app.route("/dbsn",methods=['POST'])
def dbsn():
    
    file = request.files['upfile']
    df = pd.read_csv(file)


    # Perform DBSCAN clustering on the data
    eps = float(request.form['eps'])
    min_samples = int(request.form['min_samples'])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df)

    # Generate a scatter plot of the clustered data
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c=dbscan.labels_, cmap='viridis')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('DBSCAN Clustering')
    plt.colorbar()
    
    # Save the plot to a byte buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    lime=image_memory.decode('utf-8')

    # Return the plot image as a response
    

    return render_template('dbscan0.html',image=lime)

@app.route('/csvdesc', methods=['POST'])
def csvdesc():
    
    upfile=request.files['uploadcsv1']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)
    readdes=read.describe()          
    return render_template('pandaroutput.html', tables=[readdes.to_html()], titles=[''])
@app.route('/csvinfo', methods=['POST'])
def csvinfo():
   
    upfile=request.files['csvinform']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    read=pd.read_csv(path)
    readdd=pd.DataFrame(read)
    readss=readdd.isnull()          
    return render_template('pandaroutput.html', tables=[readss.to_html()], titles=[''])

@app.route('/all', methods=['POST'])
def all():
    
    

    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    s=request.form['test']
    no=request.form['no']
    s1=float(s)
    no1=int(no)

    data=pd.read_csv(path)
    #l=float(input("enter the value of the test size: \n"))
    #k=int(input("enter the value of the number of the neighbour:\n"))
    """k-nearest number for the classification of the data"""
   



    # # Confusion Matrix
    def conf(y_t, y_p):
        cm = confusion_matrix(y_t, y_p)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax)
        

    def roc_auc_curve(y_t, y_p):
        fpr, tpr, threshold = roc_curve(y_t, y_p, pos_label=1)
        auc_fig = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='AUC = %0.2f' % auc_fig, marker='.')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        




    # Defining the Independent variable and Target variable
    Y = data['Win / Loose']
    X = data.drop('Win / Loose', axis=1)

    # Bifurcating the training and testing data among the Dependent variable and Independent variable
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=s1, random_state=0)

    # Logistic Regression Training
    logr = LogisticRegression()
    logr.fit(X_train, Y_train)
    # models comparsion for accuracy
    acc_LR = logr.score(X_test, Y_test) * 100
    # confusion matrix
    Y_predict_LR = logr.predict(X_test)

    # roc curve
    y_predict_LR = logr.predict_proba(X_test)[:, 1]

    # universal confusion matrix
    cm_LR = confusion_matrix(Y_test, Y_predict_LR)
    # f1 score
    f_s_LR = f1_score(Y_test, Y_predict_LR, average='weighted')

    # Random Forest Training
    RF = RandomForestClassifier()
    RF.fit(X_train, Y_train)
    # models comparsion for accuracy
    acc_RF = RF.score(X_test, Y_test) * 100
    # confusion matrix
    Y_predict_RF = RF.predict(X_test)

    # roc curve
    y_predict_RF = RF.predict_proba(X_test)[:, 1]

    # universal comparsion
    cm_Rf = confusion_matrix(Y_test, Y_predict_RF)
    # f1-score
    f_s_RF = f1_score(Y_test, Y_predict_RF, average='weighted')

    # KNN Training
    KNN = Kn(n_neighbors=no1)
    KNN.fit(X_train, Y_train)
    # models comparsion for accuracy
    acc_KNN = KNN.score(X_test, Y_test) * 100
    # confusion matrix
    Y_predict_KNN = KNN.predict(X_test)

    # roc curve
    y_predict_KNN = KNN.predict_proba(X_test)[:, 1]

    # universal comparsion
    cm_KNN = confusion_matrix(Y_test, Y_predict_KNN)
    # fi_score
    f_s_KNN = f1_score(Y_test, Y_predict_KNN, average='weighted')

    # SVM Training
    svc = SVC(probability=True)
    svc.fit(X_train, Y_train)
    # models comparsion for accuracy
    acc_SVM = svc.score(X_test, Y_test) * 100
    # confusion matrix
    Y_predict_SVM = svc.predict(X_test)

    # roc curve
    y_predict_SVM = svc.predict_proba(X_test)[:, 1]

    # universal comparsion
    cm_SVM = confusion_matrix(Y_test, Y_predict_SVM)
    # fi score
    f_s_SVM = f1_score(Y_test, Y_predict_SVM, average='weighted')

    # Decision Tree Training
    DT = tree.DecisionTreeClassifier(max_depth=2)
    DT.fit(X_train, Y_train)
    # models comparsion for accuracy
    acc_DT = DT.score(X_test, Y_test) * 100
    # confusion matrix
    Y_predict_DT = DT.predict(X_test)

    # roc curve
    y_predict_DT = DT.predict_proba(X_test)[:, 1]

    # universal confusion
    cm_DT = confusion_matrix(Y_test, Y_predict_DT)
    # fi-score
    f_s_DT = f1_score(Y_test, Y_predict_DT, average='weighted')
    # labels for different algoritms comparsion
    def add_labels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i].round(2), y[i].round(2))
    
    # roc comparsion
    fpr1, tpr1, thresholds1 = roc_curve(Y_test, y_predict_DT, pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(Y_test, y_predict_RF, pos_label=1)
    fpr3, tpr3, thresholds3 = roc_curve(Y_test, y_predict_LR, pos_label=1)
    fpr4, tpr4, thresholds4 = roc_curve(Y_test, y_predict_SVM, pos_label=1)
    fpr5, tpr5, thresholds5 = roc_curve(Y_test, y_predict_KNN, pos_label=1)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    roc_auc4 = auc(fpr4, tpr4)
    roc_auc5 =auc(fpr5,tpr5)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr1, tpr1, label='AUC Decision Tree = %0.2f' % roc_auc1)
    plt.plot(fpr2, tpr2, label='AUC Random Forest = %0.2f' % roc_auc2)
    plt.plot(fpr3, tpr3, label='AUC Logistic Regression = %0.2f' % roc_auc3)
    plt.plot(fpr4, tpr4, label='AUC SVM = %0.2f' % roc_auc4)
    plt.plot(fpr5, tpr5, label='AUC KNN = %0.2f' % roc_auc5)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format="png")
    buffer2.seek(0)
    image_memory = base64.b64encode(buffer2.getvalue())
    im9 = image_memory.decode('utf-8')
   









    # accuracy comparsion of different alogoritms
    accu = {'Random Forest': acc_RF, 'KNN': acc_KNN, 'Decision Tree': acc_DT, 'SVM': acc_SVM,'LOGISTIC REGRESSION': acc_LR}
    Algorithm = list(accu.keys())
    Accuracy = list(accu.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(Algorithm, Accuracy, width=0.4)
    add_labels(Algorithm, Accuracy)
    plt.title("Accuracy Comparisons of Different Algorithms ")
    plt.show()
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format="png")
    buffer1.seek(0)
    image_memory = base64.b64encode(buffer1.getvalue())
    im7 = image_memory.decode('utf-8')

    # f1_score comparsion
    score = {'Random Forest': f_s_RF, 'KNN': f_s_KNN, 'Decision Tree': f_s_DT, 'SVM': f_s_SVM,'LOGISTIC REGRESSION': f_s_LR}
    Algorithm = list(score.keys())
    F1_Score = list(score.values())
    figu = plt.figure(figsize=(10, 5))
    plt.bar(Algorithm, F1_Score, width=0.4)
    add_labels(Algorithm, F1_Score)
    plt.title("F1 Score Comparisons of Different Algorithms ")
    plt.show()
    buffer4 = io.BytesIO()
    plt.savefig(buffer4, format="png")
    buffer4.seek(0)
    image_memory = base64.b64encode(buffer4.getvalue())
    im5 = image_memory.decode('utf-8')
           
    
    # confusion matrix comparsion
    figur, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(30, 5))
    figur.subplots_adjust(wspace=0.5)
    plt.title("Confusion Matrices of different Algorithms ")
    sns.heatmap(cm_LR, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax1)
    sns.heatmap(cm_Rf, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax2)
    sns.heatmap(cm_DT, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax3)
    sns.heatmap(cm_KNN, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax4)
    sns.heatmap(cm_SVM, annot=True, linewidths=0.5, linecolor='red', fmt='.0f', ax=ax5)
    plt.show()
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format="png")
    buffer3.seek(0)
    image_memory = base64.b64encode(buffer3.getvalue())
    im4 = image_memory.decode('utf-8')
           
    
    
           
    return render_template('overallout.html',img7=im7,im4=im4,im5=im5,im9=im9)


@app.route('/kmean', methods=['POST'])
def kmean():


    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)

    
    print(path)

    # OPEN DATA SET
    data= pd.read_csv(path)
    print(data)
    print("-----------------------------------------------------")
    print(data.describe())
    # mean sum other values
    print("-----------------------------------------------------")
    # print(dataset.head()) first 5 rows
    # print(data)
    print(data.info())
    print("-----------------------------------------------------")
    # CLEANNING DATASET
    # data.dropna(inplace=True)
    # print(data.info())
    data.drop_duplicates(inplace=True)
    print("-----------------------------------------------------")
    data=data.drop(['date','result','dl_applied','umpire3'],axis=1)
    print(data)
    print("-----------------------------------------------------")
    # print(data.win_by_wickets.isnull().sum())
    data['win_by_wickets']=data['win_by_wickets'].fillna(method="ffill")
    print(data['win_by_wickets'])
    print("-----------------------------------------------------")
    data['win_by_runs']=data['win_by_runs'].fillna(method='bfill')

    print("-----------------------------------------------------")
    data["winner"]=data["winner"].fillna('normal')
    data["umpire1"]=data["umpire1"].fillna(method='ffill')
    data["umpire2"]=data["umpire2"].fillna(method='bfill')

    print(data.info())
    print("-----------------------------------------------------")

    #performing string to numeric conversion
    l = LabelEncoder()
    data['toss_decision']=l.fit_transform(data['toss_decision'])
    # data['result']=l.fit_transform(data['result'])
    data['Season']=l.fit_transform(data['Season'])
    data['city']=l.fit_transform(data['city'])
    data['team1']=l.fit_transform(data['team1'])
    data['team2']=l.fit_transform(data['team2'])
    data['toss_winner']=l.fit_transform(data['toss_winner'])
    data['winner']=l.fit_transform(data['winner'])
    data['player_of_match']=l.fit_transform(data['player_of_match'])
    data['venue']=l.fit_transform(data['venue'])
    data['umpire1']=l.fit_transform(data['umpire1'])
    data['umpire2']=l.fit_transform(data['umpire2'])

    #Defining the Independent variable and Target variable
    # Y = data['toss_decision']
    # X = data.drop('toss_decision', axis=1)

    
    wcss = []

    for i in range(1,11):
        km = KMeans(n_clusters=i)
        km.fit_predict(data)
        wcss.append(km.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(1,11),wcss)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    lime=image_memory.decode('utf-8')

    X=data.drop('toss_decision',axis=1)
    km=KMeans(n_clusters=3)
    ymeans=km.fit_predict(X)
    
    x=ymeans
    print(x)
    


    
    X['cluster_label'] = km.labels_
    
    fig, ax2 = plt.subplots()
    ax2.scatter(X["winner"], X["toss_winner"], c=X['cluster_label'], cmap='rainbow')
    ax2.set_xlabel('WINNER ')
    ax2.set_ylabel('toss_winner')
    buffer = io.BytesIO()
    ax2.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    scatter=image_memory.decode('utf-8')

    
    
    cp=sns.countplot(x=ymeans)
    cp.set_title('number of clusters') 
    buffer = io.BytesIO()
    cp.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    count=image_memory.decode('utf-8')
      
    return render_template('kclus.html',lime=lime,scatter=scatter,count=count,x=x)
@app.route('/add_numbers', methods=['POST'])
def add_numbers():
	num1 = int(request.form['num1'])
	num2 = int(request.form['num2'])
	result = num1 + num2
	return render_template('numpy.html', result=result)

@app.route('/create-array', methods=['POST'])
def create_array():
    set1 = request.form['set1']
    set2 = request.form['set2']

    set1_array = [int(num) for num in set1.split(',')]
    set2_array = [int(num) for num in set2.split(',')]

    x = np.array([set1_array, set2_array])

    return render_template('numpy.html', x=x)

@app.route('/create-index', methods=['POST'])
def create_index():
    if request.method == 'POST':
        set3 = request.form['set3']
        num_list = [int(n) for n in set3.split(',')]
        y= np.array([num_list])
        num3 = request.form['num3']
        if num3:
            try:
                value = num_list[int(num3)]
                return render_template('numpy.html', value=value, num3=num3, y=y)
            except:
                return render_template('numpy.html', error="Invalid index")
    return render_template('numpy.html')

@app.route('/sort', methods=['POST'])
def create_sort():
    import numpy as np
    set4 = request.form['set4']
    set4_array = [int(num) for num in set4.split(',')]
    sortarray=np.array(set4_array)
    sortvalue=np.sort(sortarray)
    return render_template('numpy.html',sortvalue=sortvalue)

@app.route('/eu-distance',methods=['POST'])
def eu_distance():
    from scipy.spatial.distance import euclidean

    p1 = request.form['p1']
    p1=tuple(int(a) for a in p1.split(","))
    p2 = request.form['p2']
    p2=tuple(int(a) for a in p2.split(","))

    res = euclidean(p1, p2)

    
    return render_template("scipy.html", output=res) 
@app.route('/manhattan-distance',methods=['POST'])    
def manhattan_distance():
    from scipy.spatial.distance import cityblock

    p11 = request.form['p11']
    p11=tuple(int(a) for a in p11.split(","))
    p22 = request.form['p22']
    p22=tuple(int(a) for a in p22.split(","))

    res1 = cityblock(p11, p22)

    
    return render_template("scipy.html", output1=res1) 
@app.route('/csrmatrix',methods=['POST'])    
def csr_matrix():
     import numpy as np
     from scipy.sparse import csr_matrix

     x = request.form['x']   
     x = list(int(a) for a in x.split(","))

     arr = np.array(x)
    
     z=csr_matrix(arr)
    

     return render_template("scipy.html", output=z) 

@app.route('/statsdesc',methods=['POST'])    
def stats_desc():
    import numpy as np
    from scipy.stats import describe

    n=int(request.form['n'])

    v = np.random.normal(size=n)
    res = describe(v)


    
    return render_template("scipy.html", output2=res)   
@app.route('/normalitytest',methods=['POST'])    
def normalitytest():
    import numpy as np
    from scipy.stats import skew, kurtosis, normaltest

    n=int(request.form['n'])

    v = np.random.normal(size=n)
    x = skew(v)
    y=kurtosis(v)
    z=normaltest(v)


    
    return render_template("scipy.html",o=x,q=y,r=z)
@app.route('/scikitpca',methods=['POST'])
def scikitpca():
    import csv
    import codecs

    upfile=request.files['scikitf']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)   
    data=pd.read_csv(path)
    pca=PCA()
    pca.fit(data)
    pca_data = pca.transform(data)
    plt.figure(figsize=(10,6))
    plt.scatter(pca_data[:,0], pca_data[:,1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Plot')
    plt.grid()
    buffer7 = io.BytesIO()
    plt.savefig(buffer7, format="png")
    buffer7.seek(0)
    image_memory = base64.b64encode(buffer7.getvalue())
    im4 = image_memory.decode('utf-8')
    
    return render_template('outsci.html',image=im4)


@app.route('/bayesianr',methods=['POST'])
def bayesianr():
    

    upfile=request.files['bayesian']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    
    ts=request.form['test']
    ko=float(ts)
    
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # Split the dataset into training and test sets
    X_train,X_test, Y_train,Y_test=train_test_split(X,y,test_size=ko)
    
    # Create a Bayesian Ridge regression model and fit it to the training data
    reg = BayesianRidge()
    reg.fit(X_train, Y_train)
    # Predict on the test set and calculate the mean squared error
    Y_pred = reg.predict(X_test)
    mse = (Y_test, Y_pred)
    return render_template('sklearn.html',mse=mse)

@app.route("/knnresult",methods=['POST'])     
def knnresult():
    n=request.form['n']
    s=request.form['s']
    n1=int(n)
    s1=float(s)
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    print(path)
    
    
    # Reading CSV file
    df=pd.read_csv(path)

    Y=df['Win / Loose']
    X=df.drop('Win / Loose',axis=1)


    X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=s1,random_state=0)

    KNN = Kn(n_neighbors=n1)
    KNN.fit(X_train,Y_train)
    Y_predict = KNN.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')
    


    y_predict = KNN.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_knn = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_KNN = %0.2f' % auc_knn)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc)

@app.route("/rfresult",methods=['POST'])     
def rfresult():
    s=request.form['s']
    s1=float(s)
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    print(path)

    df=pd.read_csv(path)

    Y=df['Win / Loose']
    X=df.drop('Win / Loose',axis=1)


    X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=s1,random_state=0)


    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    Y_predict = rf.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = rf.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_rf = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Random_Forest = %0.2f' % auc_rf)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')


    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc)

@app.route("/svmresult",methods=['POST'])     
def svmresult():
    s=request.form['s']
    s1=float(s)
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    print(path)

    df=pd.read_csv(path)

    Y=df['Win / Loose']
    X=df.drop('Win / Loose',axis=1)


    X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=s1,random_state=0)
    svc = SVC(probability=True)
    svc.fit(X_train,Y_train)
    Y_predict = svc.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = svc.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_svm = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_SVM = %0.2f' % auc_svm)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc)

@app.route("/lrresult",methods=['POST'])     
def lrresult():
    s=request.form['s']
    s1=float(s)
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    print(path)

    df=pd.read_csv(path)

    Y=df['Win / Loose']
    X=df.drop('Win / Loose',axis=1)


    X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=s1,random_state=0)    
    logr = LogisticRegression()
    logr.fit(X_train, Y_train)
    Y_predict = logr.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = logr.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_lr = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Logistic_regression = %0.2f' % auc_lr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')

    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc)

@app.route("/dtresult",methods=['POST'])     
def dtresult():
    n=request.form['n']
    s=request.form['s']
    n1=int(n)
    s1=float(s)
    upfile=request.files['upfile']
    filename=secure_filename(upfile.filename)
    upfile.save(os.path.join(UPLOAD_FOLDER,filename))
    path=os.path.join(UPLOAD_FOLDER,filename)
    print(path)

    df=pd.read_csv(path)

    Y=df['Win / Loose']
    X=df.drop('Win / Loose',axis=1)


    X_train,X_test, Y_train,Y_test=train_test_split(X,Y,test_size=s1,random_state=0)
    model = tree.DecisionTreeClassifier(max_depth=n1)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)
    fscore = f1_score(Y_test, Y_predict, average='weighted')

    cm = metrics.confusion_matrix(Y_test, Y_predict)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax, alpha = 1)
    buffer = io.BytesIO()
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_cm = image_memory.decode('utf-8')

    y_predict = model.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, y_predict, pos_label=1)
    auc_dt = auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label = 'AUC_Decision_Tree = %0.2f' % auc_dt)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    ax.figure.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_roc = image_memory.decode('utf-8')
    return render_template('output.html', acc=acc, fscore=fscore, img=img_cm, img1=img_roc)

if __name__=='__main__':

    app.run(debug=True)
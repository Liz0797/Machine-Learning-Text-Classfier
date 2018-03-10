# coding:utf-8
import os
import codecs
import nltk
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

'''
	文件的初始目录和处理后的目录以及符号表
'''
inital_path = "/Users/lizhen/Documents/corpus_initial/"
result_path = "/Users/lizhen/Documents/corpus_analyse/"
predict_path = "/Users/lizhen/Documents/corpus_predict/"
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
# 词形还原
porter = nltk.stem.WordNetLemmatizer()


# 读文件夹的下的所有文件 并返回文件列表
def readDir(dirPath):
	allFile = os.listdir(dirPath)
	fileList = []
	for file in allFile:
		#print(dirPath + file)
		fileList.append(file)
	return fileList

# 读文件
def readFile(dirPath,filename):
	fopen = open(dirPath + filename, "r")
	line = fopen.readline()
	rs = ""
	while line:
		rs = rs + line
		line = fopen.readline()
	fopen.close()
	return rs

# 写入文件
def writeFile(dirPath, filename, texts):
	#print(dirPath+filename, 'w')
	#print(texts)
	fwrite = open(dirPath+filename,'w')
	fwrite.write(texts)
	fwrite.close()

# 利用nltk将文本分词 包括 去符号，常用词，最后返回一个字符串
def nltkAnalyse(texts):
	# 分词
	texts_tokenize = nltk.word_tokenize(texts)
	# 去符号
	texts_punctuation = [word for word in texts_tokenize if(word not in english_punctuations)]
	# 词型还原
	texts_stemm = [porter.lemmatize(word) for word in texts_punctuation]
	#去停用词
	#texts_stopword = [ word for word in texts_stemm if(word not in stopwords.words('english')) ]
	texts_stopword = []
	for word in texts_stemm:
		if(word not in stopwords.words('english')):
			texts_stopword.append(word)
	#print(english_punctuations)
	#print(texts_tokenize)
	#print(texts_stopword)
	#print(texts_punctuation)
	#print(texts_stemm)
	rs = ' '.join(texts_stemm)
	#print(texts_stemm)
	return rs

def sample_train_data(inital_path,result_path):
	fileList = readDir(inital_path)
	for file in fileList:
		if(file != '.DS_Store'):
			text = readFile(inital_path,file)
			writeFile(result_path,file,text)


# tf-idf进行初始文本的划分 获取词表和向量表
def tfidf(corpus):
	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
	word = vectorizer.get_feature_names()
	weight = tfidf.toarray()
	#print(word)
	#print(weight)
	return word, weight

''' kMeans对初始文本进行聚类 
	KMeans(n_clusters=8, init='k-means++', n_init=10/随机种子数, max_iter=300/迭代次数, tol=0.0001, 
		precompute_distances='auto', verbose=0, random_state=None, copy_x=True, 
		n_jobs=1, algorithm='auto')
'''
def KMeans_model(word):
	kms = KMeans(n_clusters = 2,random_state = 0).fit(word)
	# print(len(kms.labels_))
	# print(kms.labels_)
	return kms

# 文本向量化
def text2Vector(words, text):
	vector = []
	tokenize = nltk.word_tokenize(nltkAnalyse(text))
	for word in words:
		if(word in tokenize):
			vector.append('1')
		else:
			vector.append('0')
	return vector
	
def svm_model(weight,labels):
	clf = LinearSVC(random_state = 0)
	clf.fit(weight,labels)
	print(clf.coef_)
	print(clf.intercept_)
	return clf;


def model_predict(model, filepath, word):
#	j=0
	fileList = readDir(filepath)
	for file in fileList:
		if(file != ".DS_Store"):
			#print(file,"")
			text = readFile(filepath, file)
			vector = text2Vector(word, text)
			#print(vector)
			vector = np.array(vector, dtype=np.float64)
			print(file,'类型为',model.predict(vector.reshape(1,-1)))
			
			'''if(model.predict(vector.reshape(1,-1))>0):
				j=j+1
	print(j)'''


#肘方法
def kmeans_cluster(weight):
	distortion = []
	for i in range(1, 11):
		km = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter = 300, random_state = 0)
		km.fit(weight)
		distortion.append(km.inertia_)
	plt.plot(range(1, 11), distortion, marker = 'o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.show()

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """smoSimple

    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        maxIter 退出前最大的循环次数
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    dataMatrix = mat(dataMatIn)
    #print('--------',shape(dataMatrix))
    # 矩阵转置 和 .T 一样的功能
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    #print('-----',m,shape(labelMat))
    # 初始化 b和alphas(alpha有点类似权重值。)
    b = 0
    alphas = mat(zeros((m, 1)))
	#	print('----------',m)
    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # w = calcWs(alphas, dataMatIn, classLabels)
        # print("w:", w)

        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # print 'alphas=', alphas
            # print 'labelMat=', labelMat
            # print 'multiply(alphas, labelMat)=', multiply(alphas, labelMat)
            # 我们预测的类别 y = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*lable[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(labelMat[i])

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):

                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没发优化了
                if L == H:
                    #print("L==H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    #print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    #print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                #print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环。
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
       # print("iteration number: %d" % iter)
    return b, alphas

def self_svm(weight,word,label,path):
	i=0
	b,alpha = smoSimple(weight, label,0.6, 0.001, 40)
	m,n=shape(weight)
	w = zeros((n, 1))
	w = w.T
	for a in alpha:
		w = w+alpha[i]*label[i]*weight[i]
		#print(x)
		#print(shape(x))
		#w = w + x
		i = i+1
	#print(w)
	fileList = readDir(path)
	print('----------------------------------------------------------------')
	#j=0
	for file in fileList:
		if(file != ".DS_Store"):
			#print(file,"")
			text = readFile(path, file)
			vector = text2Vector(word, text)
			vector = np.array(vector, dtype=np.float64)
			vector = asmatrix(vector)
			y=vector*w.T+b
			if(y>1):
				print(file,'类型为','[1]')
				#j = j+1
			elif(y<-1):
				print(file,'类型为','[0]')
			else:
				print(file,'类型不确定')
	##print(j)


if __name__ == '__main__':
	# 读取文件列表
	fileList = readDir(result_path)
	corpus = [] #聚类语料库
	for file in fileList:
		if(file != ".DS_Store"):
			rs = readFile(result_path, file)
			rs = nltkAnalyse(rs)
			corpus.append(rs)
	word,weight = tfidf(corpus)
	kmeans_cluster(weight)
	texts = []
	print(weight)
	print(len(weight))
	cluster = KMeans_model(weight)
	print(cluster.labels_)
	clf = svm_model(weight, cluster.labels_)
	fileList_predict = readDir(predict_path);
	model_predict(clf,predict_path,word)

	#print(cluster.cluster_centers_)
	#print(clf.coef_)
	svm_label = []
	for s_label in cluster.labels_:
		if s_label == 0:
			svm_label.append(-1)
		else:
			svm_label.append(1)
	self_svm(weight,word,svm_label,predict_path)
	#print(svm_label)
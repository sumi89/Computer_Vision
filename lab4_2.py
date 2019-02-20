#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 02:16:09 2018

@author: sumi
"""

pca = PCA(n_components=150, random_state = 99)
pca.fit(X_train_org)
X_train_proc= pca.transform(X_train_org)
X_test_proc = pca.transform(X_test_org)
print("Preprocessing: PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)









pca = PCA(n_components=100, random_state = 99)
pca.fit(X_train_org)
X_train_proc= pca.transform(X_train_org)
X_test_proc = pca.transform(X_test_org)
print("Preprocessing: PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)







sc = preprocessing.StandardScaler().fit(X_train_org)
X_train_proc = sc.transform(X_train_org)
X_test_proc = sc.transform(X_test_org)
print("Preprocessing: Standarization")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)










norm = preprocessing.Normalizer().fit(X_train_org)
X_train_proc = norm.transform(X_train_org)
X_test_proc =norm.transform(X_test_org)
print("Preprocessing: Normalization")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)







sc = preprocessing.StandardScaler().fit(X_train_org)
X_train_sc = sc.transform(X_train_org)
X_test_sc = sc.transform(X_test_org)
pca = PCA(n_components = 150,random_state = 99).fit(X_train_sc)
X_train_proc = pca.transform(X_train_sc)
X_test_proc = pca.transform(X_test_sc)
print("Preprocessing: Standardization and PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)









sc = preprocessing.StandardScaler().fit(X_train_org)
X_train_sc = sc.transform(X_train_org)
X_test_sc = sc.transform(X_test_org)
pca = PCA(n_components = 100,random_state = 99).fit(X_train_sc)
X_train_proc = pca.transform(X_train_sc)
X_test_proc = pca.transform(X_test_sc)
print("Preprocessing: Standardization and PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)








norm = preprocessing.Normalizer().fit(X_train_org)
X_train_norm = norm.transform(X_train_org)
X_test_norm =norm.transform(X_test_org)
pca = PCA(n_components = 150).fit(X_train_norm)
X_train_proc = pca.transform(X_train_norm)
X_test_proc = pca.transform(X_test_norm)
print("Preprocessing: Normalization and PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)








norm = preprocessing.Normalizer().fit(X_train_org)
X_train_norm = norm.transform(X_train_org)
X_test_norm =norm.transform(X_test_org)
pca = PCA(n_components = 100).fit(X_train_norm)
X_train_proc = pca.transform(X_train_norm)
X_test_proc = pca.transform(X_test_norm)
print("Preprocessing: Normalization and PCA")

classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)







univ = SelectKBest(score_func=chi2, k=150) 
univ.fit(X_train_org, y_train_org)
X_train_proc = univ.transform(X_train_org)
X_test_proc = univ.transform(X_test_org)
print("Preprocessing: Univariate")


classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)







univ = SelectKBest(score_func=chi2, k=100) 
univ.fit(X_train_org, y_train_org)
X_train_proc = univ.transform(X_train_org)
X_test_proc = univ.transform(X_test_org)
print("Preprocessing: Univariate")


classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)
classifier = KNeighborsClassifier(n_neighbors=10, algorithm = 'brute')
classifier.fit(X_train_proc, y_train_org)
p = classifier.predict(X_test_proc)
accuracy = np.sum(p==y_test_org)/y_test_org.shape[0]
print("knn accuracy =", accuracy)

classifier = svm.SVC(kernel = 'linear', C = 0.0001, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = svm.SVC(kernel = 'linear', C = 0.1, gamma = 0.1)
print("Classification: Support vector machine")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = MLPClassifier(hidden_layer_sizes=(50,50), batch_size = 20, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = MLPClassifier(hidden_layer_sizes=(100,100), batch_size = 10, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 99)
print("Classification: Multilayer perceptron")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = RandomForestClassifier(n_estimators = 50, max_features = 50, max_depth = 50)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = RandomForestClassifier(n_estimators = 20, max_features = 20, max_depth = 20)
print("Classification: Random forests")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)

classifier = tree.DecisionTreeClassifier(max_depth = 10)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)
classifier = tree.DecisionTreeClassifier(max_depth = 50)
print("Classification: Decision trees")
classifier.fit(X_train_proc, y_train_org)
pred = classifier.predict(X_test_proc)
accuracy = np.sum(pred==y_test_org)/y_test_org.shape[0]
print("accuracy =", accuracy)





两种方式实现了PCA降维
1. pca文件夹中C++的实现是基于PCA的原理，按步骤实现，利于理解PCA的实现原理；
2. sklearn文件夹中的实现源于官网的例子，放在此处仅用于比较最终结果的一致性。

pca中C++实现的使用方法：
1. 下载源码编译、执行；
(或)2. 直接执行pca\build\pca.exe（注意，由于上传文件大小限制未上传opencv_world345d.dll,请自行下载到pca.exe同目录）。

数据说明：
pca\data下：
1. iris.txt是鸢尾花原始数据；
2. iris_label.txt是与iris.txt对应的标签；
3. pca_iris.txt是C++实现的PCA的运行结果；
4. sklearn_pca_iris.txt是sklearn的运行结果。
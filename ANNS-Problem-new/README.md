# ANNS Problem

本 repo 包含两道题目，近似最近邻搜索及图索引建立。`src` 目录下为代码，`res` 目录下为数据。

`src` 目录下已经准备了 Makefile 文件，使用方法：
```
make
```

## 1. ANN Search

`src/search.cpp` 为示例ANN搜索代码，代码通过随机选点查找最近邻。使用方法：

```
search 点坐标文件路径 图索引文件路径 询问文件路径 标准答案文件路径 返回最近邻数量
```
如：
```
./search ../res/siftsmall/siftsmall_base.fvecs ../res/10k.graph ../res/siftsmall/siftsmall_query.fvecs ../res/siftsmall/siftsmall_groundtruth.ivecs 100
```

## 2. Index Building

`src/build_knn.cpp` 为示例KNN索引构建代码。使用方法：

```
build_knn 点坐标文件路径 输出图索引文件路径 KNN图中节点度数
```
如：
```
./build_knn ../res/siftsmall/siftsmall_base.fvecs ../res/10k.graph 10
```
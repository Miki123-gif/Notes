# 删除命令

- **删除光标后面的所有字符：d$**

- **删除光标前的所有字符：d0**

- **删除某一行文件：**

```
:16 d
```

# 复制命令

参考：https://www.jianshu.com/p/87ccdb08bced

- **正常的复制并粘贴：**

```
按v进入可视模式，将光标移动到复制的结尾处。然后按y进行复制。按p进行粘贴。
```

- **将第9行至第15行的数据，复制到第16行**

```
:9, 15 copy 16
:9, 15 co 16
```

- **vim与剪贴板互动**

```
yy 表示复制一行
```


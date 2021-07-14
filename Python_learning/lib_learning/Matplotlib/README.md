# 参考教程

- https://tianchi.aliyun.com/course/324/3654

# 基础知识

散点图绘制：

```
map_color = {0:'r', 1:'g'}
map_marker = {0:'.', 1:'.'}
# 将字符串转换成数字
color = df.iloc[:, 2].apply(lambda x:label_dict.index(x))
# 将数字类别转换成不同颜色
diff_color = color.apply(lambda x: map_color[x])
#diff_marker = list(color.apply(lambda x: map_marker[x]))
#如果想要不同形状：https://blog.csdn.net/u014571489/article/details/102667570
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=diff_color, marker='.');
```

绘制条形图：

```
x = df.iloc[:, 2].value_counts().index
y = df.iloc[:, 2].value_counts().values
```

## jupyter中matplotlib的含义

参考：https://www.cnblogs.com/emanlee/p/12358088.html

总结：%matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。

# 在折线中标记某些点


```
plt.plot(df)
plt.plot(df, markevery=mark, marker='o')
```


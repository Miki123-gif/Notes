现在遇到的场景如下：

现在有两个分支，一个是master，另一个是dev，其中master分支里面有file1文件，dev
分支里面有file1，file2文件，现在在dev分支下commit后，然后切换到master分支下，
对dev分支进行merge，但是最后master分支下还是只有file1文件，没有file2文件


# 探索分支

测试代码如下：

```
mkdir t && cd t
echo 'version1' >readme.md
git add .
git commit -m 'update'
git checkout -b dev
echo 'version1' >readme.md
echo 'version1' >test.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
ls
```
发现master目录下并没有text文件

```
git checkout dev
echo 'version2' >>readme.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
```

只对同名字的readme.md进行修改后，发现会同步，但是不会同步新出现的文件夹text.md

```
git checkout master
echo 'version3' >>readme.md
git add .
git commit -m 'update'
git checkout dev
git merge master
```

调用完上面的代码后，然后会发现，dev分支中的readme.md会同步，然后dev分支多出的
text.md文件被删除了

```
git checkout dev
vi readme.md # 删除到只剩下version 1
echo 'version 1' >text.md
git add .
git commit -m 'update'
git checkout master
git merge --no-ff dev
```

此时发现文件也可以同步了，readme文件中只剩下了version 1， text文件也同步出现了
, 但是没有出现分支，不知道是什么原因

# 探索冲突

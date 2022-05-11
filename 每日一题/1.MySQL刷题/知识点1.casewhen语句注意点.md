# case when 语句注意点

## 1. 语法格式

```mysql
SELECT
	if_follow,
	(
		CASE B.if_follow
		WHEN 2 THEN
			- 1
		END
	) fans_growth
FROM
	tb_user_video_log B
```

![image-20220316110031080](https://s2.loli.net/2022/03/16/vEWr38wZDyUj9pJ.png)

**注意**：这里我们指定 if_follow 字段里的 2 ， 用 -1 代替，其他的值如果**不指定默认为空**，不是原来的值。
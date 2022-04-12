# 查询面试薪资最小和最大的平均值

## 1.创建表格和插入数据

```sql
-- ----------------------------
-- Table structure for cv_info
-- ----------------------------
DROP TABLE IF EXISTS `cv_info`;
CREATE TABLE `cv_info` (
  `cv_id` int NOT NULL,
  `expect_job` varchar(16) NOT NULL,
  `expect_salary` varchar(16) NOT NULL,
  PRIMARY KEY (`cv_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of cv_info
-- ----------------------------
INSERT INTO `cv_info` VALUES ('10001', 'java工程师', '8000-12000');
INSERT INTO `cv_info` VALUES ('10002', 'java工程师', '10000-12000');
INSERT INTO `cv_info` VALUES ('10003', 'java工程师', '50000-100000');
INSERT INTO `cv_info` VALUES ('10004', 'C++工程师', '15000-17000');
INSERT INTO `cv_info` VALUES ('10005', 'java工程师', '6000_9000');
INSERT INTO `cv_info` VALUES ('10006', 'java工程师', '面议');
INSERT INTO `cv_info` VALUES ('10007', 'java工程师', '7000-9000');
INSERT INTO `cv_info` VALUES ('10008', 'java工程师', '8000_12000');
```

## 2.常用函数

select * FROM cv_info

![image-20220306105944701](https://s2.loli.net/2022/03/06/7EIdfOepRtmCUr1.png)

1. CONCAT(str1,str2,...) : 把str1、str2...合并
2. CONCAT_WS(separator,str1,str2,...) : 把str1、str2...合并，中间以分隔符 'separator' 隔开
3. SUBSTRING_INDEX(str,delim,count) : 从 分隔符(delim) 在 字符串 str 出现的第 count 次数 之前 返回子字符串
	 注意：count 为正代表从左往右数， 为负代表从右往左数; 分隔符 从 1 开始 计数
4. ROUND(X,D) : 将参数四舍五入X到 D小数位

```sql
SELECT CONCAT("I",'love','China', '!') AS 合并

SELECT CONCAT_WS(" ","I","love","China!") 分隔符合并

SELECT SUBSTR("10000-12000", 1, 5) 截取字符串指定长度

SELECT SUBSTRING_INDEX("10000-12000_20000", "-", 1)

SELECT REPLACE("10000-12000_20000", "_", '-')

SELECT ROUND(12.54467, 2) # 四舍五入
```

## 3.题目

查询面试薪资最小和最大的平均值; 除去最小薪资<1000, 最大薪资>20000的, 除去面议的

```sql
WITH t1
as
(SELECT
	SUBSTRING_INDEX(REPLACE(expect_salary, '_', '-'), '-', 1) MinSalary,
	SUBSTRING_INDEX(REPLACE(expect_salary, '_', '-'), '-', -1) MaxSalary,
	c.*
FROM
	cv_info c
WHERE
	expect_salary <> "面议"
AND SUBSTRING_INDEX(expect_salary, '-', 1) > 1000
AND substring_index (expect_salary, '-', - 1) < 20000)

SELECT ROUND(avg(t1.MinSalary), 2) 平均最小薪资,
			 ROUND(AVG(t1.MaxSalary), 2) 平均最大薪资
			 
FROM t1;
```

![image-20220306110052328](https://s2.loli.net/2022/03/06/szoLIFbf1H87vGR.png)
解释：

```sql
/*解释：
	1.WITH 语句的使用
		WITH 表名1 AS (查询语句)
		select * from 表名1     ：省去建临时表，中间查询的结果命为表名1
	2.`REPLACE`(str,from_str,to_str) : 字符串str中出现的 子字符串from_str 全部替换成 to_str 
		select REPLACE('I-love-China!', '-', 'O')
	3.SELECT SUBSTRING_INDEX(str,delim,count) : 返回以分隔符delimma 分割的第count个分隔符之前的子字符串，count为正从前往后数，为负从后往前数。  
*/
select REPLACE('I love China!', ' ', '-')
```

​			 


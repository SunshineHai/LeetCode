





```sql
-- 题目：近一个月发布的视频中热度最高的top3视频


/*
	1.近一个月怎么实现？
	2.热度怎么计算？
	3.新鲜度怎么计算？
	4.最近无播放天数怎么计算？
	5.视频的 "完播率" 怎么计算？ 
		某个视频完全播放的个数的平均值 ：  视频完全播放的个数 / 视频播放个数
		IF(TIMESTAMPDIFF(SECOND, start_time, end_time) >= duration, 1, 0)：观看视频退出时间-观看视频开始时间，这个时间差如果>=视频时长，则表明该视频已经看完。
		之后再求平均值。

	1.找到最近的视频发布日期 ...... 
			最近播放日期：我们找到 所有视频的 MAX(DATE(end_time)) 的最大日期，以该日期基准，查找近一个月的视频。
	2.近一个月 = 最近发布视频的日期 - 30
	3.计算热度：热度=(a*视频完播率+b*点赞数+c*评论数+d*转发数)*新鲜度；   这是别人定义的，净扯淡
	4.新鲜度 ： 1/(最近无播放天数+1)；
	5.当前配置的参数a,b,c,d分别为100、5、3、2。
	
*/
-- 1. 数据准备
DROP TABLE IF EXISTS tb_user_video_log, tb_video_info;
CREATE TABLE tb_user_video_log (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    uid INT NOT NULL COMMENT '用户ID',
    video_id INT NOT NULL COMMENT '视频ID',
    start_time datetime COMMENT '开始观看时间',
    end_time datetime COMMENT '结束观看时间',
    if_follow TINYINT COMMENT '是否关注',
    if_like TINYINT COMMENT '是否点赞',
    if_retweet TINYINT COMMENT '是否转发',
    comment_id INT COMMENT '评论ID'
) CHARACTER SET utf8 COLLATE utf8_bin;

CREATE TABLE tb_video_info (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '自增ID',
    video_id INT UNIQUE NOT NULL COMMENT '视频ID',
    author INT NOT NULL COMMENT '创作者ID',
    tag VARCHAR(16) NOT NULL COMMENT '类别标签',
    duration INT NOT NULL COMMENT '视频时长(秒数)',
    release_time datetime NOT NULL COMMENT '发布时间'
)CHARACTER SET utf8 COLLATE utf8_bin;

INSERT INTO tb_user_video_log(uid, video_id, start_time, end_time, if_follow, if_like, if_retweet, comment_id) VALUES
   (101, 2001, '2021-09-24 10:00:00', '2021-09-24 10:00:30', 1, 1, 1, null)
  ,(101, 2001, '2021-10-01 10:00:00', '2021-10-01 10:00:31', 1, 1, 0, null)
  ,(102, 2001, '2021-10-01 10:00:00', '2021-10-01 10:00:35', 0, 0, 1, null)
  ,(103, 2001, '2021-10-03 11:00:50', '2021-10-03 11:01:35', 1, 1, 0, 1732526)
  ,(106, 2002, '2021-10-02 10:59:05', '2021-10-02 11:00:04', 2, 0, 1, null)
  ,(107, 2002, '2021-10-02 10:59:05', '2021-10-02 11:00:06', 1, 0, 0, null)
  ,(108, 2002, '2021-10-02 10:59:05', '2021-10-02 11:00:05', 1, 1, 1, null)
  ,(109, 2002, '2021-10-03 10:59:05', '2021-10-03 11:00:01', 0, 1, 0, null)
  ,(105, 2002, '2021-09-25 11:00:00', '2021-09-25 11:00:30', 1, 0, 1, null)
  ,(101, 2003, '2021-09-26 11:00:00', '2021-09-26 11:00:30', 1, 0, 0, null)
  ,(101, 2003, '2021-09-30 11:00:00', '2021-09-30 11:00:30', 1, 1, 0, null);

INSERT INTO tb_video_info(video_id, author, tag, duration, release_time) VALUES
   (2001, 901, '旅游', 30, '2021-09-05 7:00:00')
  ,(2002, 901, '旅游', 60, '2021-09-05 7:00:00')
  ,(2003, 902, '影视', 90, '2021-09-05 7:00:00')
  ,(2004, 902, '影视', 90, '2021-09-05 8:00:00');

-- 2.思路

SELECT * FROM tb_user_video_log;
SELECT * FROM tb_video_info;					# 字段：duration ：视频的总时长 



SELECT
	video_id,
	ROUND(
		(
			100 * comp_play_rate + 5 * like_cnt + 3 * comment_cnt + 2 * retweet_cnt
		) / (
			TIMESTAMPDIFF(                  # 最近无播放天数
				DAY,
				recently_end_date,
				cur_date
			) + 1
		),
		0
	) AS hot_index
FROM
	(
		SELECT
			video_id,
			AVG(
				IF (
					TIMESTAMPDIFF(SECOND, start_time, end_time) >= duration,
					1,
					0
				)
			) AS comp_play_rate,															-- 完播率
			SUM(if_like) AS like_cnt,													-- 点赞总数
			COUNT(comment_id) AS comment_cnt,									-- 每个视频的评论人数
			SUM(if_retweet) AS retweet_cnt,										-- 转发总数
			MAX(DATE(end_time)) AS recently_end_date,					-- 最近被播放日期
			MAX(DATE(release_time)) AS release_date,				  -- 发布日期
			MAX(cur_date) AS cur_date 												-- 非分组列，加MAX避免语法错误
		FROM
			tb_user_video_log
		JOIN tb_video_info USING (video_id)
		LEFT JOIN (
			SELECT
				MAX(DATE(end_time)) AS cur_date									-- 最近播放日期
			FROM
				tb_user_video_log
		) AS t_max_date ON 1
		GROUP BY
			video_id
		HAVING
			TIMESTAMPDIFF(DAY, release_date, cur_date) < 30		-- 进一个月所播放的视频的个数
	) AS t_video_info
ORDER BY
	hot_index DESC
LIMIT 3;


-- 注：
-- 1.datetime_expr2 与 datetime_expr1之间相差的年/月/日 数， unit ： 为 year、month、day、minute、second
-- TIMESTAMPDIFF(unit,datetime_expr1,datetime_expr2) 
SELECT TIMESTAMPDIFF(day,'2002-05-01','2002-05-05');


-- 2.等值连接的实现
-- 法1：
SELECT
	*
FROM
	tb_user_video_log a
JOIN tb_video_info b ON (a.video_id = b.video_id);

-- 法2：
SELECT
	*
FROM
	tb_user_video_log a,
	tb_video_info 		b
WHERE a.video_id = b.video_id;  					# 这种连接 2个表中的 video_id 都出现

-- 法3： 
SELECT
	*
FROM
	tb_user_video_log
JOIN tb_video_info b USING (video_id);		# 使用 using 关键字， video_id 只出现一次
-- 3.左连接一个字段

SELECT
	video_id,
	start_time,
	end_time,
	tag,
	duration,
	release_time,
	cur_date
FROM
	tb_user_video_log
JOIN tb_video_info b USING (video_id)
LEFT JOIN (																	# table1 LEFT JOIN table2 ON(条件) 左连接
	SELECT
		MAX(DATE(end_time)) AS cur_date
	FROM
		tb_user_video_log
) AS t_max_date ON 1;												# ON 1 条件为1，表示连接时没有条件，table2连接到table1中的最后一列,列值全部为table中查询出的值
 
-- 4.`IF`(expr1,expr2,expr3)  : 如果表达式 expr1 为真，则值为 expr2， 否则 值为 expr3。
SELECT IF(1<2, 1, 0);
```


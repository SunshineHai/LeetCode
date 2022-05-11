/*
Navicat MySQL Data Transfer

Source Server         : yhk
Source Server Version : 80028
Source Host           : localhost:3306
Source Database       : niuke

Target Server Type    : MYSQL
Target Server Version : 80028
File Encoding         : 65001

Date: 2022-03-06 08:19:23
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for reexamine_tb
-- ----------------------------
DROP TABLE IF EXISTS `reexamine_tb`;
CREATE TABLE `reexamine_tb` (
  `id` int NOT NULL AUTO_INCREMENT,
  `sid` int NOT NULL COMMENT '学生ID',
  `cid` varchar(4) NOT NULL COMMENT '课程ID',
  `score` tinyint DEFAULT NULL COMMENT '考试分数',
  `idx` tinyint DEFAULT NULL COMMENT '补考次序',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb3;

-- ----------------------------
-- Records of reexamine_tb
-- ----------------------------
INSERT INTO `reexamine_tb` VALUES ('1', '1001', 'c01', '56', '1');
INSERT INTO `reexamine_tb` VALUES ('2', '1001', 'c02', '69', '1');
INSERT INTO `reexamine_tb` VALUES ('3', '1002', 'c01', '59', '1');
INSERT INTO `reexamine_tb` VALUES ('4', '1002', 'c02', '54', '1');

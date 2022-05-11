/*
Navicat MySQL Data Transfer

Source Server         : yhk
Source Server Version : 80028
Source Host           : localhost:3306
Source Database       : niuke

Target Server Type    : MYSQL
Target Server Version : 80028
File Encoding         : 65001

Date: 2022-03-06 10:57:22
*/

SET FOREIGN_KEY_CHECKS=0;

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

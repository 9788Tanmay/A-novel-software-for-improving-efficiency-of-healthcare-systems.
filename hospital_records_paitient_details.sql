-- MySQL dump 10.13  Distrib 5.6.23, for Win64 (x86_64)
--
-- Host: localhost    Database: hospital_records
-- ------------------------------------------------------
-- Server version	5.7.18-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `paitient_details`
--

DROP TABLE IF EXISTS `paitient_details`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `paitient_details` (
  `Sr_no` int(11) NOT NULL AUTO_INCREMENT,
  `Registration_Id` varchar(15) DEFAULT NULL,
  `name` varchar(50) DEFAULT NULL,
  `Email` varchar(100) DEFAULT NULL,
  `sex` varchar(10) DEFAULT NULL,
  `Disease` varchar(255) DEFAULT NULL,
  `Age` int(11) DEFAULT NULL,
  PRIMARY KEY (`Sr_no`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `paitient_details`
--

LOCK TABLES `paitient_details` WRITE;
/*!40000 ALTER TABLE `paitient_details` DISABLE KEYS */;
INSERT INTO `paitient_details` VALUES (2,'181IT249asd','Tanmaiyh','tanmayhitman@gmail.com','Male','Hypertension',21),(3,'181RR296asd','Rahul','rajasthanroyals@gmail.com','Male','Typhoid',21),(4,'181IT234asd','Rahul','rahulraj@gmail.com','Male','Typhoid',21),(5,'Sushmita123','Sumita','sumita@gmail.com','Male','Hypothyroidism',21),(6,'Iamreactive123','Narendra','sodiumsir@gmail.com','Male','Hyperthyroidism',22),(7,'Raveena123','Raveena','raveenaagrawal@gmail.com','Female',' Migraine',21),(8,'Trevor123','Trevor','trevorbayliss@gmail.com','Male','GERD',24),(9,'Jofra1234','Jofra ','jofraecb@gmail.com','Male',NULL,21),(10,'Harshu1234','Harsh','harshu@gmail.com','Male','Impetigo',26);
/*!40000 ALTER TABLE `paitient_details` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-11-28  3:34:23

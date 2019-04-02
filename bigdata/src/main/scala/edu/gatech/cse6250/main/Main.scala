package edu.gatech.cse6250.main

import edu.gatech.cse6250.helper.SparkHelper
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.io.Source

object Main {
  def main(args: Array[String]) = {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    loadRawData(spark)

  }

  def loadRawData(spark: SparkSession) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val patientRecord = spark.read.format("csv").option("header", true).option("sep", "|").load("../training/p00001.psv")
    patientRecord.collect.foreach(println)
  }
}
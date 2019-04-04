package edu.gatech.cse6250.main

import edu.gatech.cse6250.helper.SparkHelper
import edu.gatech.cse6250.model.Record
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

    val result = loadRawData(spark)

  }

  def loadRawData(spark: SparkSession) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    val df = spark.read.format("csv").option("header", true).option("sep", "|").load("../training/p000*.psv")
    df.createOrReplaceTempView("patient_records")

    val tmp = sqlContext.sql("SELECT HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender FROM patient_records")
      .map(x => Record(x(0).toString.toDouble, x(1).toString.toDouble, x(2).toString.toDouble, x(3).toString.toDouble, x(4).toString.toDouble, x(5).toString.toDouble, x(6).toString.toDouble, x(7).toString.toInt, x(8).toString.toInt)).rdd

    val classifier = tmp.map(x => (x.age, x.gender)).distinct()
    classifier.collect.foreach(println)

    tmp
  }
}
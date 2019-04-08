package edu.gatech.cse6250.main

import java.text.SimpleDateFormat
import java.util.Calendar

import edu.gatech.cse6250.helper.SparkHelper
import edu.gatech.cse6250.model.ICURecord
import edu.gatech.cse6250.model.SOFA
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

    val (sepsis_infection, sofa_data) = loadRawData(spark)
    val result = processData(sepsis_infection, sofa_data)

    // generated csv is listed as "icustay_id", "hr", "starttime", "sofa"
    result.map { case (a, b, c, d) => a.toString + "," + b.toString + "," + c.toString.substring(0, c.toString.indexOf('.')) + "," + d.toString }.saveAsTextFile("sofa_timeline")
    println("Complete")
  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd HH:mm:ss"): java.sql.Timestamp = {
    val dateFormat = new SimpleDateFormat(pattern)
    new java.sql.Timestamp(dateFormat.parse(input).getTime)
  }

  def loadRawData(spark: SparkSession) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext

    // val df_icu = spark.read.format("csv").option("header", true).load("../icustays.csv").select($"icustay_id", $"intime", $"outtime")
    val df_sepsis = spark.read.format("csv").option("header", true).load("../sepsis3-df.csv").select($"icustay_id", $"intime", $"outtime", $"suspected_infection_time_poe")
    val df_sofa = spark.read.format("csv").option("header", true).load("../pivoted_sofa.csv").select($"icustay_id", $"hr", $"starttime", $"endtime", $"sofa_24hours")

    df_sepsis.createOrReplaceTempView("sepsis")
    val timewindow = sqlContext.sql("SELECT icustay_id, intime, outtime, suspected_infection_time_poe FROM sepsis WHERE suspected_infection_time_poe IS NOT NULL")
      .map(x => ICURecord(x(0).toString.toInt, sqlDateParser(x(1).toString), sqlDateParser(x(2).toString), sqlDateParser(x(3).toString))).rdd

    df_sofa.createOrReplaceTempView("sofa")
    val sofawindow = sqlContext.sql("SELECT icustay_id, hr, starttime, endtime, sofa_24hours FROM sofa WHERE sofa_24hours IS NOT NULL")
      .map(x => SOFA(x(0).toString.toInt, x(1).toString.toInt, sqlDateParser(x(2).toString), sqlDateParser(x(3).toString), x(4).toString.toInt)).rdd

    (timewindow, sofawindow)
  }

  def getStartTime(intime: java.sql.Timestamp, infection_time: java.sql.Timestamp) = {
    val calculated_ms = infection_time.getTime() - 2 * 24 * 3600 * 1000
    val observe_start_time = new java.sql.Timestamp(calculated_ms)
    if (observe_start_time.before(intime)) intime else observe_start_time
  }

  def getEndTime(outtime: java.sql.Timestamp, infection_time: java.sql.Timestamp) = {
    val calculated_ms = infection_time.getTime() + 1 * 24 * 3600 * 1000
    val observe_start_time = new java.sql.Timestamp(calculated_ms)
    if (observe_start_time.after(outtime)) outtime else observe_start_time
  }

  def processData(sepsis_infection: RDD[ICURecord], sofa_data: RDD[SOFA]) = {
    val observe_window = sepsis_infection.map(x => (x.icustay_id, (getStartTime(x.intime, x.suspected_infection_time), getEndTime(x.outtime, x.suspected_infection_time))))
    // println(observe_window)
    val data = sofa_data.map(x => (x.icustay_id, (x.hr, x.starttime, x.endtime, x.sofa))).join(observe_window)
      .filter(x => x._2._1._2.after(x._2._2._1) && x._2._1._3.before(x._2._2._2)).map(x => (x._1, x._2._1._1, x._2._1._2, x._2._1._4))
    data
  }
}
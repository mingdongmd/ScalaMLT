package scala.ml.supervised.nnet

/**
 * Copyright 2016 Mingdong.
 * Licensed under the LGPL.
 * A NARX adptive plant modeller demo.
 * email: mdwkmail@yeah.net
 * @author Mingdong
 * @version 0.0.1
 **/

import scala.collection._
import scala.collection.mutable._
import scala.annotation._
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.sys._
import scala.util.control._
import scala.util._
import scala.util.matching._
import scala.concurrent._

import breeze.linalg._
import breeze.plot._
import breeze.stats.distributions._

object SMLPDemo extends App {
  val rd=Uniform(-1.0,1.0)
  var net=SMLP( XinNum=20, YfbNum=5, hiddenNum=100, coupleNum=0, coupleInNum=10)
  net.learningRate=0.5
  net.initW("uniform", -0.01, 0.01, 1.0)
  
  /*net.tuningXh=false
  net.tuningY=false
  net.YtuningG=2.0
  net.YtuningK=2.0
  net.setXhtuningG(2.0)
  net.setXhtuningK(2.0)*/
  
  var y=0.0
  var plotU=ListBuffer[Double]()
  var plotX=ListBuffer[Double]()
  var plotY=ListBuffer[Double]()
  var plotD=ListBuffer[Double]()

  var desire=0.0
  var desire1=0.0
  var u=0.0
  var u1=0.0
  for(x<-0.0 to 15.0 by 0.01){
    val couple=0.2*math.sin(1.5*x)
    u1=u
    u=0.25*math.sin(0.5*x)
    desire= desire/(1+desire*desire)+u1*u1*u1+couple+0.02*rd.draw()
    net.putXin(u1)
    net.putCoupleIn(Array(couple))
    y=net.forward(desire)
    net.upDateW()
    if(x>10.0){
      plotU.add(u)
      plotX.add(x)
      plotY.add(y)
      plotD.add(desire)
    }
  }

  val f = Figure()
  val p = f.subplot(0)
  //f.visible=false
  p += plot(plotX, plotY,'.',"b")
  p += plot(plotX, plotD,'-',"r")
  p += plot(plotX, plotU,'+',"c")
  p.xlabel = "x axis"
  p.ylabel = "y axis"
  f.saveas("lines.png")
}

